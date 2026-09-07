import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic
from models.segment_anything.build_sam import build_sam, sam_model_registry
from models.segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter
from unet_parts import *
import clip
from model import UNet_Attention_Transformer_Multiscale
from open_clip import create_model_from_pretrained, get_tokenizer


def print_memory_usage(prefix=""):
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(
        f"{prefix} Allocated: {allocated / 1024**2:.2f} MB, Cached: {cached / 1024**2:.2f} MB"
    )


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()

        # First convolutional block to reduce the channel dimension
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional block to further reduce the channel dimension
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Final convolutional layer to reduce the channels to 1 (for the mask output)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        # Sigmoid activation for mask output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply convolutions with ReLU activations
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Final convolution to get the mask with a single channel
        x = self.conv4(x)

        # Apply sigmoid to get a mask in the range [0, 1]
        x = self.sigmoid(x)

        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SegDecoderLinear(nn.Module):
    def __init__(
        self,
        num_classes=1,
        emb_dim=32,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(emb_dim // 4),
            activation(),
            nn.ConvTranspose2d(emb_dim // 4, emb_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(emb_dim // 8, emb_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim // 8, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.output_upscaling(x)
        out = self.final(out)
        return out


class SegDecoderCNN(nn.Module):
    def __init__(
        self,
        num_classes=1,
        embed_dim=32,
        num_depth=2,
        top_channel=64,
    ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(embed_dim, top_channel * 2**num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                top_channel * 2**num_depth, top_channel * 2**num_depth, kernel_size=1
            ),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(
                        top_channel * 2 ** (num_depth - i),
                        top_channel * 2 ** (num_depth - i),
                        3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        top_channel * 2 ** (num_depth - i),
                        top_channel * 2 ** (num_depth - i - 1),
                        3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        top_channel * 2 ** (num_depth - i),
                        top_channel * 2 ** (num_depth - i),
                        3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        top_channel * 2 ** (num_depth - i),
                        top_channel * 2 ** (num_depth - i),
                        3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        top_channel * 2 ** (num_depth - i),
                        top_channel * 2 ** (num_depth - i - 1),
                        2,
                        stride=2,
                    ),
                )

            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel, top_channel, 3, padding=1),
            nn.Conv2d(top_channel, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks))
            )  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As, self.w_Bs = [], []  # These are linear layers

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for param in sam_model.anchor_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            # if t_layer_i not in self.lora_layer: continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

        # self.decoder = SimpleDecoder()
        self.seg_decoder = SegDecoderCNN(num_classes=1)

        # self.unet = UNet_Attention_Transformer_Multiscale(256, 256)
        # self.unet.to("cuda")
        text_feature_dim = 512

        feature_map_channels_list = [256, 512, 1024, 2048]

        self.text_proj_layers = nn.ModuleList(
            [
                nn.Linear(text_feature_dim, fm_channels)
                for fm_channels in feature_map_channels_list
            ]
        )
        self.downtext_layers = nn.ModuleList(
            [
                nn.Linear(text_feature_dim, text_feature_dim)  # 或者使用 nn.Conv1d
                for _ in range(len(feature_map_channels_list))
            ]
        )

        # self.clip_model, _ = clip.load("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", device='cpu')
        self.downtext = nn.AvgPool1d(kernel_size=2, stride=2)

        # Define multi-scale anchors
        self.anchors = [
            "The optic disc (OD) and optic cup (OC) are positioned together in the inferior region, with the OD located near the lower margin of the fundus.",
            "The optic cup (OC) occupies approximately 50% of the optic disc (OD) diameter, indicating a moderate cup-to-disc ratio.",
            "The boundary between the optic cup (OC) and the optic disc (OD) is clearly visible, with distinct structural separation.",
            "The optic disc (OD) is centrally oriented within the fundus, and the optic cup (OC) remains contained within the disc.",
            "Mild peripapillary atrophy is present around the optic cup (OC), influencing the margin appearance of the optic disc (OD) and causing slight vessel displacement.",
            "The optic cup (OC) exhibits a relatively large cup-to-disc ratio within the optic disc (OD), which may suggest a glaucomatous structural tendency.",
        ]


        self.clip_model, preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        # self.clip_model.to("cuda")
        self.clip_model.eval()
        self.tokenizer = get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.seg_decoder.parameters():
            param.requires_grad = True

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        anchor_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save anchor encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
            self.sam, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if "anchor_encoder" in key:
                anchor_encoder_tensors[key] = value
            if "mask_decoder" in key:
                mask_decoder_tensors[key] = value

        merged_dict = {
            **a_tensors,
            **b_tensors,
            **anchor_encoder_tensors,
            **mask_decoder_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load anchor encoder
        anchor_encoder_keys = [k for k in sam_keys if "anchor_encoder" in k]
        anchor_encoder_values = [state_dict[k] for k in anchor_encoder_keys]
        anchor_encoder_new_state_dict = {
            k: v for k, v in zip(anchor_encoder_keys, anchor_encoder_values)
        }
        sam_dict.update(anchor_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if "mask_decoder" in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {
            k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)
        }
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(
        self, images, original_size, point_coords, point_labels, multimask_output=False
    ):
        """
        Forward function.

        Arguments:
            images: The input images as a torch tensor of shape (batch_size, 3, H, W).
            original_size: A tuple (H, W) representing the original size of the images.
            point_coords: The coordinates of the points as a tensor.
            point_labels: The labels of the points as a tensor.
            multimask_output: Whether to output multiple masks.

        Returns:
            The predicted masks.
        """

        # Preprocess images using SAM's preprocess
        input_images = self.sam.preprocess(images)  # Shape: (batch_size, 3, H, W)

        # Extract feature maps using SAM's ResNet backbone
        feature_maps_in = self.sam.resnet(
            input_images
        )  # List of feature maps from ResNet encoder

        # Encode each anchor corresponding to different scales
        text_features = []
        for anchor in self.anchors:
            # Tokenize and encode the anchor
            text = self.tokenizer([anchor]).to(images.device)
            text_feature = self.clip_model.encode_text(text)  # Shape: [1, 512]
            text_feature = (
                text_feature.detach()
            )  # Detach to avoid backpropagation to CLIP
            text_feature = (text_feature - 0.015) / 0.27  # Normalize (adjust as needed)
            text_features.append(text_feature)

        # Integrate text features with image feature maps at different scales
        for idx, fm in enumerate(feature_maps_in):
            if idx >= len(self.text_proj_layers):
                break  # Avoid index out of range if feature maps exceed projection layers

            # Project the corresponding text feature
            projected_text_feat = self.text_proj_layers[idx](
                text_features[idx]
            )  # Shape: [1, fm_channels]

            # Reshape and expand to match the spatial dimensions of the feature map
            projected_text_feat = projected_text_feat.unsqueeze(-1).unsqueeze(
                -1
            )  # [1, fm_channels, 1, 1]
            projected_text_feat = projected_text_feat.expand(
                -1, -1, fm.shape[2], fm.shape[3]
            )  # [1, fm_channels, H, W]

            # Fuse the projected text feature with the image feature map
            # print("projectioned_text_feat.shape", projected_text_feat.shape)
            # projectioned_text_feat.shape torch.Size([1, 256, 256, 256])
            # projectioned_text_feat.shape torch.Size([1, 512, 128, 128])
            # projectioned_text_feat.shape torch.Size([1, 1024, 64, 64])
            # projectioned_text_feat.shape torch.Size([1, 2048, 32, 32])
            feature_maps_in[idx] = fm * projected_text_feat

            # Optionally downsample text features for subsequent scales
            if idx < len(self.downtext_layers):
                text_features[idx] = self.downtext_layers[idx](text_features[idx])

        # Initialize list to store decoder feature maps
        feature_maps_out = [None] * 3

        # Process input images through SAM's image encoder patch embedding
        input_images = self.sam.image_encoder.forward_patch_embed(input_images)

        # Iterate over the layers
        for i in range(len(self.sam.global_index)):
            # Process blocks within each stage
            for j in range(2):
                block_idx = i * 3 + j
                input_images = self.sam.image_encoder.forward_block(
                    input_images, block_idx
                )

            # Integrate feature maps from ResNet via adapters_in
            current_feature_map = self.sam.adapters_in[i](feature_maps_in[i])
            current_feature_map = current_feature_map.permute(
                0, 2, 3, 1
            )  # Adjust dimensions
            input_images = input_images + current_feature_map  # Fusion of features

            # Process the global block
            input_images = self.sam.image_encoder.forward_block(
                input_images, self.sam.global_index[i]
            )

            # Collect feature maps for decoder if not in the last stage
            if i < len(self.sam.global_index) - 1:
                permuted_input_images = input_images.permute(0, 3, 1, 2)
                idx = (
                    len(self.sam.global_index) - i - 2
                )  # Reverse order for U-Net structure
                current_out_feature_map = self.sam.adapters_bridge[idx](
                    permuted_input_images
                )
                feature_maps_out[idx] = current_out_feature_map

        # Obtain image embeddings from the encoder's neck
        image_embeddings = self.sam.image_encoder.forward_neck(input_images)

        # Generate multi-scale features
        multi_scale_feature = self.sam.embedding_encoder(image_embeddings)
        for fm in feature_maps_out:
            if fm is not None:
                multi_scale_feature += fm

        # print("multi_scale_feature.shape", multi_scale_feature.shape)
        # low_res_masks = self.decoder(multi_scale_feature)

        # # Get sparse and dense embeddings from the anchor encoder
        # if point_coords is not None:
        #     points = (point_coords, point_labels)
        # else:
        #     points = None

        # sparse_embeddings, dense_embeddings = self.sam.anchor_encoder(
        #     points=points,
        #     boxes=None,
        #     masks=None,
        # )
        # # sparse_embeddings_none, dense_embeddings_none = sam.anchor_encoder(points=None, boxes=None, masks=None)

        # # Decode masks using the mask decoder
        # low_res_masks = self.sam.mask_decoder(
        #     image_embeddings=image_embeddings,
        #     image_pe=self.sam.anchor_encoder.get_dense_pe(),
        #     sparse_anchor_embeddings=sparse_embeddings,
        #     dense_anchor_embeddings=dense_embeddings,
        #     multi_scale_feature=multi_scale_feature,
        # )
        # print("low_res_masks.shape", low_res_masks.shape)

        masks = self.seg_decoder(multi_scale_feature)

        # Post-process masks to original size
        # masks = self.sam.postprocess_masks(
        #     low_res_masks, images.shape[-2:], original_size
        # )
        # print(masks.shape)
        # multi_scale_feature.shape torch.Size([1, 32, 256, 256])
        # low_res_masks.shape torch.Size([1, 1, 256, 256])
        # torch.Size([1, 1, 1024, 1024])

        return nn.Sigmoid()(masks)


# if __name__ == "__main__":
#     sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
#     lora_sam = LoRA_Sam(sam, 4)
#     lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
