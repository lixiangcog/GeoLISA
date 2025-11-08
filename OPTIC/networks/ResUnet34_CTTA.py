from torch import nn
import torch
import torch.nn.functional as F
from networks.resnet_TTA import resnet34, resnet50
from utils.convert import convert_encoder_to_target, convert_decoder_to_target, AdaBN
from utils.mix_anchor import *
import torchvision.transforms as T
from trainers import visual_anchorers
from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.utils import load_pretrained_weights, AverageMeter, MetricMeter
from my_dassl.optim import build_optimizer, build_lr_scheduler
from my_dassl.metrics import compute_accuracy
from clip import clip_clipping
from torch.cuda.amp import autocast
import os, time, datetime
import wandb

from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")  # or vit_h/l
sam.eval().cuda()


class SAManchorEncoder(nn.Module):
    def __init__(self, sam_model, out_dim=3):
        super().__init__()
        self.sam = sam_model
        self.project = nn.Conv2d(256, out_dim, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            sam_feats = self.sam.image_encoder(x)
        upsampled = F.interpolate(
            sam_feats, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.project(upsampled), None


class SaveFeatures:
    def __init__(self, m, n):
        self.features = None
        self.name = n
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        out = self.bn(F.relu(cat_p))
        return out


class ResUnet(nn.Module):
    def __init__(
        self,
        cfg,
        resnet="resnet34",
        num_classes=2,
        pretrained=False,
        convert=True,
        newBN=AdaBN,
        warm_n=5,
        freeze_backbone=True,
    ):
        super().__init__()
        if resnet == "resnet34":
            base_model = resnet34
            bottleneck = False
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == "resnet50":
            base_model = resnet50
            bottleneck = True
            feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise Exception("Only resnet34 or resnet50 supported")

        self.res = base_model(pretrained=pretrained)
        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

        self.newBN = newBN
        if convert:
            self.res = convert_encoder_to_target(
                self.res, newBN, 0, 5, False, bottleneck, warm_n
            )
            self.up1, self.up2, self.up3, self.up4, self.bnout = (
                convert_decoder_to_target(
                    [self.up1, self.up2, self.up3, self.up4, self.bnout],
                    newBN,
                    0,
                    5,
                    False,
                    warm_n,
                )
            )

        self.feature_hooks = []
        layers = [
            self.res.bn1,
            self.res.layer1,
            self.res.layer2,
            self.res.layer3,
            self.res.layer4,
        ]
        for i, layer in enumerate(layers):
            if i == 0:
                self.feature_hooks.append(SaveFeatures(layer, "first_bn"))
            else:
                for block in layer:
                    self.feature_hooks.append(SaveFeatures(block.bn1, f"{i}-bn1"))
                    self.feature_hooks.append(SaveFeatures(block.bn2, f"{i}-bn2"))
                    if resnet == "resnet50":
                        self.feature_hooks.append(SaveFeatures(block.bn3, f"{i}-bn3"))
                    if block.downsample:
                        self.feature_hooks.append(
                            SaveFeatures(block.downsample[1], f"{i}-downsample_bn")
                        )
        self.feature_hooks += [
            SaveFeatures(b.bn, f"{i}-up_bn")
            for i, b in enumerate([self.up1, self.up2, self.up3, self.up4], 1)
        ]
        self.feature_hooks.append(SaveFeatures(self.bnout, "last_bn"))

        self.dtype = torch.float32
        self.p_eps = 1.0
        self.coordinator = visual_anchorers.__dict__[cfg.TRAINER.BLACKVIP.METHOD](cfg)

        if freeze_backbone:
            for name, params in self.named_parameters():
                if "coordinator" in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False

    def forward(self, x):
        anchor, _ = self.coordinator(x.type(self.dtype))
        x = x + self.p_eps * anchor
        x, sfs = self.res(x)
        x = F.relu(x)
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))
        seg_output = self.seg_head(head_input)
        return seg_output, sfs, head_input

    def close(self):
        for sf in self.sfs:
            sf.remove()

    def save_data_anchor(self, save_path="anchor_vis/anchor.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anchor = self.data_anchor.detach().cpu()
        anchor = (anchor - anchor.min()) / (
            anchor.max() - anchor.min()
        )  # normalize to [0,1]
        to_pil = T.ToPILImage()
        img = to_pil(anchor)
        img.save(save_path)
        print(f"[anchor] Saved to {save_path}")

    def save_data_anchor_tensor(self):
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anchor = self.data_anchor.detach().cpu()
        anchor = (anchor - anchor.min()) / (
            anchor.max() - anchor.min()
        )  # normalize to [0,1]
        return anchor


@TRAINER_REGISTRY.register()
class BLACKVIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        self.model = ResUnet(
            cfg=cfg,
            resnet="resnet34",
            num_classes=cfg.DATASET.NUM_CLASSES,
            pretrained=False,
        )
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.coordinator.dec, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.coordinator.dec, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "coordinator", self.model.coordinator.dec, self.optim, self.sched
        )
        self.N_params = len(
            torch.nn.utils.parameters_to_vector(self.model.coordinator.dec.parameters())
        )
        self.o, self.c, self.a, self.alpha, self.gamma = (
            cfg.TRAINER.BLACKVIP.SPSA_PARAMS
        )
        self.opt_type = cfg.TRAINER.BLACKVIP.OPT_TYPE
        self.b1 = cfg.TRAINER.BLACKVIP.MOMS
        self.sp_avg = cfg.TRAINER.BLACKVIP.SP_AVG
        self.step = 0
        self.m1 = 0
        self.loss_fn = F.cross_entropy

    def forward_backward(self, batch):
        with torch.no_grad():
            image, label = self.parse_batch_train(batch)
            with autocast():
                ak = self.a / ((self.step + self.o) ** self.alpha)
                ck = self.c / (self.step**self.gamma)
                w = torch.nn.utils.parameters_to_vector(
                    self.model.coordinator.dec.parameters()
                )
                ghat, loss, acc = self.spsa_grad_estimate_bi(w, image, label, ck)
                if self.opt_type == "spsa-gc":
                    self.m1 = self.b1 * self.m1 + ghat if self.step > 1 else ghat
                    accum_ghat = ghat + self.b1 * self.m1
                elif self.opt_type == "spsa":
                    accum_ghat = ghat
                else:
                    raise ValueError
                w_new = w - ak * accum_ghat
                torch.nn.utils.vector_to_parameters(
                    w_new, self.model.coordinator.dec.parameters()
                )
        if self.cfg.use_wandb:
            wandb.log(
                {"train_ep_acc": acc, "train_ep_loss": loss.item(), "gain_seq": ak}
            )
        return {"loss": loss, "acc": acc}

    def spsa_grad_estimate_bi(self, w, image, label, ck):
        ghats = []
        for _ in range(self.sp_avg):
            p_side = (torch.rand(self.N_params).reshape(-1, 1) + 1) / 2
            samples = torch.cat([p_side, -p_side], dim=1)
            perturb = (
                torch.gather(
                    samples, 1, torch.bernoulli(torch.ones_like(p_side) / 2).long()
                )
                .reshape(-1)
                .cuda()
            )
            w_r, w_l = w + ck * perturb, w - ck * perturb
            torch.nn.utils.vector_to_parameters(
                w_r, self.model.coordinator.dec.parameters()
            )
            output1, _, _ = self.model(image)
            torch.nn.utils.vector_to_parameters(
                w_l, self.model.coordinator.dec.parameters()
            )
            output2, _, _ = self.model(image)
            loss1, loss2 = self.loss_fn(output1, label), self.loss_fn(output2, label)
            ghat = (loss1 - loss2) / ((2 * ck) * perturb)
            ghats.append(ghat.reshape(1, -1))
        ghat = torch.cat(ghats, dim=0).mean(dim=0) if self.sp_avg > 1 else ghats[0]
        acc = (
            (compute_accuracy(output1, label)[0] + compute_accuracy(output2, label)[0])
            / 2
        ).item()
        return ghat, (loss1 + loss2) / 2, acc

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses, batch_time, data_time = MetricMeter(), AverageMeter(), AverageMeter()
        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.step += 1
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            if (
                (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
                or self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            ):
                eta_seconds = batch_time.avg * (
                    self.num_batches
                    - self.batch_idx
                    - 1
                    + (self.max_epoch - self.epoch - 1) * self.num_batches
                )
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}",
                ]
                print(" ".join(info))
            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()

    def after_epoch(self):
        if (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 or (
            self.epoch + 1
        ) == self.max_epoch:
            self.save_model(self.epoch, self.output_dir)

    def after_train(self):
        print("Finish training")
        elapsed = round(time.time() - self.time_start)
        print(f"Elapsed: {str(datetime.timedelta(seconds=elapsed))}")
        self.close_writer()
