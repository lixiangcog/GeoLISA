import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.metrics import calculate_metrics
from networks.ResUnet_TTA import ResUnet
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from gradcam import GradCAM, GradCAMpp
from utils_grad import visualize_cam, Normalize
import PIL
from torchvision.utils import make_grid, save_image


torch.set_num_threads(1)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


class VPTTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, "VPTTA")
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + ".log")
        sys.stdout = Logger(log_path, sys.stdout)
        self.segmentation_loss_fn = DiceLoss().to("cuda")
        self.lambda_bn = 0.3
        self.lambda_seg = 0.7

        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            if target != "REFUGE_Valid":
                target_test_csv.append(target + "_train.csv")
                target_test_csv.append(target + "_test.csv")
            else:
                target_test_csv.append(target + ".csv")
        ts_img_list, ts_label_list, pseudo_label_list = convert_labeled_list(
            config.dataset_root, target_test_csv
        )
        target_test_dataset = OPTIC_dataset(
            config.dataset_root,
            ts_img_list,
            ts_label_list,
            pseudo_label_list,
            config.image_size,
            img_normalize=True,
        )
        self.target_test_loader = DataLoader(
            dataset=target_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_wo_transform,
            num_workers=config.num_workers,
        )
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(
            config.model_root, str(config.Source_Dataset)
        )  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device

        # Warm-up
        self.warm_n = config.warm_n
        self.iters = config.iters

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print("***" * 20)

    def build_model(self):
        self.model = ResUnet(
            resnet=self.backbone,
            num_classes=self.out_ch,
            pretrained=False,
            newBN=AdaBN,
            warm_n=self.warm_n,
        ).to(self.device)

        checkpoint = torch.load(
            os.path.join(self.load_model, "last-Res_Unet.pth"), weights_only=True
        )

        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        self.model.load_state_dict(model_dict, strict=False)

        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                trainable_params.append(param)

        if self.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "Adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

    def run(self):
        metric_dict = ["Disc_Dice", "Disc_ASD", "Cup_Dice", "Cup_ASD"]

        # Valid on Target
        metrics_test = [[], [], [], []]

        for pass_num in range(3):

            for batch, data in enumerate(
                tqdm(self.target_test_loader, desc="Processing batches", ncols=100)
            ):
                x, y, pseudo_label = data["data"], data["mask"], data["pseudo_mask"]
                x = torch.from_numpy(x).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.long)
                pseudo_label = torch.from_numpy(pseudo_label).to(dtype=torch.long)

                x, y, pseudo_label = (
                    Variable(x).to(self.device),
                    Variable(y).to(self.device),
                    Variable(pseudo_label).to(self.device),
                )

                self.model.eval()
                self.model.change_BN_status(new_sample=True)

                for tr_iter in range(self.iters):
                    pred_logit, fea, head_input = self.model(x)
                    times, bn_loss = 0, 0
                    for nm, m in self.model.named_modules():
                        if isinstance(m, AdaBN):
                            bn_loss += m.bn_loss
                            times += 1
                    bn_loss = bn_loss / times

                    seg_loss = self.segmentation_loss_fn(pred_logit, pseudo_label)

                    loss = self.lambda_bn * bn_loss + self.lambda_seg * seg_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.model.change_BN_status(new_sample=False)

                # Inference
                self.model.eval()
                with torch.no_grad():
                    pred_logit, fea, head_input = self.model(x)

                # Calculate the metrics
                seg_output = torch.sigmoid(pred_logit)
                metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
                for i in range(len(metrics)):
                    assert isinstance(
                        metrics[i], list
                    ), "The metrics value is not list type."
                    metrics_test[i] += metrics[i]

                def get_next_filename(base_dir, prefix, ext=".png"):
                    os.makedirs(base_dir, exist_ok=True)
                    existing = [
                        int(f[len(prefix) : -len(ext)])
                        for f in os.listdir(base_dir)
                        if f.startswith(prefix)
                        and f.endswith(ext)
                        and f[len(prefix) : -len(ext)].isdigit()
                    ]
                    next_index = max(existing, default=0) + 1
                    return os.path.join(base_dir, f"{prefix}{next_index}{ext}")

                # from torchvision.utils import save_image

                # input_tensor = x.clone().detach().cpu()  # shape: [B, C, H, W]

                # # 保存路径
                # img_base = get_next_filename("image_vis", "domainA_anchor_")

                # # 保存单张图像（仅 batch[0]）
                # save_image(input_tensor[0], img_base)

                # # 使用方法
                # save_path = get_next_filename("anchor_vis", "domainA_anchor_")
                # self.model.save_data_anchor(save_path)

                # cam_dict = dict()

                # resnet_model_dict = dict(type='resnet', arch=self.model, layer_name='layer4', input_size=(512, 512))
                # resnet_gradcam = GradCAM(resnet_model_dict, True)
                # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
                # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

                # # print(cam_dict)

                # img_dir = '/home/lixiang/PLASTIC/OPTIC/image_vis'
                # # img_name = 'collies.JPG'
                # # img_name = 'multiple_dogs.jpg'
                # # img_name = 'snake.JPEG'
                # img_name = 'domainA_anchor_4176.png'
                # img_path = os.path.join(img_dir, img_name)

                # pil_img = PIL.Image.open(img_path)

                # normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
                # torch_img = F.upsample(torch_img, size=(512, 512), mode='bilinear', align_corners=False)
                # normed_torch_img = normalizer(torch_img)

                # images = []
                # for gradcam, gradcam_pp in cam_dict.values():
                #     # print(normed_torch_img)
                #     mask, _ = gradcam(normed_torch_img)
                #     heatmap, result = visualize_cam(mask.cpu(), torch_img.cpu())

                #     mask_pp, _ = gradcam_pp(normed_torch_img)
                #     heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_img.cpu())

                #     images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

                # images = make_grid(torch.cat(images, 0), nrow=5)

                # output_dir = 'outputs'
                # os.makedirs(output_dir, exist_ok=True)
                # output_name = img_name
                # output_path = os.path.join(output_dir, output_name)

                # save_image(images, output_path)
                # PIL.Image.open(output_path)

            test_metrics_y = np.mean(metrics_test, axis=1)
            print_test_metric_mean = {}
            for i in range(len(test_metrics_y)):
                print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
            print("Test Metrics: ", print_test_metric_mean)
            print(
                "Mean Dice:",
                (
                    print_test_metric_mean["Disc_Dice"]
                    + print_test_metric_mean["Cup_Dice"]
                )
                / 2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument(
        "--Source_Dataset",
        type=str,
        default="RIM_ONE_r3",
        help="RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS",
    )
    parser.add_argument("--Target_Dataset", type=list)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)

    # Model
    parser.add_argument(
        "--backbone", type=str, default="resnet34", help="resnet34/resnet50"
    )
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=2)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="Adam", help="SGD/Adam")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.99)  # momentum in SGD
    parser.add_argument("--beta1", type=float, default=0.9)  # beta1 in Adam
    parser.add_argument("--beta2", type=float, default=0.99)  # beta2 in Adam.
    parser.add_argument("--weight_decay", type=float, default=0.00)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)

    parser.add_argument("--warm_n", type=int, default=5)

    # Path
    parser.add_argument("--path_save_log", type=str, default="./logs")
    parser.add_argument("--model_root", type=str, default="./models")
    parser.add_argument(
        "--dataset_root", type=str, default="/media/userdisk0/zychen/Datasets/Fundus"
    )

    # Cuda (default: the first available device)
    parser.add_argument("--device", type=str, default="cuda:0")

    config = parser.parse_args()

    config.Target_Dataset = [
        "RIM_ONE_r3",
        "REFUGE",
        "ORIGA",
        "REFUGE_Valid",
        "Drishti_GS",
    ]
    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = VPTTA(config)
    TTA.run()
