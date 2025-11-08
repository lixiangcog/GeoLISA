#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F
import numpy as np  # 确保导入了 numpy
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from torchvision import transforms
from utils.Utils import *
from utils.metrics import *
from networks.deeplabv3 import DeepLab
import cv2
import torch.backends.cudnn as cudnn
import random
import logging
import sys
import tqdm
from matplotlib.colors import ListedColormap

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus Segmentation Inference Script with Visualization"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="./logs/source/source_model.pth.tar",
        help="Path to the model file",
    )
    parser.add_argument("--dataset", type=str, default="Domain1", help="Dataset name")
    parser.add_argument(
        "--batchsize", type=int, default=8, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--source", type=str, default="Domain3", help="Source domain name"
    )
    parser.add_argument("-g", "--gpu", type=int, default=1, help="GPU ID to use")
    parser.add_argument(
        "--data-dir", type=str, default="Fundus", help="Base directory for data"
    )
    parser.add_argument(
        "--out-stride", type=int, default=16, help="Output stride for DeepLab"
    )
    parser.add_argument(
        "--save-root-ent",
        type=str,
        default="./results/ent/",
        help="Save root for entropy maps",
    )
    parser.add_argument(
        "--save-root-mask",
        type=str,
        default="./results/mask/",
        help="Save root for masks",
    )
    parser.add_argument(
        "--test-prediction-save-path",
        type=str,
        default="./results/baseline/",
        help="Path to save test predictions",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=10,
        help="Number of forward passes for uncertainty estimation",
    )
    parser.add_argument(
        "--seed", type=int, default=3377, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Enable saving of visualization images",
    )  # 新增参数
    parser.add_argument(
        "--vis-save-dir",
        type=str,
        default="./results/visualizations/",
        help="Directory to save visualization images",
    )  # 新增参数
    return parser.parse_args()


def create_color_map(num_classes):
    """
    创建一个颜色映射表，用于将类别标签转换为颜色。

    参数:
        num_classes (int): 类别数。

    返回:
        color_map (numpy.ndarray): 颜色映射表，形状为 (num_classes, 3)。
    """
    print("num_classes:", num_classes)
    if num_classes == 2:
        # 例如：背景为黑色，目标为红色
        color_map = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)  # 背景  # 目标
    else:
        # 使用matplotlib的'jet'颜色映射生成多类别颜色
        cmap = plt.cm.get_cmap("jet", num_classes)
        # 生成颜色映射表，并将颜色值从[0,1]缩放到[0,255]
        color_map = (cmap(np.linspace(0, 1, num_classes))[:, :3] * 255).astype(np.uint8)
        print(color_map)

    return color_map


def overlay_mask_on_image(image, mask, color_map):
    """
    将掩码叠加在原始图像上。

    参数:
        image (numpy.ndarray): 原始图像，形状为 (H, W, 3)。
        mask (numpy.ndarray): 掩码，形状为 (H, W)，类别标签。
        color_map (numpy.ndarray): 颜色映射表，形状为 (num_classes, 3)。

    返回:
        overlaid_image (numpy.ndarray): 叠加后的图像，形状为 (H, W, 3)。
    """
    colored_mask = color_map[mask]
    overlaid_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlaid_image


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 日志记录
    logger.info("Loading model from %s", args.model_file)

    # 1. 数据集
    composed_transforms_test = transforms.Compose(
        [tr.Resize(512), tr.Normalize_tf(), tr.ToTensor()]
    )
    db_train = DL.FundusSegmentation(
        base_dir=args.data_dir,
        dataset=args.dataset,
        split="train/ROIs",
        transform=composed_transforms_test,
    )
    db_test = DL.FundusSegmentation(
        base_dir=args.data_dir,
        dataset=args.dataset,
        split="test/ROIs",
        transform=composed_transforms_test,
    )
    db_source = DL.FundusSegmentation(
        base_dir=args.data_dir,
        dataset=args.source,
        split="train/ROIs",
        transform=composed_transforms_test,
    )

    train_loader = DataLoader(
        db_train, batch_size=args.batchsize, shuffle=False, num_workers=1
    )
    test_loader = DataLoader(
        db_test, batch_size=args.batchsize, shuffle=False, num_workers=1
    )
    source_loader = DataLoader(
        db_source, batch_size=args.batchsize, shuffle=False, num_workers=1
    )

    # 2. 模型
    model = DeepLab(
        num_classes=2,
        backbone="mobilenet",
        output_stride=args.out_stride,
        sync_bn=True,
        freeze_bn=False,
    )

    if torch.cuda.is_available():
        model = model.cuda()
    try:
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        logger.error("Model file not found: %s", args.model_file)
        sys.exit(1)
    except KeyError:
        logger.error('Key "model_state_dict" not found in checkpoint.')
        sys.exit(1)

    model.eval()  # 设置为评估模式

    # 初始化字典
    pseudo_label_dic = {}
    uncertain_dic = {}
    proto_pseudo_dic = {}
    distance_0_obj_dic = {}
    distance_0_bck_dic = {}
    distance_1_bck_dic = {}
    distance_1_obj_dic = {}
    centroid_0_obj_dic = {}
    centroid_0_bck_dic = {}
    centroid_1_obj_dic = {}
    centroid_1_bck_dic = {}

    # 如果启用了可视化，创建保存目录
    if args.save_visualization:
        vis_save_dir = args.vis_save_dir
        os.makedirs(vis_save_dir, exist_ok=True)
        original_save_dir = osp.join(vis_save_dir, "original")
        pseudo_label_save_dir = osp.join(vis_save_dir, "pseudo_labels")
        overlay_save_dir = osp.join(vis_save_dir, "overlay")
        os.makedirs(original_save_dir, exist_ok=True)
        os.makedirs(pseudo_label_save_dir, exist_ok=True)
        os.makedirs(overlay_save_dir, exist_ok=True)
        color_map = create_color_map(num_classes=3)  # 根据类别数调整

    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader), ncols=80, leave=False
        ):
            data, target, img_name = sample["image"], sample["map"], sample["img_name"]
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # 无需使用 Variable
            # data, target = Variable(data), Variable(target)

            preds = torch.zeros(
                [args.num_passes, data.shape[0], 2, data.shape[2], data.shape[3]],
                device="cuda",
            )
            features = torch.zeros(
                [args.num_passes, data.shape[0], 305, 128, 128], device="cuda"
            )  # 305 可以根据模型输出动态调整
            for i in range(args.num_passes):
                p, _, f = model(data)
                preds[i, ...] = p
                features[i, ...] = f
            preds1 = torch.sigmoid(preds)
            preds = torch.sigmoid(preds / 2.0)
            std_map = torch.std(preds, dim=0)

            prediction = torch.mean(preds1, dim=0)
            pseudo_label = prediction.clone()

            # 定义阈值
            low_threshold = 0.25
            high_threshold = 0.75

            # 创建掩码
            mask_high = pseudo_label > high_threshold
            mask_middle = (pseudo_label > low_threshold) & (
                pseudo_label <= high_threshold
            )
            mask_low = pseudo_label <= low_threshold

            # 赋值
            pseudo_label[mask_high] = 1.0
            pseudo_label[mask_middle] = 0.5
            pseudo_label[mask_low] = 0.0

            feature = torch.mean(features, dim=0)

            target_0_obj = F.interpolate(
                pseudo_label[:, 0:1, ...], size=feature.size()[2:], mode="nearest"
            )
            target_1_obj = F.interpolate(
                pseudo_label[:, 1:, ...], size=feature.size()[2:], mode="nearest"
            )
            prediction_small = F.interpolate(
                prediction, size=feature.size()[2:], mode="bilinear", align_corners=True
            )
            std_map_small = F.interpolate(
                std_map, size=feature.size()[2:], mode="bilinear", align_corners=True
            )
            target_0_bck = 1.0 - target_0_obj
            target_1_bck = 1.0 - target_1_obj

            # 计算掩码，保持为布尔类型
            mask_0_obj = std_map_small[:, 0:1, ...] < 0.05
            mask_0_bck = std_map_small[:, 0:1, ...] < 0.05
            mask_1_obj = std_map_small[:, 1:, ...] < 0.05
            mask_1_bck = std_map_small[:, 1:, ...] < 0.05

            # 执行按位或操作
            mask_0 = mask_0_obj | mask_0_bck
            mask_1 = mask_1_obj | mask_1_bck

            # 将掩码转换为浮点数以用于后续操作
            mask_0 = mask_0.float()
            mask_1 = mask_1.float()
            mask = torch.cat((mask_0, mask_1), dim=1)

            # 继续后续操作
            feature_0_obj = feature * target_0_obj * mask_0_obj.float()
            feature_1_obj = feature * target_1_obj * mask_1_obj.float()
            feature_0_bck = feature * target_0_bck * mask_0_bck.float()
            feature_1_bck = feature * target_1_bck * mask_1_bck.float()

            # 计算质心
            centroid_0_obj = torch.sum(
                feature_0_obj * prediction_small[:, 0:1, ...],
                dim=[0, 2, 3],
                keepdim=True,
            )
            centroid_1_obj = torch.sum(
                feature_1_obj * prediction_small[:, 1:, ...],
                dim=[0, 2, 3],
                keepdim=True,
            )
            centroid_0_bck = torch.sum(
                feature_0_bck * (1.0 - prediction_small[:, 0:1, ...]),
                dim=[0, 2, 3],
                keepdim=True,
            )
            centroid_1_bck = torch.sum(
                feature_1_bck * (1.0 - prediction_small[:, 1:, ...]),
                dim=[0, 2, 3],
                keepdim=True,
            )

            # 计算计数，避免除以零
            target_0_obj_cnt = torch.sum(
                mask_0_obj * target_0_obj * prediction_small[:, 0:1, ...],
                dim=[0, 2, 3],
                keepdim=True,
            )
            target_1_obj_cnt = torch.sum(
                mask_1_obj * target_1_obj * prediction_small[:, 1:, ...],
                dim=[0, 2, 3],
                keepdim=True,
            )
            target_0_bck_cnt = torch.sum(
                mask_0_bck * target_0_bck * (1.0 - prediction_small[:, 0:1, ...]),
                dim=[0, 2, 3],
                keepdim=True,
            )
            target_1_bck_cnt = torch.sum(
                mask_1_bck * target_1_bck * (1.0 - prediction_small[:, 1:, ...]),
                dim=[0, 2, 3],
                keepdim=True,
            )

            # 计算质心，添加 epsilon 避免除以零
            epsilon = 1e-8
            centroid_0_obj = torch.divide(centroid_0_obj, target_0_obj_cnt + epsilon)
            centroid_1_obj = torch.divide(centroid_1_obj, target_1_obj_cnt + epsilon)
            centroid_0_bck = torch.divide(centroid_0_bck, target_0_bck_cnt + epsilon)
            centroid_1_bck = torch.divide(centroid_1_bck, target_1_bck_cnt + epsilon)

            # 计算距离
            distance_0_obj = torch.sum(
                (feature - centroid_0_obj) ** 2, dim=1, keepdim=True
            )
            distance_0_bck = torch.sum(
                (feature - centroid_0_bck) ** 2, dim=1, keepdim=True
            )
            distance_1_obj = torch.sum(
                (feature - centroid_1_obj) ** 2, dim=1, keepdim=True
            )
            distance_1_bck = torch.sum(
                (feature - centroid_1_bck) ** 2, dim=1, keepdim=True
            )

            # 生成 proto_pseudo 标签
            proto_pseudo_0 = (distance_0_obj < distance_0_bck).float()
            proto_pseudo_1 = (distance_1_obj < distance_1_bck).float()
            proto_pseudo = torch.cat((proto_pseudo_0, proto_pseudo_1), dim=1)
            proto_pseudo = F.interpolate(
                proto_pseudo, size=data.size()[2:], mode="nearest"
            )

            # 转换为 numpy
            pseudo_label_np = pseudo_label.detach().cpu().numpy()
            std_map_np = std_map.detach().cpu().numpy()
            proto_pseudo_np = proto_pseudo.detach().cpu().numpy()
            distance_0_obj_np = distance_0_obj.detach().cpu().numpy()
            distance_0_bck_np = distance_0_bck.detach().cpu().numpy()
            distance_1_obj_np = distance_1_obj.detach().cpu().numpy()
            distance_1_bck_np = distance_1_bck.detach().cpu().numpy()
            centroid_0_obj_np = centroid_0_obj.detach().cpu().numpy()
            centroid_0_bck_np = centroid_0_bck.detach().cpu().numpy()
            centroid_1_obj_np = centroid_1_obj.detach().cpu().numpy()
            centroid_1_bck_np = centroid_1_bck.detach().cpu().numpy()

            # 更新字典
            for i in range(prediction.shape[0]):
                img = img_name[i]
                pseudo_label_dic[img] = pseudo_label_np[i]
                uncertain_dic[img] = std_map_np[i]
                proto_pseudo_dic[img] = proto_pseudo_np[i]
                distance_0_obj_dic[img] = distance_0_obj_np[i]
                distance_0_bck_dic[img] = distance_0_bck_np[i]
                distance_1_obj_dic[img] = distance_1_obj_np[i]
                distance_1_bck_dic[img] = distance_1_bck_np[i]
                centroid_0_obj_dic[img] = centroid_0_obj_np
                centroid_0_bck_dic[img] = centroid_0_bck_np
                centroid_1_obj_dic[img] = centroid_1_obj_np
                centroid_1_bck_dic[img] = centroid_1_bck_np

                # 如果启用了可视化，保存图像
                if args.save_visualization:
                    # 获取原始图像
                    # 假设原始图像在 sample['original_image'] 中，并且已经被标准化
                    # 如果不是，请根据实际情况调整
                    original_image = sample.get("original_image")
                    if original_image is not None:
                        original_image = (
                            original_image[i].detach().cpu().numpy().transpose(1, 2, 0)
                        )
                        # 反标准化（根据您的数据预处理步骤调整）
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        original_image = std * original_image + mean
                        original_image = np.clip(original_image * 255, 0, 255).astype(
                            np.uint8
                        )

                        # 保存原始图像
                        original_save_path = osp.join(original_save_dir, f"{img}.png")
                        cv2.imwrite(
                            original_save_path,
                            cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
                        )

                    # 获取伪标签
                    print(np.unique(pseudo_label_np[i]))
                    pseudo_label_single = (
                        pseudo_label_np[i].argmax(axis=0).astype(np.uint8)
                    )  # 假设是多类别
                    print(np.unique(pseudo_label_single))
                    # 保存伪标签
                    pseudo_label_save_path = osp.join(
                        pseudo_label_save_dir, f"{img}_pseudo.png"
                    )
                    cv2.imwrite(
                        pseudo_label_save_path, pseudo_label_single * 255
                    )  # 将标签转为二值图像

                    # 叠加伪标签在原始图像上
                    if original_image is not None:
                        overlaid_image = overlay_mask_on_image(
                            original_image, pseudo_label_single, color_map
                        )
                        overlay_save_path = osp.join(
                            overlay_save_dir, f"{img}_overlay.png"
                        )
                        cv2.imwrite(
                            overlay_save_path,
                            cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR),
                        )

    # 保存结果
    if args.dataset == "Domain1":
        np.savez(
            osp.join("./results/prototype/", "pseudolabel_D1.npz"),
            pseudo_label_dic=pseudo_label_dic,
            uncertain_dic=uncertain_dic,
            proto_pseudo_dic=proto_pseudo_dic,
            distance_0_obj_dic=distance_0_obj_dic,
            distance_0_bck_dic=distance_0_bck_dic,
            distance_1_obj_dic=distance_1_obj_dic,
            distance_1_bck_dic=distance_1_bck_dic,
            centroid_0_obj_dic=centroid_0_obj_dic,
            centroid_0_bck_dic=centroid_0_bck_dic,
            centroid_1_obj_dic=centroid_1_obj_dic,
            centroid_1_bck_dic=centroid_1_bck_dic,
        )

    elif args.dataset == "Domain2":
        np.savez(
            osp.join("./results/prototype/", "pseudolabel_D2.npz"),
            pseudo_label_dic=pseudo_label_dic,
            uncertain_dic=uncertain_dic,
            proto_pseudo_dic=proto_pseudo_dic,
            distance_0_obj_dic=distance_0_obj_dic,
            distance_0_bck_dic=distance_0_bck_dic,
            distance_1_obj_dic=distance_1_obj_dic,
            distance_1_bck_dic=distance_1_bck_dic,
            centroid_0_obj_dic=centroid_0_obj_dic,
            centroid_0_bck_dic=centroid_0_bck_dic,
            centroid_1_obj_dic=centroid_1_obj_dic,
            centroid_1_bck_dic=centroid_1_bck_dic,
        )

    logger.info("伪标签和相关指标已保存。")


if __name__ == "__main__":
    main()
