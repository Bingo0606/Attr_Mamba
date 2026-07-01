"""Training and evaluation utilities for Attr-Mamba."""

from typing import Iterable, Optional

import datetime
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEma

import utils


class FocalLoss(nn.Module):
    """Binary focal loss used with Dice and boundary-aware BCE."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


def dice_loss(inputs, targets, smooth=1):
    """Dice loss for binary segmentation."""
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    intersection = (inputs * targets).sum(1)
    dice = (2.0 * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)
    return 1 - dice.mean()


def boundary_loss(inputs, targets, radius=1, eta=5.0):
    """Boundary-aware BCE: W = 1 + eta * (Dilate(G, r) - Erode(G, r))."""
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - targets, kernel_size=kernel_size, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp_(0.0, 1.0)
    weights = 1.0 + eta * boundary
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return (weights * bce).mean()


def compute_loss(pred, target, boundary_radius=1, boundary_eta=5.0, boundary_weight=0.1):
    """Fallback loss for models that return logits without an internal loss."""
    if isinstance(pred, tuple):
        pred = pred[0]
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"compute_loss expects a Tensor, but got {type(pred)}")
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

    target = target.float()
    loss_focal = FocalLoss(alpha=0.25, gamma=2.0)(pred, target)
    loss_dice = dice_loss(pred, target)
    loss_boundary = boundary_loss(pred, target, radius=boundary_radius, eta=boundary_eta)
    total_loss = loss_dice + loss_focal + boundary_weight * loss_boundary
    return total_loss, {
        "loss_focal": loss_focal,
        "loss_dice": loss_dice,
        "loss_boundary": loss_boundary,
    }


def denormalize_image(img_tensor):
    img = img_tensor.cpu().detach().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize_prediction(img_tensor, pred_mask, gt_mask, save_path, iou_score, dice_score):
    try:
        img = denormalize_image(img_tensor)
        img_orig = img.copy()
        img_gt = img.copy()
        img_pred = img.copy()

        green_layer = np.zeros_like(img)
        green_layer[:, :, 1] = 255
        mask_indices = gt_mask > 0
        if np.any(mask_indices):
            img_gt[mask_indices] = cv2.addWeighted(
                img_gt[mask_indices], 0.6, green_layer[mask_indices], 0.4, 0
            )

        red_layer = np.zeros_like(img)
        red_layer[:, :, 2] = 255
        pred_indices = pred_mask > 0
        if np.any(pred_indices):
            img_pred[pred_indices] = cv2.addWeighted(
                img_pred[pred_indices], 0.6, red_layer[pred_indices], 0.4, 0
            )

        img_binary = cv2.cvtColor((pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            img_pred,
            f"Pred | IoU:{iou_score:.2f} Dice:{dice_score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(img_binary, "Binary Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(save_path, np.hstack((img_orig, img_gt, img_pred, img_binary)))
    except Exception as exc:
        print(f"Warning: failed to save visualization: {exc}")


def trainMetricGPU(output, target, threshold=0.5):
    if isinstance(output, tuple):
        output = output[0]
    output = F.interpolate(output, size=target.shape[-2:], mode="bilinear", align_corners=False)
    output = (torch.sigmoid(output) > threshold).float()
    inter = (output * target).sum(dim=(1, 2, 3))
    union = (output + target).sum(dim=(1, 2, 3)) - inter
    return (inter / (union + 1e-6)).mean()


def _surface(mask):
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (mask ^ eroded).astype(bool)


def compute_hd95(pred_mask, gt_mask):
    """Compute the symmetric 95th-percentile Hausdorff distance in pixels."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        h, w = gt_mask.shape[-2:]
        return float(np.sqrt(h * h + w * w))

    pred_surface = _surface(pred_mask)
    gt_surface = _surface(gt_mask)
    if not pred_surface.any() or not gt_surface.any():
        return 0.0

    dist_to_gt = cv2.distanceTransform((~gt_surface).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform((~pred_surface).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([dist_to_gt[pred_surface], dist_to_pred[gt_surface]])
    return float(np.percentile(distances, 95))


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    amp_autocast,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    set_training_mode=True,
    args=None,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    for data_iter_step, (samples, targets, texts) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        gt_masks = torch.stack([t["masks"].to(device) for t in targets]).float()
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        with amp_autocast():
            outputs = model(samples, texts, gt_masks)
            loss_value = None
            loss_dict = {}
            pred = None

            if isinstance(outputs, dict):
                if "loss" in outputs:
                    loss_value = outputs["loss"]
                    loss_dict = {k: v.item() for k, v in outputs.items() if "loss" in k and k != "loss"}
                if "pred_masks" in outputs:
                    pred = outputs["pred_masks"]
            elif isinstance(outputs, tuple):
                for item in outputs:
                    if isinstance(item, torch.Tensor):
                        if item.ndim == 0 and loss_value is None:
                            loss_value = item
                        elif item.ndim >= 3 and pred is None:
                            pred = item
                    elif isinstance(item, dict) and "pred_masks" in item:
                        pred = item["pred_masks"]
                if pred is None and len(outputs) > 0:
                    pred = outputs[0]
            elif isinstance(outputs, torch.Tensor):
                pred = outputs

            if loss_value is None:
                if pred is None:
                    continue
                loss_value, loss_dict_computed = compute_loss(pred, gt_masks)
                loss_dict = {k: v.item() for k, v in loss_dict_computed.items()}

            if pred is not None:
                train_iou = trainMetricGPU(pred.detach(), gt_masks)
                metric_logger.update(train_iou=train_iou.item())

        if not torch.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        accum_steps = getattr(args, "accumulation_steps", 1)
        loss_value = loss_value / accum_steps

        if loss_scaler != "none":
            is_update_step = (data_iter_step + 1) % accum_steps == 0
            loss_scaler(
                loss_value,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                update_grad=is_update_step,
            )
        else:
            loss_value.backward()

        if (data_iter_step + 1) % accum_steps == 0:
            if loss_scaler == "none":
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        metric_logger.update(loss=loss_value.item() * accum_steps, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, log_every=50, threshold=0.5, vis_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    save_vis = bool(vis_dir)
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
        log_file_path = os.path.abspath(os.path.join(vis_dir, "evaluation_logs.txt"))
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 20} Evaluation Start: {datetime.datetime.now()} {'=' * 20}\n")
        print(f"Evaluation visualizations will be saved to: {vis_dir}")
    else:
        log_file_path = None

    for i, (samples, targets, texts) in enumerate(metric_logger.log_every(data_loader, log_every, header)):
        samples = samples.to(device)
        with amp_autocast():
            outputs = model(samples, texts)
            if isinstance(outputs, dict):
                pred = outputs.get("pred_masks", None)
            elif isinstance(outputs, tuple):
                pred = next((x for x in outputs if isinstance(x, torch.Tensor) and x.ndim >= 3), outputs[0])
            else:
                pred = outputs

        gt_mask_list = [t["masks"].to(device) for t in targets]
        for batch_idx, (single_pred, gt_mask) in enumerate(zip(pred, gt_mask_list)):
            if single_pred.dim() == 2:
                single_pred = single_pred.unsqueeze(0)
            single_pred = F.interpolate(
                single_pred[None], size=gt_mask.shape[-2:], mode="bilinear", align_corners=False
            )[0]
            single_pred_np = (single_pred.sigmoid() > threshold).cpu().numpy().astype(np.uint8)
            if single_pred_np.ndim == 3:
                single_pred_np = single_pred_np[0]

            gt_mask_np = (gt_mask > 0).cpu().numpy().astype(np.uint8)
            if gt_mask_np.ndim == 3:
                gt_mask_np = gt_mask_np[0]

            inter = np.logical_and(single_pred_np, gt_mask_np).sum()
            union = np.logical_or(single_pred_np, gt_mask_np).sum()
            iou = 1.0 if union == 0 and inter == 0 else inter / (union + 1e-6)
            dice = 2 * inter / (union + inter + 1e-6) if (union + inter) > 0 else 1.0
            hd95 = compute_hd95(single_pred_np, gt_mask_np)

            if save_vis:
                try:
                    current_file_id = targets[batch_idx].get("file_id", f"batch_{i}_{batch_idx}")
                    current_sentence = texts[batch_idx]
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(f"[File]: {current_file_id}\n")
                        f.write(f"[Score]: IoU={iou:.4f}, Dice={dice:.4f}, HD95={hd95:.4f}\n")
                        f.write(f"[Prompt]: {current_sentence}\n")
                        f.write("-" * 50 + "\n")

                    current_img = samples[batch_idx] if isinstance(samples, torch.Tensor) else samples.tensors[batch_idx]
                    safe_id = str(current_file_id).replace("/", "_").replace("\\", "_")
                    save_path = os.path.join(vis_dir, f"res_iou{iou:.2f}_{safe_id}.jpg")
                    visualize_prediction(current_img, single_pred_np, gt_mask_np, save_path, iou, dice)
                except Exception as exc:
                    print(f"Warning: failed to log or visualize a sample: {exc}")

            metric_logger.meters["iou"].update(iou)
            metric_logger.meters["dice"].update(dice)
            metric_logger.meters["hd95"].update(hd95)
            metric_logger.meters["inter"].update(inter)
            metric_logger.meters["union"].update(union)

    m_iou = metric_logger.iou.global_avg
    m_dice = metric_logger.dice.global_avg
    m_hd95 = metric_logger.hd95.global_avg
    o_iou = metric_logger.inter.global_avg / (metric_logger.union.global_avg + 1e-6)

    print(f"* mIoU {m_iou:.5f}  mDice {m_dice:.5f}  HD95 {m_hd95:.5f}  oIoU {o_iou:.5f}")
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["oiou"] = o_iou
    return stats
