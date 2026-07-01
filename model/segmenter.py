import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def dice_loss(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def boundary_loss(pred, target, radius=1, eta=5.0):
    """Boundary-aware BCE: W = 1 + eta * (Dilate(G, r) - Erode(G, r))."""
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=kernel_size, stride=1, padding=radius)
    boundary_mask = (dilated - eroded).clamp_(0.0, 1.0)
    weights = 1.0 + eta * boundary_mask
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (bce_loss * weights).mean()


class BaseSegmenter(nn.Module):
    def __init__(self, backbone, decoder, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

        model_name = kwargs.get("bert_path", "./checkpoint/RadBERT")
        print(f"Loading text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        print("Freezing RadBERT text encoder.")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, text, mask=None, **kwargs):
        encode_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        text_ids = encode_text["input_ids"].to(x.device, non_blocking=True)
        l_mask = encode_text["attention_mask"].to(x.device, non_blocking=True)

        input_shape = x.shape[-2:]
        ret = self.text_encoder(text_ids, attention_mask=l_mask)
        l_feats = ret["last_hidden_state"].permute(0, 2, 1)
        l_mask = l_mask.unsqueeze(dim=-1)
        pooler_out = ret["pooler_output"] if "pooler_output" in ret else ret["last_hidden_state"][:, 0, :]

        img_feat = self.backbone(x, l_feats, l_mask)[-1]
        pred = self.decoder(img_feat, l_feats, l_mask, pooler_out)
        pred = F.interpolate(pred, input_shape, mode="bilinear", align_corners=True)

        if self.training:
            if mask is None:
                raise ValueError("Training requires ground-truth masks.")
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.float()

            l_dice = dice_loss(pred, mask)
            l_focal = sigmoid_focal_loss(pred, mask, alpha=0.25, gamma=2.0)
            l_boundary = boundary_loss(pred, mask, radius=1, eta=5.0)
            loss = l_dice + l_focal + 0.1 * l_boundary

            return {
                "pred_masks": pred,
                "loss": loss,
                "loss_dice": l_dice,
                "loss_focal": l_focal,
                "loss_boundary": l_boundary,
            }

        return pred
