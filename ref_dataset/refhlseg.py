import json
import os
import re

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class RefHLSegDataset(data.Dataset):
    """Medical referring segmentation dataset used for Attr-Mamba.

    This public loader preserves every metadata entry and uses the mask file
    exactly as provided. It does not apply hidden size filtering, connected-
    component selection, or random data augmentation.
    """

    def __init__(
        self,
        data_root,
        split="train",
        image_size=512,
        json_prefix="refhlseg",
    ):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size

        json_path = self._resolve_json_path(json_prefix, split)
        print(f"[{split.upper()}] Loading dataset: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"[{split.upper()}] Samples: {len(self.data)}")

        try:
            resize = A.Resize(
                image_size,
                image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            )
        except TypeError:
            resize = A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR)

        self.transforms = A.Compose(
            [
                resize,
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _resolve_json_path(self, json_prefix, split):
        candidates = [
            f"{json_prefix}_{split}.json",
            f"refhlseg_{split}.json",
            f"ref_lits_{split}.json",
            f"ref_lidc_{split}.json",
            f"{split}.json",
        ]
        for filename in candidates:
            path = os.path.join(self.data_root, filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Could not find split metadata for split='{split}' in {self.data_root}. "
            f"Tried: {', '.join(candidates)}"
        )

    def __len__(self):
        return len(self.data)

    def _resolve_path(self, path):
        path = path.replace("\\", "/")
        if os.path.exists(path):
            return path
        candidate = os.path.join(self.data_root, path)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(path)

    def __getitem__(self, index):
        item = self.data[index]
        image_key = "image_path" if "image_path" in item else "img_path"
        image_path = self._resolve_path(item[image_key])
        mask_path = self._resolve_path(item["mask_path"])

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        mask = (mask > 0).astype(np.uint8)

        try:
            transformed = self.transforms(image=image, mask=mask)
        except Exception:
            try:
                resize = A.Resize(
                    self.image_size,
                    self.image_size,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                )
            except TypeError:
                resize = A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR)
            fallback = A.Compose(
                [
                    resize,
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            transformed = fallback(image=image, mask=mask)

        img_tensor = transformed["image"]
        mask_tensor = transformed["mask"].float().unsqueeze(0)

        if torch.sum(mask_tensor) > 0:
            ys, xs = torch.where(mask_tensor[0] > 0.5)
            bbox_xyxy = torch.tensor(
                [torch.min(xs), torch.min(ys), torch.max(xs) + 1, torch.max(ys) + 1],
                dtype=torch.float,
            )
        else:
            bbox_xyxy = torch.tensor([0, 0, 1, 1], dtype=torch.float)

        raw_sentence = item.get("sentence", item.get("text", ""))
        raw_sentence = re.sub(r"\s+", " ", raw_sentence).strip()

        return {
            "query_img": img_tensor,
            "query_mask": mask_tensor,
            "query_idx": index,
            "sentence": raw_sentence,
            "bbox": bbox_xyxy,
            "org_gt": mask_tensor,
            "file_id": item.get("file_id", str(index)),
        }


def collate_fn_medical(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
    images = torch.stack([item["query_img"] for item in batch], dim=0)
    texts = [item["sentence"] for item in batch]
    targets = [
        {
            "masks": item["query_mask"],
            "boxes": item["bbox"],
            "file_id": item["file_id"],
        }
        for item in batch
    ]
    return images, targets, texts
