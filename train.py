import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')


import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
)

from model_structure import get_custom_segformer


@dataclass
class ExperimentConfig:
    exp_name: str = "model"
    data_root: str = "data"
    split_dir: str = "splits"
    model_checkpoint: str = "nvidia/mit-b0"
    output_root: str = "outputs"

    seed: Optional[int] = 42
    use_random_seed: bool = False
    batch_size: int = 4
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    image_size: int = 512
    early_stopping_patience: int = 20
    use_augmentation: bool = True

    id2label: dict = None
    label2id: dict = None
    num_classes: int = 3
    ignore_index: int = 255
    background_class_id: int = 0
    solar_class_id: int = 1
    bird_class_id: int = 2

    # Model architecture options
    use_custom_head: bool = False
    use_dfesa: bool = False
    use_scpp: bool = False
    custom_decoder_dim: int = 128
    dfesa_base_ratio: float = 0.1
    dfesa_fusion_lambda: float = 0.6

    # Loss function options
    use_cwc: bool = False
    contour_width: int = 3
    cwce_lambda: float = 2.0
    compound_alpha: float = 0.5
    separable_beta: float = 0.6

    # Class weights
    use_class_weights: bool = False
    class_weights: Tuple[float, float, float] = (1.0, 1.0, 1.5)

    def __post_init__(self):
        if self.id2label is None:
            self.id2label = {0: "background", 1: "solar_panel", 2: "bird_drop"}
        if self.label2id is None:
            self.label2id = {v: k for k, v in self.id2label.items()}


CFG = ExperimentConfig(
    exp_name="model",
    use_custom_head=True,
    use_dfesa=True,
    use_scpp=True,
    use_cwc=True,
    use_class_weights=True,
    class_weights=(1.0, 1.0, 1.5),
    learning_rate=1e-4,
    batch_size=4,
    epochs=200,
    early_stopping_patience=20,
    custom_decoder_dim=128,
    dfesa_base_ratio=0.1,
    dfesa_fusion_lambda=0.6,
    contour_width=3,
    cwce_lambda=2,
    compound_alpha=0.5,
    separable_beta=0.6,
)


OUTPUT_DIR = os.path.join(CFG.output_root, CFG.exp_name)


def set_seed(seed: Optional[int] = 42, use_random_seed: bool = False) -> int:
    if use_random_seed or seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
    print(f"Random seed for this training run: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def save_json(data: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_experiment_config(cfg: ExperimentConfig, output_dir: str):
    save_json(asdict(cfg), os.path.join(output_dir, "experiment_config.json"))


class SolarDataset(Dataset):
    def __init__(self, root_dir: str, processor, split_file: Optional[str] = None, transform=None):
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations")

        if split_file is not None:
            with open(split_file, "r", encoding="utf-8") as f:
                image_list = [line.strip() for line in f if line.strip()]
        else:
            image_list = sorted([
                name for name in os.listdir(self.img_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
            ])

        self.images = []
        for img_name in image_list:
            mask_name = os.path.splitext(img_name)[0] + ".png"
            if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                self.images.append(img_name)
            else:
                raise FileNotFoundError(f"Missing annotation: {img_name}")
        if not self.images:
            raise ValueError("The dataset has no valid image-mask pairs.")
        if len(self.images) != len(set(self.images)):
            raise ValueError("Duplicate images in split list.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        image_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError(f"Invalid mask shape: {mask_path}")
        if not np.isin(mask, [0, 1, 2, 255]).all():
            raise ValueError(f"Invalid mask labels: {mask_path}")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        encoded = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoded.items()}


class CrossEntropySegLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, class_weights: Optional[Tuple[float, ...]] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        weight = None
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32, device=logits.device)
        ce = F.cross_entropy(logits, labels, weight=weight, ignore_index=self.ignore_index)
        return ce, {"ce": ce.detach()}


class ContourWeightedCompoundLoss(nn.Module):
    def __init__(
        self,
        bird_class_id: int = 2,
        ignore_index: int = 255,
        contour_width: int = 3,
        cwce_lambda: float = 2.0,
        compound_alpha: float = 0.5,
        separable_beta: float = 0.6,
        class_weights: Optional[Tuple[float, ...]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.bird_class_id = bird_class_id
        self.ignore_index = ignore_index
        if not isinstance(contour_width, int) or contour_width < 1:
            raise ValueError("contour_width must be a positive integer in pixels.")
        self.contour_width = contour_width
        self.cwce_lambda = cwce_lambda
        self.compound_alpha = compound_alpha
        self.separable_beta = separable_beta
        self.class_weights = class_weights
        self.eps = eps

    def _binary_erode(self, mask: torch.Tensor) -> torch.Tensor:
        # Each 3x3 erosion removes one pixel inward, including at image borders.
        # Repeating it contour_width times gives the requested inner contour band.
        eroded = mask
        for _ in range(self.contour_width):
            inverse = F.pad(1.0 - eroded, (1, 1, 1, 1), value=1.0)
            eroded = 1.0 - F.max_pool2d(inverse, kernel_size=3, stride=1)
        return eroded.clamp(0.0, 1.0)

    def _extract_contour(self, mask: torch.Tensor) -> torch.Tensor:
        eroded = self._binary_erode(mask)
        return (mask - eroded).clamp(0.0, 1.0)

    def _region_dice(self, prob: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        prob_region = prob * region_mask
        target_region = target * region_mask
        numerator = 2.0 * (prob_region * target_region).sum()
        denominator = prob_region.pow(2).sum() + target_region.pow(2).sum() + self.eps
        return 1.0 - numerator / denominator

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        device = logits.device
        valid_mask = (labels != self.ignore_index).float()
        bird_target = ((labels == self.bird_class_id).float() * valid_mask).unsqueeze(1)
        contour_mask = self._extract_contour(bird_target)

        weight = None
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32, device=device)

        ce_map = F.cross_entropy(logits, labels, weight=weight, ignore_index=self.ignore_index, reduction="none")
        cw_map = 1.0 + self.cwce_lambda * contour_mask.squeeze(1)

        per_image_cwce = (ce_map * cw_map * valid_mask).flatten(1).sum(dim=1) / logits.shape[1]
        cwce = per_image_cwce.mean()

        prob = F.softmax(logits, dim=1)[:, self.bird_class_id:self.bird_class_id + 1]
        contour_region = contour_mask * bird_target
        non_contour_region = (bird_target - contour_region).clamp(0.0, 1.0)

        if contour_region.sum() < 1:
            contour_region = bird_target
        if non_contour_region.sum() < 1:
            non_contour_region = bird_target

        loss_contour = self._region_dice(prob, bird_target, contour_region)
        loss_non_contour = self._region_dice(prob, bird_target, non_contour_region)
        sdl = self.separable_beta * loss_contour + (1.0 - self.separable_beta) * loss_non_contour
        total = self.compound_alpha * sdl + (1.0 - self.compound_alpha) * cwce

        return total, {
            "cwce": cwce.detach(),
            "sdl": sdl.detach(),
            "contour_dice": loss_contour.detach(),
            "non_contour_dice": loss_non_contour.detach(),
        }


def build_model(cfg: ExperimentConfig):
    if cfg.use_custom_head:
        return get_custom_segformer(
            model_checkpoint=cfg.model_checkpoint,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
            custom_decoder_dim=cfg.custom_decoder_dim,
            dfesa_base_ratio=cfg.dfesa_base_ratio,
            dfesa_fusion_lambda=cfg.dfesa_fusion_lambda,
            use_dfesa=cfg.use_dfesa,
            use_scpp=cfg.use_scpp,
        )

    return SegformerForSemanticSegmentation.from_pretrained(
        cfg.model_checkpoint,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
        ignore_mismatched_sizes=True,
    )


class SegformerBirdDropTrainer(Trainer):
    def __init__(self, cfg: ExperimentConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_weights = cfg.class_weights if cfg.use_class_weights else None
        if cfg.use_cwc:
            self.loss_fn = ContourWeightedCompoundLoss(
                bird_class_id=cfg.bird_class_id,
                ignore_index=cfg.ignore_index,
                contour_width=cfg.contour_width,
                cwce_lambda=cfg.cwce_lambda,
                compound_alpha=cfg.compound_alpha,
                separable_beta=cfg.separable_beta,
                class_weights=class_weights,
            )
        else:
            self.loss_fn = CrossEntropySegLoss(ignore_index=cfg.ignore_index, class_weights=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        pixel_values = inputs["pixel_values"]
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss, loss_items = self.loss_fn(logits, labels)
        if hasattr(outputs, "loss"):
            outputs.loss = loss
        if self.state.global_step % max(self.args.logging_steps, 1) == 0 and loss_items:
            self.log({k: float(v.item()) for k, v in loss_items.items()})
        return (loss, outputs) if return_outputs else loss


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _build_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, label in zip(predictions, labels):
        valid_mask = label != ignore_index
        gt = label[valid_mask].reshape(-1)
        pd = pred[valid_mask].reshape(-1)
        if gt.size == 0:
            continue
        indices = gt * num_classes + pd
        bincount = np.bincount(indices, minlength=num_classes * num_classes)
        conf_mat += bincount.reshape(num_classes, num_classes)

    return conf_mat


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    predictions = F.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1).cpu().numpy()

    conf_mat = _build_confusion_matrix(
        predictions=predictions,
        labels=labels,
        num_classes=CFG.num_classes,
        ignore_index=CFG.ignore_index,
    )

    tp = np.diag(conf_mat).astype(np.float64)
    fp = conf_mat.sum(axis=0).astype(np.float64) - tp
    fn = conf_mat.sum(axis=1).astype(np.float64) - tp
    total_valid = conf_mat.sum().astype(np.float64)

    iou = _safe_divide(tp, tp + fp + fn)
    recall = _safe_divide(tp, tp + fn)
    precision = _safe_divide(tp, tp + fp)
    fscore = _safe_divide(2.0 * precision * recall, precision + recall)
    overall_acc = float(tp.sum() / total_valid) if total_valid > 0 else 0.0

    bird_id = CFG.bird_class_id

    results = {
        "mIoU": float(np.nanmean(iou)),
        "mRecall": float(np.nanmean(recall)),
        "mFscore": float(np.nanmean(fscore)),
        "Acc": overall_acc,
        "bird_iou": float(iou[bird_id]) if not np.isnan(iou[bird_id]) else 0.0,
        "bird_acc": float(recall[bird_id]) if not np.isnan(recall[bird_id]) else 0.0,
    }

    return results


def build_train_transform(cfg: ExperimentConfig):
    if not cfg.use_augmentation:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.05, p=0.5, border_mode=0),
    ])


def prepare_datasets(cfg, processor):
    """Use user-preassigned subsets; never silently random-split related images."""
    from pathlib import Path
    root = Path(cfg.data_root)
    train_list = root/cfg.split_dir/'train.txt'
    val_list = root/cfg.split_dir/'val.txt'
    if train_list.is_file() and val_list.is_file():
        train_data = SolarDataset(str(root), processor, str(train_list), build_train_transform(cfg))
        val_data = SolarDataset(str(root), processor, str(val_list), None)
        if set(train_data.images) & set(val_data.images):
            raise ValueError('Training and validation lists contain overlapping image names.')
        test_list = root/cfg.split_dir/'test.txt'
        if test_list.is_file():
            test_names = {line.strip() for line in test_list.read_text(encoding='utf-8').splitlines() if line.strip()}
            if (set(train_data.images) | set(val_data.images)) & test_names:
                raise ValueError('The test list overlaps training/validation images.')
    elif (root/'train'/'images').is_dir() and (root/'val'/'images').is_dir():
        train_data = SolarDataset(str(root/'train'), processor, transform=build_train_transform(cfg))
        val_data = SolarDataset(str(root/'val'), processor, transform=None)
    else:
        raise FileNotFoundError('Prepare panel/session-grouped train/val subsets: '
                                'splits/train.txt and val.txt, or train/images, train/annotations, '
                                'val/images, val/annotations. No random image split is performed.')
    return train_data, val_data


def run_experiment(cfg):
    global CFG
    import math
    CFG = cfg
    cfg.seed = set_seed(cfg.seed, False)
    output_dir = os.path.join(cfg.output_root, cfg.exp_name)
    if os.path.exists(os.path.join(output_dir, 'final_model')):
        raise FileExistsError(f'Refusing to overwrite a completed run: {output_dir}')
    processor = SegformerImageProcessor.from_pretrained(
        cfg.model_checkpoint, do_reduce_labels=False,
        size={'height': cfg.image_size, 'width': cfg.image_size})
    train_data, val_data = prepare_datasets(cfg, processor)
    os.makedirs(output_dir, exist_ok=True)
    save_experiment_config(cfg, output_dir)
    steps = max(math.ceil(len(train_data)/cfg.batch_size)*5, 10)
    model = build_model(cfg)
    args = TrainingArguments(
        output_dir=output_dir, learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay, num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size,
        save_total_limit=2, eval_strategy='steps', save_strategy='steps',
        eval_steps=steps, save_steps=steps, logging_steps=10,
        remove_unused_columns=False, push_to_hub=False, load_best_model_at_end=True,
        metric_for_best_model='bird_iou', greater_is_better=True,
        seed=cfg.seed, data_seed=cfg.seed, report_to='none',
        max_grad_norm=1.0, warmup_ratio=0.05, optim='adamw_torch')
    trainer = SegformerBirdDropTrainer(
        cfg=cfg, model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)])
    trainer.train()
    results = trainer.evaluate(eval_dataset=val_data)
    save_json(results, os.path.join(output_dir, 'final_eval_metrics.json'))
    final_dir = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    import argparse
    from dataclasses import replace
    parser = argparse.ArgumentParser(description='Train the full model with fixed user-preassigned subsets.')
    parser.add_argument('--data-root', default=CFG.data_root)
    parser.add_argument('--output-root', default=CFG.output_root)
    parser.add_argument('--model-checkpoint', default=CFG.model_checkpoint)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024],
                        help='Three configurable run seeds; these are defaults, not recorded manuscript seed IDs.')
    parser.add_argument('--exp-name', default='full_model')
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Seeds must be different.')
    runs = []
    for seed in args.seeds:
        cfg = replace(CFG, seed=seed, data_root=args.data_root, output_root=args.output_root,
                      model_checkpoint=args.model_checkpoint, exp_name=f'{args.exp_name}_seed{seed}')
        metrics = run_experiment(cfg)
        runs.append({'seed': seed, 'validation_metrics': metrics})
    summary = {'runs': runs, 'evaluation_subset': 'validation', 'training_metric_units': 'fractions',
               'mean': {}, 'sample_standard_deviation': {}}
    for key in ['mIoU', 'mRecall', 'mFscore', 'Acc', 'bird_iou', 'bird_acc']:
        values = [r['validation_metrics']['eval_'+key] for r in runs]
        summary['mean'][key] = float(np.mean(values))
        summary['sample_standard_deviation'][key] = float(np.std(values, ddof=1)) if len(values)>1 else None
    save_json(summary, os.path.join(args.output_root, args.exp_name+'_summary.json'))


if __name__ == '__main__':
    main()


