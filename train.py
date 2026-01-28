from __future__ import annotations

import os
# 尽量减少Windows下torch首次加载CUDA DLL时的内存压力
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import argparse
import csv
import json
import math
import random
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, resnet50

try:
    # torchvision>=0.13
    from torchvision.models import vit_b_16  # type: ignore
except Exception:
    vit_b_16 = None  # type: ignore
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


LABEL_MAP: Dict[str, int] = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(text: str, mode: str = "basic") -> str:
    """文本清洗。

    mode:
    - basic: 移除URL，并把非常规字符替换为空格（当前默认行为）
    - none: 仅移除URL与多余空白，尽量保留emoji/特殊符号（情感任务可能更有效）
    """

    text = re.sub(r"http\S+|www\S+", "", text)
    if mode == "none":
        return re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：,.!?;:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_label_file(path: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            guid = row[0].strip()
            tag = row[1].strip()
            items.append((guid, tag))
    return items


def split_train_val_stratified(
    items: Sequence[Tuple[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    items = list(items)
    y = [LABEL_MAP[tag] for _, tag in items]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(items)), y))
    train_list = [items[i] for i in train_idx.tolist()]
    val_list = [items[i] for i in val_idx.tolist()]
    return train_list, val_list


def split_train_val(
    items: Sequence[Tuple[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_size = max(1, int(len(items) * val_ratio))
    return items[:-val_size], items[-val_size:]


def get_transforms(train: bool, augment_mode: str) -> transforms.Compose:
    if train and augment_mode != "off":
        if augment_mode == "mild":
            # 对含文字/表情包类图片更友好：不旋转、不翻转
            return transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        # strong：原来的增强
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class MMDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        items: Sequence[Tuple[str, str]],
        tokenizer,
        transform,
        mode: str,
        text_clean: str = "basic",
        max_length: int = 128,
    ):
        self.data_dir = Path(data_dir)
        self.items = list(items)
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode
        self.text_clean = text_clean
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        guid, tag = self.items[idx]
        txt_path = self.data_dir / f"{guid}.txt"
        img_path = self.data_dir / f"{guid}.jpg"

        try:
            text = txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            text = ""
        text = clean_text(text, mode=self.text_clean)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        if self.mode in {"train", "val"}:
            label_id = LABEL_MAP[tag]
            return input_ids, attention_mask, image, torch.tensor(label_id, dtype=torch.long), guid
        return input_ids, attention_mask, image, guid


def collate_train(batch):
    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    images = torch.stack([b[2] for b in batch])
    labels = torch.stack([b[3] for b in batch])
    guids = [b[4] for b in batch]
    return input_ids, attention_mask, images, labels, guids


def collate_test(batch):
    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    images = torch.stack([b[2] for b in batch])
    guids = [b[3] for b in batch]
    return input_ids, attention_mask, images, guids


class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()

        def _load_with_fallback(builder, weights_enum_name: str):
            """兼容 torchvision 不同版本的 weights/pretrained API。"""

            try:
                # 新版：weights=XXX.DEFAULT
                mod = __import__("torchvision.models", fromlist=[weights_enum_name])
                weights_enum = getattr(mod, weights_enum_name)
                return builder(weights=weights_enum.DEFAULT)
            except Exception:
                # 旧版：pretrained=True
                return builder(pretrained=True)

        if backbone == "resnet50":
            net = _load_with_fallback(resnet50, "ResNet50_Weights")
            feat_dim = 2048
            net.fc = nn.Identity()
        elif backbone == "resnet18":
            net = _load_with_fallback(resnet18, "ResNet18_Weights")
            feat_dim = 512
            net.fc = nn.Identity()
        elif backbone == "vit_b_16":
            if vit_b_16 is None:
                raise ValueError("当前 torchvision 版本不支持 vit_b_16，请升级 torchvision 或改用 resnet50")
            net = _load_with_fallback(vit_b_16, "ViT_B_16_Weights")
            # ViT 的分类头叫 heads
            if hasattr(net, "heads"):
                net.heads = nn.Identity()
            feat_dim = 768
        else:
            raise ValueError(f"不支持的 image_backbone: {backbone}. 可选：resnet18/resnet50/vit_b_16")

        self.net = net
        self.feat_dim = feat_dim

    def forward(self, images):
        return self.net(images)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        text_model: str,
        image_backbone: str,
        fusion: str,
        modality: str,
        num_classes: int = 3,
        proj_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.modality = modality
        self.fusion = fusion

        self.text_enc = TextEncoder(text_model)
        self.img_enc = ImageEncoder(image_backbone)

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_enc.hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(self.img_enc.feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if fusion == "gated":
            # 简化的门控机制
            self.gate = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.Sigmoid(),
            )
        elif fusion == "attention":
            # 交叉注意力融合
            self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads=4, dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(proj_dim)

        head_in = proj_dim
        if modality == "multimodal":
            head_in = proj_dim * 2
        # 平衡的分类头：适度容量 + 有效正则化
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        if self.modality in {"text", "multimodal"}:
            t = self.text_enc(input_ids, attention_mask)
            t = self.text_proj(t)
        else:
            t = None

        if self.modality in {"image", "multimodal"}:
            v = self.img_enc(images)
            v = self.img_proj(v)
        else:
            v = None

        if self.modality == "text":
            feats = t
        elif self.modality == "image":
            feats = v
        else:
            if self.fusion == "gated":
                raw = torch.cat([t, v], dim=1)
                g = self.gate(raw)
                t2 = t * g
                v2 = v * (1 - g)
                feats = torch.cat([t2, v2], dim=1)
            elif self.fusion == "attention":
                # 交叉注意力：text作为query，image作为key/value
                t_seq = t.unsqueeze(1)  # (B, 1, D)
                v_seq = v.unsqueeze(1)  # (B, 1, D)
                attn_out, _ = self.cross_attn(t_seq, v_seq, v_seq)
                t_enhanced = self.attn_norm(t + attn_out.squeeze(1))
                feats = torch.cat([t_enhanced, v], dim=1)
            else:
                feats = torch.cat([t, v], dim=1)

        return self.classifier(feats)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_acc: float
    val_f1: float
    lr: float


class FocalLoss(nn.Module):
    """多分类 Focal Loss（支持alpha类别权重）。"""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(train_list: Sequence[Tuple[str, str]], device: torch.device) -> torch.Tensor:
    labels = [LABEL_MAP[tag] for _, tag in train_list]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=3).float()
    # 逆频率权重：越少的类权重越大
    weights = counts.sum() / (counts.clamp_min(1.0) * len(counts))
    return weights.to(device)


def make_weighted_sampler(train_list: Sequence[Tuple[str, str]]) -> WeightedRandomSampler:
    labels = np.array([LABEL_MAP[tag] for _, tag in train_list], dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    inv = 1.0 / np.clip(counts, 1, None)
    sample_weights = inv[labels].astype(np.float64)
    return WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)


def evaluate(model, loader, device, use_amp: bool) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    trues: List[int] = []
    with torch.no_grad():
        for input_ids, attention_mask, images, labels, _ in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(input_ids, attention_mask, images)
            else:
                logits = model(input_ids, attention_mask, images)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.detach().cpu().tolist())
            trues.extend(labels.detach().cpu().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    return acc, f1, preds, trues


def predict_test(model, loader, device, use_amp: bool) -> List[Tuple[str, str]]:
    model.eval()
    results: List[Tuple[str, str]] = []
    with torch.no_grad():
        for input_ids, attention_mask, images, guids in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(input_ids, attention_mask, images)
            else:
                logits = model(input_ids, attention_mask, images)
            pred = torch.argmax(logits, dim=1).detach().cpu().tolist()
            for g, p in zip(guids, pred):
                results.append((g, ID2LABEL[p]))
    return results


def save_plots(output_dir: Path, logs: List[EpochLog], cm: np.ndarray) -> None:
    epochs = [x.epoch for x in logs]
    train_loss = [x.train_loss for x in logs]
    val_acc = [x.val_acc for x in logs]
    val_f1 = [x.val_f1 for x in logs]
    lr = [x.lr for x in logs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("训练损失")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_acc, marker="o")
    axes[1].set_title("验证准确率")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_f1, marker="o")
    axes[2].set_title("验证Macro-F1")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, lr, marker="o")
    axes[3].set_title("学习率")
    axes[3].set_yscale("log")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # confusion matrix
    fig2, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    labels = [ID2LABEL[i] for i in range(3)]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig2.colorbar(im, ax=ax)
    plt.tight_layout()
    fig2.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig2)


def train_one_run(args, run_name: str, output_root: Path) -> Dict:
    set_seed(args.seed, deterministic=not args.nondeterministic)

    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # output
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    train_items = read_label_file(args.train_file)
    test_items = read_label_file(args.test_file)
    if args.stratified_split:
        train_list, val_list = split_train_val_stratified(train_items, args.val_ratio, args.seed)
    else:
        train_list, val_list = split_train_val(train_items, args.val_ratio, args.seed)

    train_tf = get_transforms(train=True, augment_mode=args.augment_mode)
    eval_tf = get_transforms(train=False, augment_mode="off")

    train_ds = MMDataset(
        args.data_dir,
        train_list,
        tokenizer,
        train_tf,
        mode="train",
        text_clean=args.text_clean,
        max_length=args.max_length,
    )
    val_ds = MMDataset(
        args.data_dir,
        val_list,
        tokenizer,
        eval_tf,
        mode="val",
        text_clean=args.text_clean,
        max_length=args.max_length,
    )
    test_ds = MMDataset(
        args.data_dir,
        test_items,
        tokenizer,
        eval_tf,
        mode="test",
        text_clean=args.text_clean,
        max_length=args.max_length,
    )

    sampler = make_weighted_sampler(train_list) if args.use_weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_train,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_test,
    )

    model = FusionClassifier(
        text_model=args.text_model,
        image_backbone=args.image_backbone,
        fusion=args.fusion,
        modality=args.modality,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
    ).to(device)

    def _set_encoder_trainable(text_trainable: bool, image_trainable: bool) -> None:
        for p in model.text_enc.parameters():
            p.requires_grad = bool(text_trainable)
        for p in model.img_enc.parameters():
            p.requires_grad = bool(image_trainable)

    # 永久冻结（如果指定了）
    if args.freeze_text or args.freeze_image:
        _set_encoder_trainable(text_trainable=not args.freeze_text, image_trainable=not args.freeze_image)

    # 分组学习率：encoder更小，head更大，通常更容易提准
    lr_text = args.lr * args.lr_text_mult
    lr_image = args.lr * args.lr_image_mult
    lr_head = args.lr * args.lr_head_mult

    head_modules = [model.text_proj, model.img_proj, model.classifier]
    if hasattr(model, "gate"):
        head_modules.append(model.gate)

    head_params: List[torch.nn.Parameter] = []
    for m in head_modules:
        head_params.extend([p for p in m.parameters() if p.requires_grad])

    optimizer = AdamW(
        [
            # 注意：这里不要按 requires_grad 过滤，否则“先冻结后解冻”会导致参数不在 optimizer 里
            {"params": list(model.text_enc.parameters()), "lr": lr_text},
            {"params": list(model.img_enc.parameters()), "lr": lr_image},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # loss：重点解决 neutral 类偏弱问题
    class_weights = compute_class_weights(train_list, device) if args.use_class_weights else None
    if args.loss == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    scaler = torch.amp.GradScaler(device="cuda") if (args.use_amp and device.type == "cuda") else None

    best_metric = -1.0
    best_epoch = 0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 0
    logs: List[EpochLog] = []

    # train loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # 分阶段训练：先训练 head（冻结 encoder），再解冻全量微调；对小数据集更稳、也更容易冲高点。
        if (not args.freeze_text) and (not args.freeze_image) and args.freeze_epochs > 0:
            if epoch <= args.freeze_epochs:
                _set_encoder_trainable(text_trainable=False, image_trainable=False)
            else:
                _set_encoder_trainable(text_trainable=True, image_trainable=True)

        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"{run_name} Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (input_ids, attention_mask, images, labels, _) in enumerate(pbar, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(input_ids, attention_mask, images)
                    loss = criterion(logits, labels)
                    loss_to_backprop = loss / max(1, args.grad_accum_steps)
                scaler.scale(loss_to_backprop).backward()
            else:
                logits = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels)
                loss_to_backprop = loss / max(1, args.grad_accum_steps)
                loss_to_backprop.backward()

            do_step = (step_idx % max(1, args.grad_accum_steps) == 0) or (step_idx == len(train_loader))
            if do_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                scheduler.step()

            running += float(loss.item())
            pbar.set_postfix(loss=running / max(1, pbar.n + 1), step=f"{global_step}/{total_steps}")

        train_loss = running / max(1, len(train_loader))
        val_acc, val_f1, val_preds, val_trues = evaluate(model, val_loader, device, use_amp=args.use_amp)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # 记录head学习率（最后一个param_group）
        lr = float(optimizer.param_groups[-1]["lr"])
        logs.append(EpochLog(epoch=epoch, train_loss=train_loss, val_acc=val_acc, val_f1=val_f1, lr=lr))

        # save epoch log
        with open(out_dir / "epoch_log.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_acc", "val_f1", "lr"])
            for x in logs:
                writer.writerow([x.epoch, f"{x.train_loss:.6f}", f"{x.val_acc:.6f}", f"{x.val_f1:.6f}", f"{x.lr:.8e}"])

        print(f"\n[{run_name}] Epoch {epoch}: loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} lr={lr:.2e}\n")

        metric_value = float(val_acc) if args.select_metric == "acc" else float(val_f1)
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            best_val_acc = float(val_acc)
            best_val_f1 = float(val_f1)
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best": {
                        "epoch": best_epoch,
                        "val_acc": best_val_acc,
                        "val_f1": best_val_f1,
                        "select_metric": args.select_metric,
                        "select_value": best_metric,
                    },
                },
                out_dir / "best_model.pt",
            )
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[{run_name}] 早停触发：{args.patience}轮 {args.select_metric} 无提升")
                break

    # load best for final eval + test pred
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    val_acc, val_f1, val_preds, val_trues = evaluate(model, val_loader, device, use_amp=args.use_amp)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cm = confusion_matrix(val_trues, val_preds)
    save_plots(out_dir, logs, cm)

    test_preds = predict_test(model, test_loader, device, use_amp=args.use_amp)
    with open(out_dir / "test_predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["guid", "tag"])
        for guid, tag in test_preds:
            w.writerow([guid, tag])

    summary = {
        "run_name": run_name,
        "device": str(device),
        "select_metric": args.select_metric,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "best_selected_value": float(best_metric),
        "final_val_acc": float(val_acc),
        "final_val_f1": float(val_f1),
        "epochs_ran": int(logs[-1].epoch if logs else 0),
        "fusion": args.fusion,
        "modality": args.modality,
        "text_model": args.text_model,
        "image_backbone": args.image_backbone,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./project5/data")
    p.add_argument("--train_file", type=str, default="./project5/train.txt")
    p.add_argument("--test_file", type=str, default="./project5/test_without_label.txt")

    p.add_argument("--output_dir", type=str, default="./output_ppt")

    p.add_argument("--text_model", type=str, default="bert-base-uncased")
    p.add_argument(
        "--image_backbone",
        type=str,
        default="resnet18",
        help="图像骨干：resnet18/resnet50/vit_b_16（vit需要较新torchvision）",
    )

    p.add_argument("--fusion", type=str, default="gated", choices=["late", "gated", "attention"])
    p.add_argument("--modality", type=str, default="multimodal", choices=["multimodal", "text", "image"])

    p.add_argument("--run_ablation", action="store_true", help="一次性跑：text/image/multimodal 三组（用于报告消融实验）")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--scheduler", choices=["linear", "cosine"], default="linear", help="学习率调度：linear/cosine")
    p.add_argument(
        "--select_metric",
        choices=["acc", "f1"],
        default="acc",
        help="保存最优模型/早停的指标：acc(验证准确率) 或 f1(验证macro-F1)",
    )

    # 分组学习率（常用于提升收敛与最终精度）
    p.add_argument("--lr_text_mult", type=float, default=1.0, help="文本编码器 lr 倍数")
    p.add_argument("--lr_image_mult", type=float, default=1.0, help="图像编码器 lr 倍数")
    p.add_argument("--lr_head_mult", type=float, default=5.0, help="投影层/融合层/分类头 lr 倍数")

    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数（等效增大batch，缓解显存/提升稳定性）")
    p.add_argument("--freeze_epochs", type=int, default=0, help="前N个epoch冻结text+image编码器，仅训练head；之后解冻全量微调")

    # 类别不均衡对策（重点提升 neutral/negative 的召回，从而带动 macro-F1）
    p.add_argument("--stratified_split", action="store_true", help="分层划分 train/val，稳定验证集类别分布")
    p.add_argument("--use_weighted_sampler", action="store_true", help="训练集使用 WeightedRandomSampler")
    p.add_argument("--use_class_weights", action="store_true", help="loss 使用类别权重")
    p.add_argument("--loss", choices=["ce", "focal"], default="ce", help="损失函数：ce/focal")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="FocalLoss gamma")

    p.add_argument("--data_augment", action="store_true", help="兼容旧参数：等价于 --augment_mode strong")
    p.add_argument("--augment_mode", choices=["off", "mild", "strong"], default="off")
    p.add_argument("--text_clean", choices=["basic", "none"], default="basic")
    p.add_argument("--max_length", type=int, default=128)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--nondeterministic", action="store_true", help="关闭尽量deterministic设置（可能更快）")

    p.add_argument("--freeze_text", action="store_true")
    p.add_argument("--freeze_image", action="store_true")

    p.add_argument("--num_workers", type=int, default=2)

    args = p.parse_args()

    # 兼容旧参数：如果设置了 --data_augment 且未显式指定 augment_mode，则使用 strong
    if args.data_augment and args.augment_mode == "off":
        args.augment_mode = "strong"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / ts
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_ablation:
        summaries = []
        # 保持同一组超参，仅切换modality，满足“消融实验”公平性
        for m in ["text", "image", "multimodal"]:
            args2 = argparse.Namespace(**vars(args))
            args2.modality = m
            run_name = f"ablation_{m}"
            summaries.append(train_one_run(args2, run_name, output_root))

        with open(output_root / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

        # 画个对比柱状图（acc/f1）
        labels = [s["modality"] for s in summaries]
        accs = [s["final_val_acc"] for s in summaries]
        f1s = [s["final_val_f1"] for s in summaries]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, accs, width, label="Acc")
        ax.bar(x + width / 2, f1s, width, label="Macro-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title("消融实验对比（验证集）")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_root / "ablation_bar.png", dpi=150)
        plt.close(fig)

        print(f"消融实验完成，输出目录：{output_root}")
        return

    run_name = f"run_{args.modality}_{args.fusion}"
    summary = train_one_run(args, run_name, output_root)
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"训练完成，输出目录：{output_root}")


if __name__ == "__main__":
    main()
