"""Member C helpers: ResNet34 transfer learning, preprocessing conditions,
training/evaluation, Grad-CAM and checkpoint I/O.

Used by notebooks/04_ResNet34_GradCAM.ipynb (training + evaluation) and by the
LIVE DEMONSTRATION cell (inference only, no retraining).

Preprocessing reuses Member B's functions from common/preprocess.py:
  * baseline  : original grayscale image -> resize -> RGB -> ImageNet normalisation
  * fixed     : fixed_preprocess (CLAHE -> median -> unsharp) -> resize -> ...
  * adaptive  : adaptive_preprocess (quality-driven CLAHE / median / unsharp)
                with DEFAULT_THRESHOLDS -> resize -> ...
DIP is applied at the native image resolution (the resolution B calibrated its
thresholds on) and the result is then resized to 224x224.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import Dataset

from common.preprocess import (DEFAULT_THRESHOLDS, adaptive_preprocess,
                               fixed_preprocess)

# ---------------------------------------------------------------------------
# Constants (match the labels already used in shared/split.csv)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS_MODES = ("baseline", "fixed", "adaptive")
EXPECTED_SPLIT_COUNTS = {"train": 4361, "val": 754, "test": 1598}


# ---------------------------------------------------------------------------
# Data: split loading and path verification
# ---------------------------------------------------------------------------
def normalise_split_name(value: str) -> str:
    aliases = {"training": "train", "train": "train",
               "validation": "val", "valid": "val", "val": "val",
               "testing": "test", "test": "test"}
    return aliases.get(str(value).strip().lower(), str(value).strip().lower())


def resolve_image_path(filename: str, data_dir: Path) -> Path | None:
    """Map a split.csv filename (Windows or POSIX separators) to a real file."""
    rel = Path(str(filename).replace("\\", "/"))
    candidates = [data_dir / rel]
    if rel.parts and rel.parts[0].lower() == "datasets":
        candidates.append(data_dir.parent / rel)
        candidates.append(data_dir / Path(*rel.parts[1:]))
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_split(split_csv: Path, data_dir: Path, verify_paths: bool = True,
               check_counts: bool = True) -> pd.DataFrame:
    """Load shared/split.csv, verify counts, labels, leakage and image paths.

    Returns a DataFrame with columns: filename, label, split, y, path.
    Raises ValueError on any integrity problem. Never modifies the split file.
    """
    df = pd.read_csv(split_csv)
    missing_cols = {"filename", "label", "split"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"split.csv is missing columns: {missing_cols}")

    df = df.copy()
    df["split"] = df["split"].map(normalise_split_name)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    unknown = sorted(set(df["label"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown labels in split.csv: {unknown}")

    # Leakage: no file may appear in more than one split.
    dup = df[df.duplicated("filename", keep=False)]
    if not dup.empty:
        raise ValueError(f"{dup['filename'].nunique()} filenames appear in "
                         f"more than one row/split (data leakage risk).")

    counts = df["split"].value_counts().to_dict()
    if check_counts:
        for name, expected in EXPECTED_SPLIT_COUNTS.items():
            if counts.get(name, 0) != expected:
                raise ValueError(f"Split '{name}' has {counts.get(name, 0)} rows, "
                                 f"expected {expected}.")

    df["y"] = df["label"].map(CLASS_TO_IDX)
    paths = [resolve_image_path(f, data_dir) for f in df["filename"]]
    df["path"] = [str(p) if p is not None else None for p in paths]

    if verify_paths:
        missing = df.loc[df["path"].isna(), "filename"].tolist()
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images from split.csv were not found under "
                f"{data_dir}. First few: {missing[:5]}")
    return df


# ---------------------------------------------------------------------------
# Preprocessing (single source of truth for training, eval and the demo)
# ---------------------------------------------------------------------------
def load_gray_native(path: str | Path) -> np.ndarray:
    """Load an MRI as float32 grayscale in [0, 1] at its native resolution."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def preprocess_gray(path: str | Path, mode: str,
                    thresholds: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Apply the requested preprocessing condition.

    Returns (gray_224, operations). gray_224 is float32 HxW in [0, 1].
    """
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocessing mode '{mode}'. "
                         f"Choose from {PREPROCESS_MODES}.")
    gray = load_gray_native(path)
    ops: list[str] = []
    if mode == "fixed":
        gray = fixed_preprocess(gray)
        ops = ["CLAHE", "Median filtering", "Unsharp masking"]
    elif mode == "adaptive":
        th = thresholds or DEFAULT_THRESHOLDS
        gray, ops = adaptive_preprocess(
            gray,
            th["contrast_threshold"],
            th["noise_threshold"],
            th["sharpness_threshold"],
        )
    gray = cv2.resize(gray.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_LINEAR)
    return np.clip(gray, 0.0, 1.0), ops


def augment_gray(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light, label-preserving augmentation for MRI slices."""
    out = gray
    if rng.random() < 0.5:
        out = np.fliplr(out)
    angle = rng.uniform(-10, 10)
    h, w = out.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = out * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def gray_to_tensor(gray: np.ndarray) -> torch.Tensor:
    """HxW grayscale in [0,1] -> 3xHxW float tensor with ImageNet normalisation."""
    rgb = np.stack([gray, gray, gray], axis=0)  # convert to 3 channels
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return torch.from_numpy(((rgb - mean) / std).astype(np.float32))


class MRIDataset(Dataset):
    """Dataset over split.csv rows for one preprocessing condition."""

    def __init__(self, df: pd.DataFrame, mode: str,
                 thresholds: dict | None = None, augment: bool = False,
                 seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.thresholds = thresholds
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        gray, _ = preprocess_gray(row["path"], self.mode, self.thresholds)
        if self.augment:
            # Per-sample RNG keeps augmentation reproducible across workers.
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            gray = augment_gray(gray, rng)
        return gray_to_tensor(gray), int(row["y"]), i


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_resnet34(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet34 with ImageNet weights and a fresh 4-way final layer.

    Raises a clear error if pretrained weights cannot be downloaded, so that
    experiments never silently fall back to random initialisation.
    """
    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = torchvision.models.resnet34(weights=weights)
    except Exception as exc:  # network / cache failure
        raise RuntimeError(
            "Could not load ImageNet-pretrained ResNet34 weights. Check the "
            "internet connection or pre-download them to the torch cache. "
            f"Original error: {exc}") from exc
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    labels = list(range(NUM_CLASSES))
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    wf = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {"accuracy": acc, "macro_precision": mp, "macro_recall": mr,
            "macro_f1": mf, "weighted_f1": wf}


def per_class_metrics(y_true, y_pred) -> pd.DataFrame:
    labels = list(range(NUM_CLASSES))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": CLASS_NAMES, "precision": p, "recall": r,
                         "f1": f, "support": s})


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------
def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, ys, ps = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            ys.append(y.cpu().numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    m = compute_metrics(y_true, y_pred)
    return {"loss": total_loss / len(y_true), **m}


def train_with_early_stopping(model, train_loader, val_loader, device,
                              epochs=15, lr=1e-4, weight_decay=1e-4,
                              patience=5, log_fn=print):
    """Fine-tune all layers; select the epoch with the best validation macro F1.

    Returns (best_model_on_cpu_state_loaded, history_df, best_epoch).
    Only validation data influences model selection.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history, best_f1, best_epoch, best_state, bad = [], -1.0, -1, None, 0
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, device, optimizer)
        va = _run_epoch(model, val_loader, criterion, device, None)
        scheduler.step()
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        log_fn(f"epoch {epoch:02d} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} "
               f"F1 {tr['macro_f1']:.4f} | val loss {va['loss']:.4f} "
               f"acc {va['accuracy']:.4f} F1 {va['macro_f1']:.4f}")
        if va["macro_f1"] > best_f1:
            best_f1, best_epoch = va["macro_f1"], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log_fn(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


@torch.no_grad()
def predict_loader(model, loader, device):
    """Returns (logits [N,4], probs [N,4], y_true [N], row_indices [N])."""
    model.eval()
    logits_all, ys, idxs = [], [], []
    for x, y, idx in loader:
        logits_all.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
        idxs.append(idx.numpy())
    logits = np.concatenate(logits_all)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return logits, probs, np.concatenate(ys), np.concatenate(idxs)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: str | Path, state_dict: dict, preprocess_mode: str,
                    thresholds: dict | None, val_metrics: dict,
                    training_config: dict) -> None:
    """Save weights plus everything needed to reproduce inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": state_dict,
        "architecture": "resnet34",
        "model_settings": {"num_classes": NUM_CLASSES, "fc_replaced": True,
                           "init": "ImageNet1K_V1 then fine-tuned"},
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "preprocessing": {
            "mode": preprocess_mode,  # 'baseline' | 'fixed' | 'adaptive'
            "image_size": IMAGE_SIZE,
            "channels": "grayscale replicated to RGB",
            "resize": "native resolution -> 224x224 (bilinear)",
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "dip_applied_at": "native resolution, before resize",
            "adaptive_thresholds": thresholds if preprocess_mode == "adaptive" else None,
        },
        "val_metrics": val_metrics,
        "training_config": training_config,
    }, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None):
    """Load a saved checkpoint for inference. Returns (model_in_eval_mode, ckpt)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet34(num_classes=ckpt["model_settings"]["num_classes"],
                           pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM on a chosen layer (default for ResNet34: model.layer4[-1]).

    This is an approximate visual explanation of where the network looked. It is
    NOT a tumour segmentation.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: 1x3xHxW tensor on the model's device. Returns (cam HxW in [0,1], class_idx, probs)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        grads = torch.autograd.grad(score, self.activations, retain_graph=False)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32), class_idx, probs


def overlay_cam(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a JET heatmap over a grayscale image. Returns HxWx3 float in [0,1]."""
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    heat = heat[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    base = np.stack([gray] * 3, axis=-1)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def explain_image(model, gray_224: np.ndarray, device, target_layer=None,
                  class_idx: int | None = None):
    """Run Grad-CAM for an already-preprocessed 224x224 grayscale image."""
    target_layer = target_layer if target_layer is not None else model.layer4[-1]
    # Register the forward hook once per (model, layer); repeated calls reuse it.
    cam_obj = getattr(model, "_gradcam_obj", None)
    if cam_obj is None or cam_obj.target_layer is not target_layer:
        cam_obj = GradCAM(model, target_layer)
        model._gradcam_obj = cam_obj
    x = gray_to_tensor(gray_224).unsqueeze(0).to(device)
    cam, pred, probs = cam_obj(x, class_idx=class_idx)
    return cam, pred, probs


def predict_single(model, ckpt: dict, path: str | Path, device, thresholds=None):
    """Full single-image inference with the checkpoint's own preprocessing.

    Returns a dict with gray (224), cam, overlay, predicted class, confidence,
    probabilities and the applied preprocessing operations.
    """
    mode = ckpt["preprocessing"]["mode"]
    th = thresholds or ckpt["preprocessing"].get("adaptive_thresholds") or DEFAULT_THRESHOLDS
    gray, ops = preprocess_gray(path, mode, th)
    cam, pred, probs = explain_image(model, gray, device)
    return {
        "gray": gray, "cam": cam, "overlay": overlay_cam(gray, cam),
        "pred_idx": pred, "pred_label": IDX_TO_CLASS[pred],
        "confidence": float(probs[pred]), "probs": probs, "ops": ops, "mode": mode,
    }
