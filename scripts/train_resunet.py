import json
import os
import random
import shutil
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from augment_data import SpectralAugmenter
from models.resunet1d import ResUNet1D
from pair_data import FTIRPairsDataset, denormalize_output, get_pair_file_lists, split_pair_files
from spectral_eval_utils import aggregate_spectrum_metrics, evaluate_spectrum, quality_score
from spectral_losses import CompositeSpectralLoss

PAIRS_DIR = os.getenv("PAIRS_DIR", os.path.join(_project_root, "data", "pairs"))
TARGET_LEN = int(os.getenv("TARGET_LEN", "1868"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "24"))
MAX_LR = float(os.getenv("MAX_LR", "8e-5"))
MIN_LR = float(os.getenv("MIN_LR", "8e-6"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "5e-5"))
BASE_CHANNELS = int(os.getenv("BASE_CHANNELS", "64"))
SEED = int(os.getenv("SEED", "42"))
PATIENCE = int(os.getenv("PATIENCE", "8"))
USE_AUGMENT = os.getenv("AUGMENT", "1") == "1"
NORMALIZATION = os.getenv("NORMALIZATION", "joint_zscore")
EMA_DECAY = float(os.getenv("EMA_DECAY", "0.995"))
RESIDUAL_LEARNING = os.getenv("RESIDUAL_LEARNING", "1") == "1"
NORM_TYPE = os.getenv("NORM_TYPE", "group")
USE_SKIP_GATES = os.getenv("USE_SKIP_GATES", "1") == "1"
USE_SE = os.getenv("USE_SE", "1") == "1"
USE_INPUT_MEDIAN = os.getenv("USE_INPUT_MEDIAN", "0") == "1"
INPUT_MEDIAN_KERNEL = int(os.getenv("INPUT_MEDIAN_KERNEL", "5"))
INPUT_MEDIAN_BLEND = float(os.getenv("INPUT_MEDIAN_BLEND", "0.15"))
USE_SPIKE_SUPPRESSOR = os.getenv("USE_SPIKE_SUPPRESSOR", "0") == "1"
SPIKE_SUPPRESSOR_THRESHOLD = float(os.getenv("SPIKE_SUPPRESSOR_THRESHOLD", "2.5"))
SPIKE_SUPPRESSOR_BLEND = float(os.getenv("SPIKE_SUPPRESSOR_BLEND", "0.85"))
SPIKE_SUPPRESSOR_EDGE_POINTS = int(os.getenv("SPIKE_SUPPRESSOR_EDGE_POINTS", "96"))
SPIKE_SUPPRESSOR_EDGE_GAIN = float(os.getenv("SPIKE_SUPPRESSOR_EDGE_GAIN", "1.8"))
USE_MULTISCALE_CONTEXT = os.getenv("USE_MULTISCALE_CONTEXT", "1") == "1"
USE_DETAIL_HEAD = os.getenv("USE_DETAIL_HEAD", "1") == "1"
USE_POSITIONAL_BIAS = os.getenv("USE_POSITIONAL_BIAS", "1") == "1"
USE_DERIVATIVE_BIAS = os.getenv("USE_DERIVATIVE_BIAS", "1") == "1"
USE_LOCAL_REFINER = os.getenv("USE_LOCAL_REFINER", "1") == "1"
USE_SPECTRAL_ATTENTION = os.getenv("USE_SPECTRAL_ATTENTION", "1") == "1"
SPECTRAL_ATTENTION_PRIOR = float(os.getenv("SPECTRAL_ATTENTION_PRIOR", "0.0"))
USE_STRIDED_DOWNSAMPLE = os.getenv("USE_STRIDED_DOWNSAMPLE", "0") == "1"
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(_project_root, "models", "resunet1d_scientific_candidate.pth"))
PROMOTE_MODEL_PATH = os.getenv(
    "PROMOTE_MODEL_PATH",
    os.path.join(_project_root, "models", "resunet1d_single_stage_final.pth"),
)
PROMOTE_RESULTS_PATH = os.getenv(
    "PROMOTE_RESULTS_PATH",
    os.path.join(_project_root, "results", "resunet_single_stage_final.json"),
)
MIN_PROMOTION_QUALITY = float(os.getenv("MIN_PROMOTION_QUALITY", "95.0"))
AUTO_PROMOTE = os.getenv("AUTO_PROMOTE", "1") == "1"
REQUIRE_QUALITY_IMPROVEMENT = os.getenv("REQUIRE_QUALITY_IMPROVEMENT", "1") == "1"
MIN_IMPROVEMENT_DELTA = float(os.getenv("MIN_IMPROVEMENT_DELTA", "0.0"))
MAX_PROMOTION_FP_RMSE = float(os.getenv("MAX_PROMOTION_FP_RMSE", "inf"))
MAX_PROMOTION_PEAK_SHIFT = float(os.getenv("MAX_PROMOTION_PEAK_SHIFT", "inf"))
MAX_PROMOTION_PEAK_AMP_MAE = float(os.getenv("MAX_PROMOTION_PEAK_AMP_MAE", "inf"))
RESULTS_PATH = os.getenv(
    "RESULTS_PATH",
    os.path.join(_project_root, "results", "resunet_scientific_candidate_metrics.json"),
)
INIT_MODEL_PATH = os.getenv("INIT_MODEL_PATH", PROMOTE_MODEL_PATH if os.path.isfile(PROMOTE_MODEL_PATH) else "")
LOSS_W_MSE = float(os.getenv("LOSS_W_MSE", "0.75"))
LOSS_W_L1 = float(os.getenv("LOSS_W_L1", "0.18"))
LOSS_W_D1 = float(os.getenv("LOSS_W_D1", "0.50"))
LOSS_W_D2 = float(os.getenv("LOSS_W_D2", "0.45"))
LOSS_W_TV = float(os.getenv("LOSS_W_TV", "0.0012"))
LOSS_W_FFT = float(os.getenv("LOSS_W_FFT", "0.05"))
LOSS_W_AMP = float(os.getenv("LOSS_W_AMP", "0.26"))
LOSS_W_BASELINE = float(os.getenv("LOSS_W_BASELINE", "0.02"))
LOSS_W_PEAK_PROFILE = float(os.getenv("LOSS_W_PEAK_PROFILE", "0.58"))
LOSS_W_PEAK_CENTER = float(os.getenv("LOSS_W_PEAK_CENTER", "1.35"))
LOSS_W_EDGE_L1 = float(os.getenv("LOSS_W_EDGE_L1", "0.06"))
LOSS_W_EDGE_D1 = float(os.getenv("LOSS_W_EDGE_D1", "0.05"))
LOSS_EDGE_POINTS = int(os.getenv("LOSS_EDGE_POINTS", "96"))
PEAK_WINDOW_RADIUS = int(os.getenv("PEAK_WINDOW_RADIUS", "4"))
PEAK_QUANTILE = float(os.getenv("PEAK_QUANTILE", "0.80"))
PEAK_SOFTMAX_TEMP = float(os.getenv("PEAK_SOFTMAX_TEMP", "28.0"))
MASK_LEADING_POINTS = int(os.getenv("MASK_LEADING_POINTS", "3"))
LOSS_W_SMOOTH_CONSISTENCY = float(os.getenv("LOSS_W_SMOOTH_CONSISTENCY", "0.01"))
SMOOTH_KERNEL_SIZE = int(os.getenv("SMOOTH_KERNEL_SIZE", "9"))
LOSS_W_FINGERPRINT_L1 = float(os.getenv("LOSS_W_FINGERPRINT_L1", "0.28"))
LOSS_W_FINGERPRINT_MSE = float(os.getenv("LOSS_W_FINGERPRINT_MSE", "0.22"))
LOSS_W_FINGERPRINT_D1 = float(os.getenv("LOSS_W_FINGERPRINT_D1", "0.18"))
FINGERPRINT_START = int(os.getenv("FINGERPRINT_START", "104"))
FINGERPRINT_END = int(os.getenv("FINGERPRINT_END", "726"))
LOSS_W_CURVATURE_MSE = float(os.getenv("LOSS_W_CURVATURE_MSE", "0.40"))
CURVATURE_SCALE = float(os.getenv("CURVATURE_SCALE", "12.0"))
LOSS_W_VALLEY_UNDER = float(os.getenv("LOSS_W_VALLEY_UNDER", "0.30"))
VALLEY_QUANTILE = float(os.getenv("VALLEY_QUANTILE", "0.18"))
LOSS_W_VALLEY_CENTER = float(os.getenv("LOSS_W_VALLEY_CENTER", "0.20"))
LOSS_W_PEAK_ALIGN = float(os.getenv("LOSS_W_PEAK_ALIGN", "0.38"))
PEAK_ALIGN_WINDOW = int(os.getenv("PEAK_ALIGN_WINDOW", "5"))
PEAK_ALIGN_WEIGHT = float(os.getenv("PEAK_ALIGN_WEIGHT", "2.8"))
HARD_SAMPLE_FILE = os.getenv("HARD_SAMPLE_FILE", "")
HARD_SAMPLE_MULTIPLIER = float(os.getenv("HARD_SAMPLE_MULTIPLIER", "1.0"))
PEAK_POINT_WEIGHT = float(os.getenv("PEAK_POINT_WEIGHT", "2.7"))
SLOPE_POINT_WEIGHT = float(os.getenv("SLOPE_POINT_WEIGHT", "1.1"))
FINGERPRINT_POINT_WEIGHT = float(os.getenv("FINGERPRINT_POINT_WEIGHT", "3.0"))
PEAK_FOCUS_START = int(os.getenv("PEAK_FOCUS_START", str(FINGERPRINT_START)))
PEAK_FOCUS_END = int(os.getenv("PEAK_FOCUS_END", str(FINGERPRINT_END)))
LOSS_W_PEAK_DOMINANCE = float(os.getenv("LOSS_W_PEAK_DOMINANCE", "0.12"))
PEAK_DOMINANCE_MARGIN_SCALE = float(os.getenv("PEAK_DOMINANCE_MARGIN_SCALE", "0.25"))
LOSS_W_EXTREMA_MSE = float(os.getenv("LOSS_W_EXTREMA_MSE", "0.22"))
EXTREMA_WINDOW = int(os.getenv("EXTREMA_WINDOW", "7"))
EXTREMA_WEIGHT = float(os.getenv("EXTREMA_WEIGHT", "3.0"))
LOSS_W_SOBOLEV_L1 = float(os.getenv("LOSS_W_SOBOLEV_L1", "0.14"))
LOSS_W_BASELINE_DRIFT = float(os.getenv("LOSS_W_BASELINE_DRIFT", "0.08"))
BASELINE_DRIFT_KERNEL = int(os.getenv("BASELINE_DRIFT_KERNEL", "65"))
CHECKPOINT_SCORE_MODE = os.getenv("CHECKPOINT_SCORE_MODE", "review")
REQUIRE_CUDA_FOR_TRAINING = os.getenv("REQUIRE_CUDA_FOR_TRAINING", "1") == "1"
TRAINABLE_SCOPE = os.getenv("TRAINABLE_SCOPE", "all")


def get_device() -> torch.device:
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1" or REQUIRE_CUDA_FOR_TRAINING
    if torch.cuda.is_available():
        return torch.device("cuda")
    if force_cuda:
        raise RuntimeError(
            "CUDA is required for training, but torch.cuda.is_available() is False. "
            "Fix NVIDIA driver/runtime or set REQUIRE_CUDA_FOR_TRAINING=0 for an explicit CPU debug run."
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def safe_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and np.isnan(value):
        return 0.0
    return float(value)


def current_promoted_quality() -> float | None:
    if not os.path.isfile(PROMOTE_RESULTS_PATH):
        return None
    try:
        with open(PROMOTE_RESULTS_PATH) as f:
            results = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    value = results.get("overall_quality")
    if value is None:
        return None
    return safe_float(value)


def selection_score(metrics: Dict[str, float]) -> float:
    overall, _, _, _, _, _ = quality_score(
        metrics["mean_mse"],
        metrics["mean_psnr"],
        metrics["mean_ssim"],
        metrics["mean_corr"],
    )
    return (
        overall
        - 6.0 * safe_float(metrics.get("mean_noise_floor_std"))
        - 1.0 * safe_float(metrics.get("mean_peak_amp_mae"))
        - 0.15 * safe_float(metrics.get("mean_peak_shift_mean_abs"))
    )


def review_score(metrics: Dict[str, float]) -> float:
    overall, _, _, _, _, _ = quality_score(
        metrics["mean_mse"],
        metrics["mean_psnr"],
        metrics["mean_ssim"],
        metrics["mean_corr"],
    )
    return (
        overall
        - 10.0 * safe_float(metrics.get("mean_rmse_fingerprint"))
        - 0.75 * safe_float(metrics.get("mean_peak_shift_mean_abs"))
        - 0.20 * safe_float(metrics.get("mean_peak_amp_mae"))
        - 3.0 * safe_float(metrics.get("mean_noise_floor_std"))
    )


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                param.copy_(self.backup[name])


def create_model() -> ResUNet1D:
    return ResUNet1D(
        base_channels=BASE_CHANNELS,
        residual_learning=RESIDUAL_LEARNING,
        norm_type=NORM_TYPE,
        use_skip_gates=USE_SKIP_GATES,
        use_se=USE_SE,
        use_input_median=USE_INPUT_MEDIAN,
        input_median_kernel=INPUT_MEDIAN_KERNEL,
        input_median_blend=INPUT_MEDIAN_BLEND,
        use_spike_suppressor=USE_SPIKE_SUPPRESSOR,
        spike_suppressor_threshold=SPIKE_SUPPRESSOR_THRESHOLD,
        spike_suppressor_blend=SPIKE_SUPPRESSOR_BLEND,
        spike_suppressor_edge_points=SPIKE_SUPPRESSOR_EDGE_POINTS,
        spike_suppressor_edge_gain=SPIKE_SUPPRESSOR_EDGE_GAIN,
        use_multiscale_context=USE_MULTISCALE_CONTEXT,
        use_detail_head=USE_DETAIL_HEAD,
        use_positional_bias=USE_POSITIONAL_BIAS,
        use_derivative_bias=USE_DERIVATIVE_BIAS,
        use_local_refiner=USE_LOCAL_REFINER,
        use_spectral_attention=USE_SPECTRAL_ATTENTION,
        target_len=TARGET_LEN,
        attention_focus_start=FINGERPRINT_START,
        attention_focus_end=FINGERPRINT_END,
        spectral_attention_prior=SPECTRAL_ATTENTION_PRIOR,
        use_strided_downsample=USE_STRIDED_DOWNSAMPLE,
    )


def configure_trainable_scope(model: nn.Module, scope: str) -> int:
    scope = scope.lower().strip()
    if scope == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif scope == "head":
        trainable_prefixes = ("out_conv", "detail_refiner", "local_refiner")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    elif scope == "decoder":
        trainable_prefixes = ("dec", "out_conv", "detail_refiner", "local_refiner")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    elif scope == "downsample_decoder":
        trainable_prefixes = ("down", "dec", "out_conv", "detail_refiner", "local_refiner")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    elif scope == "attention_head":
        trainable_prefixes = ("spectral_attn", "out_conv", "detail_refiner", "local_refiner")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    else:
        raise ValueError(
            f"Unsupported TRAINABLE_SCOPE={scope!r}. "
            "Use one of: all, head, decoder, downsample_decoder, attention_head."
        )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable_params <= 0:
        raise ValueError(f"TRAINABLE_SCOPE={scope!r} left no trainable parameters")
    return trainable_params


def load_hard_sample_weights(train_clean_files: List[str]):
    if not HARD_SAMPLE_FILE or not os.path.isfile(HARD_SAMPLE_FILE) or HARD_SAMPLE_MULTIPLIER <= 1.0:
        return None, 0

    with open(HARD_SAMPLE_FILE) as f:
        hard_ids = {line.strip() for line in f if line.strip()}

    weights = []
    matched = 0
    for path in train_clean_files:
        sample_id = os.path.basename(path).replace("_clean.npy", "")
        if sample_id in hard_ids:
            weights.append(HARD_SAMPLE_MULTIPLIER)
            matched += 1
        else:
            weights.append(1.0)
    if matched == 0:
        return None, 0
    return torch.as_tensor(weights, dtype=torch.double), matched


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    sample_names: List[str],
    keep_cases: bool = False,
):
    model.eval()
    val_loss = 0.0
    per_sample = []
    sample_index = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            val_loss += criterion(output, clean).item()

            output_np = output.detach().cpu().numpy()
            for batch_offset, o in enumerate(output_np):
                clean_raw, noisy_raw = val_loader.dataset.get_raw_pair(sample_index)
                output_raw = denormalize_output(o.squeeze(), clean_raw, noisy_raw, normalization=NORMALIZATION)
                metrics = evaluate_spectrum(output_raw, clean_raw)
                if keep_cases:
                    metrics["sample_name"] = sample_names[sample_index]
                per_sample.append(metrics)
                sample_index += 1

    val_loss /= len(val_loader)
    aggregate = aggregate_spectrum_metrics(per_sample)
    aggregate["val_loss"] = float(val_loss)
    aggregate["selection_score"] = selection_score(aggregate)
    aggregate["review_score"] = review_score(aggregate)
    overall, mse_q, psnr_q, ssim_q, corr_q, weights = quality_score(
        aggregate["mean_mse"],
        aggregate["mean_psnr"],
        aggregate["mean_ssim"],
        aggregate["mean_corr"],
    )
    aggregate["overall_quality"] = overall
    aggregate["mse_quality"] = mse_q
    aggregate["psnr_quality"] = psnr_q
    aggregate["ssim_quality"] = ssim_q
    aggregate["corr_quality"] = corr_q
    aggregate["weights"] = weights
    return aggregate, per_sample


def create_scheduler(optimizer):
    warmup_epochs = min(4, max(2, EPOCHS // 10))
    if warmup_epochs >= EPOCHS:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(MIN_LR / MAX_LR, 0.2),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(EPOCHS - warmup_epochs, 1),
        eta_min=MIN_LR,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def main():
    set_seed(SEED)
    device = get_device()
    if AUTO_PROMOTE and os.path.abspath(MODEL_PATH) == os.path.abspath(PROMOTE_MODEL_PATH):
        raise ValueError(
            "Unsafe training configuration: MODEL_PATH points at PROMOTE_MODEL_PATH. "
            "Use a separate candidate MODEL_PATH so the promoted model cannot be overwritten before the 95% gate."
        )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: len={TARGET_LEN}, norm={NORMALIZATION}, residual_learning={RESIDUAL_LEARNING}, "
        f"norm_type={NORM_TYPE}, gates={USE_SKIP_GATES}, se={USE_SE}, "
        f"input_median={USE_INPUT_MEDIAN}, median_kernel={INPUT_MEDIAN_KERNEL}, "
        f"median_blend={INPUT_MEDIAN_BLEND}, spike_suppressor={USE_SPIKE_SUPPRESSOR}, "
        f"spike_thr={SPIKE_SUPPRESSOR_THRESHOLD}, spike_blend={SPIKE_SUPPRESSOR_BLEND}, "
        f"multiscale={USE_MULTISCALE_CONTEXT}, detail_head={USE_DETAIL_HEAD}, "
        f"positional_bias={USE_POSITIONAL_BIAS}, derivative_bias={USE_DERIVATIVE_BIAS}, local_refiner={USE_LOCAL_REFINER}, "
        f"spectral_attention={USE_SPECTRAL_ATTENTION}, spectral_attention_prior={SPECTRAL_ATTENTION_PRIOR}, "
        f"strided_downsample={USE_STRIDED_DOWNSAMPLE}, "
        f"mask_leading_points={MASK_LEADING_POINTS}, "
        f"smooth_w={LOSS_W_SMOOTH_CONSISTENCY}, smooth_k={SMOOTH_KERNEL_SIZE}, "
        f"fp_l1={LOSS_W_FINGERPRINT_L1}, fp_mse={LOSS_W_FINGERPRINT_MSE}, fp_d1={LOSS_W_FINGERPRINT_D1}, "
        f"curv_mse={LOSS_W_CURVATURE_MSE}, curv_scale={CURVATURE_SCALE}, "
        f"valley_under={LOSS_W_VALLEY_UNDER}, valley_center={LOSS_W_VALLEY_CENTER}, valley_q={VALLEY_QUANTILE}, "
        f"peak_align={LOSS_W_PEAK_ALIGN}, peak_align_win={PEAK_ALIGN_WINDOW}, "
        f"extrema_mse={LOSS_W_EXTREMA_MSE}, sobolev_l1={LOSS_W_SOBOLEV_L1}, baseline_drift={LOSS_W_BASELINE_DRIFT}, "
        f"peak_point_w={PEAK_POINT_WEIGHT}, slope_point_w={SLOPE_POINT_WEIGHT}, fp_point_w={FINGERPRINT_POINT_WEIGHT}, "
        f"peak_focus=({PEAK_FOCUS_START},{PEAK_FOCUS_END}), "
        f"peak_dom={LOSS_W_PEAK_DOMINANCE}, peak_dom_margin={PEAK_DOMINANCE_MARGIN_SCALE}, "
        f"candidate_model={MODEL_PATH}, promote_model={PROMOTE_MODEL_PATH}, min_quality={MIN_PROMOTION_QUALITY}, "
        f"require_improvement={REQUIRE_QUALITY_IMPROVEMENT}, min_delta={MIN_IMPROVEMENT_DELTA}, "
        f"max_fp_rmse={MAX_PROMOTION_FP_RMSE}, "
        f"max_peak_shift={MAX_PROMOTION_PEAK_SHIFT}, max_peak_amp_mae={MAX_PROMOTION_PEAK_AMP_MAE}, "
        f"require_cuda={REQUIRE_CUDA_FOR_TRAINING}, trainable_scope={TRAINABLE_SCOPE}, "
        f"score_mode={CHECKPOINT_SCORE_MODE}, "
        f"hard_file={HARD_SAMPLE_FILE or 'none'}, hard_mult={HARD_SAMPLE_MULTIPLIER}, "
        f"batch={BATCH_SIZE}, epochs={EPOCHS}, max_lr={MAX_LR}, min_lr={MIN_LR}, wd={WEIGHT_DECAY}"
    )

    clean_files, noisy_files = get_pair_file_lists(PAIRS_DIR)
    train_c, val_c, train_n, val_n = split_pair_files(clean_files, noisy_files, test_size=0.2, seed=SEED)
    val_sample_names = [os.path.basename(path).replace("_clean.npy", "") for path in val_c]

    augmenter = SpectralAugmenter() if USE_AUGMENT else None

    train_ds = FTIRPairsDataset(
        train_c,
        train_n,
        target_len=TARGET_LEN,
        normalization=NORMALIZATION,
        augment_fn=augmenter,
        cache_in_memory=True,
    )
    val_ds = FTIRPairsDataset(
        val_c,
        val_n,
        target_len=TARGET_LEN,
        normalization=NORMALIZATION,
        augment_fn=None,
        cache_in_memory=True,
    )

    train_weights, hard_matched = load_hard_sample_weights(train_c)
    train_sampler = None
    train_shuffle = True
    if train_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
        )
        train_shuffle = False
        print(f"Using hard-sample oversampling from {HARD_SAMPLE_FILE} (matched {hard_matched} train samples, multiplier={HARD_SAMPLE_MULTIPLIER})")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = create_model().to(device)
    if INIT_MODEL_PATH and os.path.isfile(INIT_MODEL_PATH):
        model.load_state_dict(torch.load(INIT_MODEL_PATH, map_location=device, weights_only=True), strict=False)
        print(f"Initialized from {INIT_MODEL_PATH}")
    trainable_params = configure_trainable_scope(model, TRAINABLE_SCOPE)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Trainable scope: {TRAINABLE_SCOPE} ({trainable_params / 1e6:.3f}M / {total_params / 1e6:.3f}M parameters)")

    optimizer = optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = create_scheduler(optimizer)
    criterion = CompositeSpectralLoss(
        w_mse=LOSS_W_MSE,
        w_l1=LOSS_W_L1,
        w_d1=LOSS_W_D1,
        w_d2=LOSS_W_D2,
        w_tv=LOSS_W_TV,
        w_fft=LOSS_W_FFT,
        w_amp=LOSS_W_AMP,
        w_baseline=LOSS_W_BASELINE,
        w_peak_profile=LOSS_W_PEAK_PROFILE,
        w_peak_center=LOSS_W_PEAK_CENTER,
        w_edge_l1=LOSS_W_EDGE_L1,
        w_edge_d1=LOSS_W_EDGE_D1,
        edge_points=LOSS_EDGE_POINTS,
        peak_window_radius=PEAK_WINDOW_RADIUS,
        peak_quantile=PEAK_QUANTILE,
        peak_softmax_temp=PEAK_SOFTMAX_TEMP,
        mask_leading_points=MASK_LEADING_POINTS,
        w_smooth_consistency=LOSS_W_SMOOTH_CONSISTENCY,
        smooth_kernel_size=SMOOTH_KERNEL_SIZE,
        w_fingerprint_l1=LOSS_W_FINGERPRINT_L1,
        w_fingerprint_mse=LOSS_W_FINGERPRINT_MSE,
        w_fingerprint_d1=LOSS_W_FINGERPRINT_D1,
        fingerprint_start=FINGERPRINT_START,
        fingerprint_end=FINGERPRINT_END,
        w_curvature_mse=LOSS_W_CURVATURE_MSE,
        curvature_scale=CURVATURE_SCALE,
        w_valley_under=LOSS_W_VALLEY_UNDER,
        valley_quantile=VALLEY_QUANTILE,
        w_valley_center=LOSS_W_VALLEY_CENTER,
        w_peak_align=LOSS_W_PEAK_ALIGN,
        peak_align_window=PEAK_ALIGN_WINDOW,
        peak_align_weight=PEAK_ALIGN_WEIGHT,
        peak_point_weight=PEAK_POINT_WEIGHT,
        slope_point_weight=SLOPE_POINT_WEIGHT,
        fingerprint_point_weight=FINGERPRINT_POINT_WEIGHT,
        peak_focus_start=PEAK_FOCUS_START,
        peak_focus_end=PEAK_FOCUS_END,
        w_peak_dominance=LOSS_W_PEAK_DOMINANCE,
        peak_dominance_margin_scale=PEAK_DOMINANCE_MARGIN_SCALE,
        w_extrema_mse=LOSS_W_EXTREMA_MSE,
        extrema_window=EXTREMA_WINDOW,
        extrema_weight=EXTREMA_WEIGHT,
        w_sobolev_l1=LOSS_W_SOBOLEV_L1,
        w_baseline_drift=LOSS_W_BASELINE_DRIFT,
        baseline_drift_kernel=BASELINE_DRIFT_KERNEL,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = ExponentialMovingAverage(model, decay=EMA_DECAY)

    best_score = -float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        if augmenter is not None:
            augmenter.set_epoch(epoch, EPOCHS)

        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                output = model(noisy)
                loss = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            train_loss += loss.item()

        train_loss /= len(train_loader)

        ema.apply_to(model)
        val_metrics, _ = evaluate_validation(model, val_loader, criterion, device, val_sample_names)
        ema.restore(model)

        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_metrics["val_loss"],
                "selection_score": val_metrics["selection_score"],
                "review_score": val_metrics["review_score"],
                "overall_quality": val_metrics["overall_quality"],
                "mean_psnr": val_metrics["mean_psnr"],
                "mean_ssim": val_metrics["mean_ssim"],
                "mean_corr": val_metrics["mean_corr"],
                "mean_noise_floor_std": val_metrics["mean_noise_floor_std"],
                "mean_rmse_fingerprint": val_metrics["mean_rmse_fingerprint"],
                "mean_peak_amp_mae": val_metrics["mean_peak_amp_mae"],
                "mean_peak_shift_mean_abs": val_metrics["mean_peak_shift_mean_abs"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if CHECKPOINT_SCORE_MODE == "overall":
            score_value = val_metrics["overall_quality"]
        elif CHECKPOINT_SCORE_MODE == "review":
            score_value = val_metrics["review_score"]
        else:
            score_value = val_metrics["selection_score"]

        if score_value > best_score:
            best_score = score_value
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            ema.apply_to(model)
            torch.save(model.state_dict(), MODEL_PATH)
            ema.restore(model)
            marker = "  <- saved"
        else:
            patience_counter += 1
            marker = ""

        if epoch <= 5 or epoch % 5 == 0 or marker:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {train_loss:.5f} | Val: {val_metrics['val_loss']:.5f} | "
                f"Score[{CHECKPOINT_SCORE_MODE}]: {score_value:.3f} | "
                f"Q: {val_metrics['overall_quality']:.2f}% | "
                f"FP-RMSE: {safe_float(val_metrics['mean_rmse_fingerprint']):.4f} | "
                f"NoiseFloor: {safe_float(val_metrics['mean_noise_floor_std']):.4f} | "
                f"PeakAmpMAE: {safe_float(val_metrics['mean_peak_amp_mae']):.4f} | "
                f"PeakShift: {safe_float(val_metrics['mean_peak_shift_mean_abs']):.3f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}{marker}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    best_model = create_model().to(device)
    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    final_metrics, final_cases = evaluate_validation(
        best_model,
        val_loader,
        criterion,
        device,
        val_sample_names,
        keep_cases=True,
    )

    def rank_key(item):
        peak_penalty = 10.0 if safe_float(item.get("num_peaks")) <= 0.0 else 0.0
        return (
            safe_float(item.get("rmse_fingerprint"))
            + 0.75 * safe_float(item.get("rmse_high_wavenumber"))
            + 0.5 * safe_float(item.get("peak_amp_mae"))
            + 0.2 * safe_float(item.get("peak_shift_mean_abs"))
            + peak_penalty
        )

    peaked_cases = [case for case in final_cases if safe_float(case.get("num_peaks")) > 0.0]
    flat_cases = [case for case in final_cases if safe_float(case.get("num_peaks")) <= 0.0]

    ranked_pool = peaked_cases if peaked_cases else final_cases
    worst_cases = sorted(ranked_pool, key=rank_key, reverse=True)[:10]
    best_cases = sorted(ranked_pool, key=rank_key)[:10]
    best_flat_cases = sorted(flat_cases, key=lambda item: safe_float(item.get("rmse_fingerprint")))[:10]

    results = {
        "model": "ResUNet1D",
        "stage": "single_stage",
        "target_len": TARGET_LEN,
        "normalization": NORMALIZATION,
        "residual_learning": RESIDUAL_LEARNING,
        "norm_type": NORM_TYPE,
        "use_skip_gates": USE_SKIP_GATES,
        "use_se": USE_SE,
        "use_input_median": USE_INPUT_MEDIAN,
        "input_median_kernel": INPUT_MEDIAN_KERNEL,
        "input_median_blend": INPUT_MEDIAN_BLEND,
        "use_spike_suppressor": USE_SPIKE_SUPPRESSOR,
        "spike_suppressor_threshold": SPIKE_SUPPRESSOR_THRESHOLD,
        "spike_suppressor_blend": SPIKE_SUPPRESSOR_BLEND,
        "spike_suppressor_edge_points": SPIKE_SUPPRESSOR_EDGE_POINTS,
        "spike_suppressor_edge_gain": SPIKE_SUPPRESSOR_EDGE_GAIN,
        "use_multiscale_context": USE_MULTISCALE_CONTEXT,
        "use_detail_head": USE_DETAIL_HEAD,
        "use_positional_bias": USE_POSITIONAL_BIAS,
        "use_derivative_bias": USE_DERIVATIVE_BIAS,
        "use_local_refiner": USE_LOCAL_REFINER,
        "use_spectral_attention": USE_SPECTRAL_ATTENTION,
        "spectral_attention_prior": SPECTRAL_ATTENTION_PRIOR,
        "use_strided_downsample": USE_STRIDED_DOWNSAMPLE,
        "base_channels": BASE_CHANNELS,
        "train_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_lr": MAX_LR,
            "min_lr": MIN_LR,
            "weight_decay": WEIGHT_DECAY,
            "ema_decay": EMA_DECAY,
            "patience": PATIENCE,
            "seed": SEED,
            "augment": USE_AUGMENT,
            "checkpoint_score_mode": CHECKPOINT_SCORE_MODE,
            "require_cuda_for_training": REQUIRE_CUDA_FOR_TRAINING,
            "min_promotion_quality": MIN_PROMOTION_QUALITY,
            "auto_promote": AUTO_PROMOTE,
            "require_quality_improvement": REQUIRE_QUALITY_IMPROVEMENT,
            "min_improvement_delta": MIN_IMPROVEMENT_DELTA,
            "max_promotion_fp_rmse": MAX_PROMOTION_FP_RMSE,
            "max_promotion_peak_shift": MAX_PROMOTION_PEAK_SHIFT,
            "max_promotion_peak_amp_mae": MAX_PROMOTION_PEAK_AMP_MAE,
            "trainable_scope": TRAINABLE_SCOPE,
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
        "loss_config": {
            "w_mse": LOSS_W_MSE,
            "w_l1": LOSS_W_L1,
            "w_d1": LOSS_W_D1,
            "w_d2": LOSS_W_D2,
            "w_tv": LOSS_W_TV,
            "w_fft": LOSS_W_FFT,
            "w_amp": LOSS_W_AMP,
            "w_baseline": LOSS_W_BASELINE,
            "w_peak_profile": LOSS_W_PEAK_PROFILE,
            "w_peak_center": LOSS_W_PEAK_CENTER,
            "w_edge_l1": LOSS_W_EDGE_L1,
            "w_edge_d1": LOSS_W_EDGE_D1,
            "edge_points": LOSS_EDGE_POINTS,
            "peak_window_radius": PEAK_WINDOW_RADIUS,
            "peak_quantile": PEAK_QUANTILE,
            "peak_softmax_temp": PEAK_SOFTMAX_TEMP,
            "mask_leading_points": MASK_LEADING_POINTS,
            "w_smooth_consistency": LOSS_W_SMOOTH_CONSISTENCY,
            "smooth_kernel_size": SMOOTH_KERNEL_SIZE,
            "w_fingerprint_l1": LOSS_W_FINGERPRINT_L1,
            "w_fingerprint_mse": LOSS_W_FINGERPRINT_MSE,
            "w_fingerprint_d1": LOSS_W_FINGERPRINT_D1,
            "fingerprint_start": FINGERPRINT_START,
            "fingerprint_end": FINGERPRINT_END,
            "w_curvature_mse": LOSS_W_CURVATURE_MSE,
            "curvature_scale": CURVATURE_SCALE,
            "w_valley_under": LOSS_W_VALLEY_UNDER,
            "valley_quantile": VALLEY_QUANTILE,
            "w_valley_center": LOSS_W_VALLEY_CENTER,
            "w_peak_align": LOSS_W_PEAK_ALIGN,
            "peak_align_window": PEAK_ALIGN_WINDOW,
            "peak_align_weight": PEAK_ALIGN_WEIGHT,
            "peak_point_weight": PEAK_POINT_WEIGHT,
            "slope_point_weight": SLOPE_POINT_WEIGHT,
            "fingerprint_point_weight": FINGERPRINT_POINT_WEIGHT,
            "peak_focus_start": PEAK_FOCUS_START,
            "peak_focus_end": PEAK_FOCUS_END,
            "w_peak_dominance": LOSS_W_PEAK_DOMINANCE,
            "peak_dominance_margin_scale": PEAK_DOMINANCE_MARGIN_SCALE,
            "w_extrema_mse": LOSS_W_EXTREMA_MSE,
            "extrema_window": EXTREMA_WINDOW,
            "extrema_weight": EXTREMA_WEIGHT,
            "w_sobolev_l1": LOSS_W_SOBOLEV_L1,
            "w_baseline_drift": LOSS_W_BASELINE_DRIFT,
            "baseline_drift_kernel": BASELINE_DRIFT_KERNEL,
            "hard_sample_file": HARD_SAMPLE_FILE,
            "hard_sample_multiplier": HARD_SAMPLE_MULTIPLIER,
        },
        "model_path": MODEL_PATH,
        "promote_model_path": PROMOTE_MODEL_PATH,
        "promotion_threshold": MIN_PROMOTION_QUALITY,
        "pairs_dir": PAIRS_DIR,
        **final_metrics,
        "history": history,
        "best_cases": best_cases,
        "worst_cases": worst_cases,
        "best_flat_cases": best_flat_cases,
    }

    print("\nFinal Validation Results:")
    print(f"  Overall Quality : {results['overall_quality']:.2f}%")
    print(f"  Selection Score : {results['selection_score']:.3f}")
    print(f"  Review Score    : {results['review_score']:.3f}")
    print(f"  Mean PSNR       : {results['mean_psnr']:.2f} dB")
    print(f"  Mean SSIM       : {results['mean_ssim']:.4f}")
    print(f"  Mean Corr       : {results['mean_corr']:.4f}")
    print(f"  Noise Floor Std : {safe_float(results['mean_noise_floor_std']):.4f}")
    print(f"  FP RMSE         : {safe_float(results['mean_rmse_fingerprint']):.4f}")
    print(f"  Peak Amp MAE    : {safe_float(results['mean_peak_amp_mae']):.4f}")
    print(f"  Peak Shift Mean : {safe_float(results['mean_peak_shift_mean_abs']):.3f}")

    promoted = False
    prior_quality = current_promoted_quality()
    improvement_floor = None
    if REQUIRE_QUALITY_IMPROVEMENT and prior_quality is not None:
        improvement_floor = prior_quality + MIN_IMPROVEMENT_DELTA
    quality_passed = results["overall_quality"] >= MIN_PROMOTION_QUALITY
    improvement_passed = improvement_floor is None or results["overall_quality"] >= improvement_floor
    fp_rmse_passed = safe_float(results.get("mean_rmse_fingerprint")) <= MAX_PROMOTION_FP_RMSE
    peak_shift_passed = safe_float(results.get("mean_peak_shift_mean_abs")) <= MAX_PROMOTION_PEAK_SHIFT
    peak_amp_passed = safe_float(results.get("mean_peak_amp_mae")) <= MAX_PROMOTION_PEAK_AMP_MAE

    results["prior_promoted_quality"] = prior_quality
    results["promotion_quality_passed"] = quality_passed
    results["promotion_improvement_passed"] = improvement_passed
    results["promotion_fp_rmse_passed"] = fp_rmse_passed
    results["promotion_peak_shift_passed"] = peak_shift_passed
    results["promotion_peak_amp_passed"] = peak_amp_passed

    if AUTO_PROMOTE and quality_passed and improvement_passed and fp_rmse_passed and peak_shift_passed and peak_amp_passed:
        os.makedirs(os.path.dirname(PROMOTE_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(PROMOTE_RESULTS_PATH), exist_ok=True)
        if os.path.abspath(MODEL_PATH) != os.path.abspath(PROMOTE_MODEL_PATH):
            shutil.copy2(MODEL_PATH, PROMOTE_MODEL_PATH)
        promoted = True
        print(
            f"Promotion gate passed: overall_quality={results['overall_quality']:.2f}% "
            f">= {MIN_PROMOTION_QUALITY:.2f}%. Promoted to {PROMOTE_MODEL_PATH}"
        )
    elif AUTO_PROMOTE:
        reason = []
        if not quality_passed:
            reason.append(f"below {MIN_PROMOTION_QUALITY:.2f}% minimum")
        if not improvement_passed and improvement_floor is not None:
            reason.append(f"below current promoted quality floor {improvement_floor:.2f}%")
        if not fp_rmse_passed:
            reason.append(f"fingerprint RMSE above {MAX_PROMOTION_FP_RMSE:.4f}")
        if not peak_shift_passed:
            reason.append(f"peak shift above {MAX_PROMOTION_PEAK_SHIFT:.4f}")
        if not peak_amp_passed:
            reason.append(f"peak amp MAE above {MAX_PROMOTION_PEAK_AMP_MAE:.4f}")
        print(
            f"Promotion gate failed: overall_quality={results['overall_quality']:.2f}% "
            f"({'; '.join(reason)}). Existing promoted model left unchanged at {PROMOTE_MODEL_PATH}"
        )
    else:
        print("Auto-promotion disabled; candidate model left unpromoted.")

    results["promoted"] = promoted

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_PATH}")
    if promoted and os.path.abspath(RESULTS_PATH) != os.path.abspath(PROMOTE_RESULTS_PATH):
        shutil.copy2(RESULTS_PATH, PROMOTE_RESULTS_PATH)
        print(f"Promoted metrics saved to {PROMOTE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
