from __future__ import annotations

# ============================================================
# 环境配置（必须在第一行）
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"           # OpenMP 冲突修复
os.environ["TORCHDYNAMO_DISABLE"] = "1"               # 禁用 Dynamo
os.environ["TORCH_COMPILE_DISABLE"] = "1"             # 禁用编译
os.environ["OMP_NUM_THREADS"] = "8"                   # 线程数限制
os.environ["MKL_NUM_THREADS"] = "8"
# ============================================================

from pathlib import Path
import argparse

import dataclasses
import math
import random
import sys
import time
from pathlib import Path

import math
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import seaborn as sns
from matplotlib.colors import LogNorm

import warnings
warnings.filterwarnings('ignore')

# 统一 DDM 字段优先级，避免统计与训练读取来源不一致。
DDM_ARRAY_KEYS: Tuple[str, ...] = ("ddm_tensor", "ddm", "M", "ddm_4ch")

# 设置中文字体和全局绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)

# 模块 1：运行环境与兼容性注入（Numpy Pickle Compatibility Shim）
def _install_numpy_core_compat() -> None:
    try:
        import numpy.core as npcore  # noqa
        import numpy as np  # noqa
        import numpy.core._multiarray_umath as mum  # noqa
    except Exception:
        return
    mod = types.ModuleType("numpy._core")
    mod.__dict__.update(npcore.__dict__)
    sys.modules.setdefault("numpy._core", mod)
    sys.modules.setdefault("numpy._core._multiarray_umath", mum)
    
_install_numpy_core_compat()

# 可选的重依赖项，仅在数据集构建模式下需要
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import xarray as xr  # type: ignore
else:
    try:
        import xarray as xr  # type: ignore
    except Exception:
        xr = None

# 模块 2：实验可复现性控制（Determinism Control Plane）
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

# 模块 3：GPU 性能优化（在 set_seed 函数后添加）
def setup_gpu_optimizations(*, deterministic: bool = True) -> None:
    """
    针对 RTX 5090 / RTX 40/50 系列的性能优化
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，跳过 GPU 优化")
        return
    
    if deterministic:
        # 确定性模式：优先复现，不开启动态算法选择。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        # 性能模式：允许 TF32 与 benchmark 自动调优。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 3. 设置 CUDA 内存分配策略（减少碎片）
    if hasattr(torch.cuda, "set_per_process_memory_fraction"):
        try:
            torch.cuda.set_per_process_memory_fraction(0.95, device=0)  # 使用 95% 显存
        except (RuntimeError, ValueError) as e:
            print(f"⚠️ 显存占比设置失败，继续默认策略: {e}")
    
    # 4. 打印 GPU 信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU 配置已应用: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"   - deterministic: {'ON' if deterministic else 'OFF'}")
    print(f"   - TF32: {'ON' if torch.backends.cuda.matmul.allow_tf32 else 'OFF'}")
    print(f"   - cuDNN Benchmark: {'ON' if torch.backends.cudnn.benchmark else 'OFF'}")


# 模块 4：评估指标体系（Metric Kernel: RMSE/Bias/Corr + Interval）
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if y_true.size == 0:
        return {"rmse": float("nan"), "bias": float("nan"), "corr": float("nan")}
    
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    bias = float(np.mean(y_pred - y_true))
    
    y_true_c = y_true - y_true.mean()
    y_pred_c = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(y_true_c**2) * np.sum(y_pred_c**2))
    corr = float(np.sum(y_true_c * y_pred_c) / denom) if denom > 0 else float("nan")
    
    return {"rmse": rmse, "bias": bias, "corr": corr}

def wind_speed_intervals() -> List[Tuple[float, float]]:
    return [(2.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, float("inf"))]

def compute_interval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    out: List[Dict[str, float]] = []
    
    for lo, hi in wind_speed_intervals():
        m = (y_true >= lo) & (y_true < hi) & np.isfinite(y_true) & np.isfinite(y_pred)
        met = compute_metrics(y_true[m], y_pred[m])
        met.update({"lo": lo, "hi": hi, "n": int(m.sum())})
        out.append(met)
    return out


def aggregate_window_predictions_to_observations(
    windows: Sequence[np.ndarray],
    window_preds: np.ndarray,
    df: pd.DataFrame,
    *,
    target_col: str = "WS",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将“窗口级预测”聚合为“原始观测点级预测”：
    - 每个窗口预测值复制到该窗口内所有观测点索引
    - 对同一观测点在不同窗口中的预测取平均
    """
    preds = np.asarray(window_preds, dtype=np.float64)
    if len(windows) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    if preds.ndim == 1:
        # Backward-compatible path: one scalar prediction per window.
        if len(preds) != len(windows):
            raise ValueError(
                f"窗口预测数量与窗口数量不一致: preds={len(preds)}, windows={len(windows)}"
            )
        lengths = np.asarray([len(w) for w in windows], dtype=np.int64)
        flat_indices = np.concatenate([np.asarray(w, dtype=np.int64).reshape(-1) for w in windows], axis=0)
        flat_preds = np.repeat(preds, lengths)
    elif preds.ndim == 2:
        # Paper-aligned path: one prediction for each time step in each window.
        if preds.shape[0] != len(windows):
            raise ValueError(
                f"窗口预测数量与窗口数量不一致: preds.shape[0]={preds.shape[0]}, windows={len(windows)}"
            )
        flat_indices_parts: List[np.ndarray] = []
        flat_preds_parts: List[np.ndarray] = []
        for wi, w in enumerate(windows):
            w_idx = np.asarray(w, dtype=np.int64).reshape(-1)
            if preds.shape[1] < len(w_idx):
                raise ValueError(
                    f"第 {wi} 个窗口长度为 {len(w_idx)}，但预测序列长度仅为 {preds.shape[1]}"
                )
            flat_indices_parts.append(w_idx)
            flat_preds_parts.append(preds[wi, : len(w_idx)].reshape(-1))
        flat_indices = np.concatenate(flat_indices_parts, axis=0)
        flat_preds = np.concatenate(flat_preds_parts, axis=0)
    else:
        raise ValueError(f"window_preds 仅支持 1D 或 2D，当前 ndim={preds.ndim}")

    unique_indices, inv = np.unique(flat_indices, return_inverse=True)
    vote_counts = np.bincount(inv).astype(np.int64)
    pred_sums = np.bincount(inv, weights=flat_preds)
    pred_means = pred_sums / np.maximum(vote_counts, 1)
    y_true = df.loc[unique_indices, target_col].to_numpy(dtype=np.float64)
    return unique_indices, y_true, pred_means.astype(np.float64), vote_counts


def assert_no_window_striping_artifacts(
    raw_window_preds: np.ndarray,
    obs_indices: np.ndarray,
    y_true_obs: np.ndarray,
    y_pred_obs: np.ndarray,
    vote_counts: np.ndarray,
    *,
    time_steps: int,
    set_name: str,
) -> None:
    """
    在绘图前做 3 个断言，快速拦截最常见的“窗口重复条纹”来源。
    """
    raw = np.asarray(raw_window_preds)
    obs_indices = np.asarray(obs_indices, dtype=np.int64).reshape(-1)
    y_true_obs = np.asarray(y_true_obs, dtype=np.float64).reshape(-1)
    y_pred_obs = np.asarray(y_pred_obs, dtype=np.float64).reshape(-1)
    vote_counts = np.asarray(vote_counts, dtype=np.int64).reshape(-1)

    # 1) 聚合后的观测索引必须一一对应（不能仍是窗口级重复点）。
    if (
        len(obs_indices) != len(y_true_obs)
        or len(y_true_obs) != len(y_pred_obs)
        or len(y_pred_obs) != len(vote_counts)
        or len(np.unique(obs_indices)) != len(obs_indices)
    ):
        raise RuntimeError(
            f"{set_name}: 聚合后索引/预测长度不一致或存在重复观测索引，"
            "疑似仍在使用窗口级样本绘图。"
        )

    # 2) 投票数应落在合法范围 [1, time_steps]。
    if len(vote_counts) == 0 or int(vote_counts.min()) < 1 or int(vote_counts.max()) > int(time_steps):
        raise RuntimeError(
            f"{set_name}: vote_count 范围异常 [{int(vote_counts.min()) if len(vote_counts) else 'NA'}, "
            f"{int(vote_counts.max()) if len(vote_counts) else 'NA'}]，"
            f"期望位于 [1, {int(time_steps)}]。"
        )

    # 3) 聚合后点数应明显小于窗口展开点数，否则通常是没正确聚合。
    raw_flat_n = int(raw.size) if raw.ndim >= 1 else int(len(raw))
    if raw_flat_n > 0:
        compression = len(y_true_obs) / raw_flat_n
        if compression >= 0.75:
            raise RuntimeError(
                f"{set_name}: 聚合压缩比异常 (obs/raw={compression:.3f})，"
                "窗口重复点可能未正确折叠。"
            )


# 模块 5：DDM 张量抽取与外部存储适配（DDM Access Layer）
class DDMExternalStore:
    """
    外部 DDM 存储类。
    用于 "元数据 pkl + 独立 DDM 大数组文件" 的工作流，避免 Pandas DataFrame 占用过多内存。
    深度学习中，如果数据集太大无法一次性读入内存，通常使用这种内存映射 (mmap) 技术。
    支持 .npy 和 .npz 格式。
    """
    def __init__(self, path: Path, *, mmap: bool = True):
        self.path = Path(path)
        if self.path.suffix.lower() == ".npy":
            # 使用内存映射模式 (mmap_mode)，避免一次性加载到 RAM
            self.arr = np.load(self.path, mmap_mode="r" if mmap else None)
        elif self.path.suffix.lower() == ".npz":
            z = np.load(self.path)
            if "ddm" not in z:
                raise KeyError(f"{path} .npz 文件缺少 'ddm' 数组")
            self.arr = z["ddm"]
        else:
            raise ValueError(f"不支持的 DDM 存储格式: {path}")
        if self.arr.ndim != 4 or tuple(self.arr.shape[1:]) != (4, 17, 11):
            raise ValueError(f"DDM 存储形状必须为 (N, 4, 17, 11); 当前为 {self.arr.shape}")

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def get(self, idx: int) -> np.ndarray:
        return np.asarray(self.arr[idx], dtype=np.float32)

class DDMIndexResolver:
    """
    解析器：决定如何将 DataFrame 的行索引映射到 DDM 存储的索引。
    """
    def __init__(self, df: pd.DataFrame, store: DDMExternalStore):
        self.use_sample = False
        self.store_len = len(store)

        if self.store_len == len(df):
            self.use_sample = False
            return

        # 如果长度不一致，检查是否存在 'sample' 列用于索引
        if "sample" in df.columns:
            smax = int(np.nanmax(df["sample"].to_numpy(dtype=np.int64)))
            if self.store_len >= (smax + 1):
                self.use_sample = True
                return

        raise ValueError(
            f"无法对齐 DDM 存储长度 N={self.store_len} 与 DataFrame 行数 len(df)={len(df)}. "
            f"请提供 N=len(df) 的存储，或者包含 'sample' 列且满足 max(sample)+1 <= N."
        )

    def index(self, df_row_idx: int, df: pd.DataFrame) -> int:
        if not self.use_sample:
            return int(df_row_idx)
        return int(df.loc[df_row_idx, "sample"])


# 模块 6：时空序列窗口构建（Spatiotemporal Windowing Kernel）
def make_windows_from_observations(
    df: pd.DataFrame,
    *,
    group_col: str = "final_seq_group",
    time_col: str = "ddm_timestamp_utc",
    lat_col: str = "sp_lat",
    lon_col: str = "sp_lon",
    window: int = 10,
    max_dt_sec: float = 3600.0,
    max_dlat: float = 0.5,
    max_dlon: float = 0.5,
) -> List[np.ndarray]:
    """
    生成符合论文约束的长度为 'window' (默认10) 的滑动窗口索引列表。
    约束条件：同一组、时间连续、空间跨度不过大。
    返回: List[int64 array]，每个数组包含一个窗口内的行索引。
    这是时空序列预测的关键步骤：将离散的观测点组合成时间序列样本。
    """
    if group_col not in df.columns:
        raise KeyError(f"数据集缺少分组列 '{group_col}'，无法构建 10 步序列。")
    if time_col not in df.columns:
        raise KeyError(f"数据集缺少排序列 '{time_col}'，无法构建 10 步序列。")

    windows: List[np.ndarray] = []

    # 先按组和时间排序，确保滑动窗口的逻辑正确
    df_sorted = df.sort_values([group_col, time_col]).reset_index()
    # 对每个连续组进行处理
    for _, g in df_sorted.groupby(group_col, sort=False):
        if len(g) < window:
            continue
        times = pd.to_datetime(g[time_col]).values.astype("datetime64[s]").astype(np.int64)
        lat = g[lat_col].values.astype(np.float64) if lat_col in g.columns else None
        lon = g[lon_col].values.astype(np.float64) if lon_col in g.columns else None
        orig_idx = g["index"].values.astype(np.int64)

        # 滑动窗口：从 s 开始，长度为 window
        for s in range(0, len(g) - window + 1):
            e = s + window
            # 检查时间跨度 (max_dt_sec)
            if (times[e - 1] - times[s]) > max_dt_sec:
                continue
            # 检查空间跨度 (max_dlat, max_dlon)
            if lat is not None and (np.max(lat[s:e]) - np.min(lat[s:e])) > max_dlat:
                continue
            if lon is not None and (np.max(lon[s:e]) - np.min(lon[s:e])) > max_dlon:
                continue
            windows.append(orig_idx[s:e].copy())

    return windows


# 模块 7：数据集划分策略（Time-based Split + Fallback）
def split_by_paper_time(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    按绝对时间顺序划分数据集 (8:1:1)
    - 前 80% 时间的数据 → 训练集
    - 中间 10% 时间的数据 → 验证集  
    - 后 10% 时间的数据 → 测试集
    
    注意：这种方式可能会切断某些序列组，需要后续在窗口构建时处理
    """
    if "ddm_timestamp_utc" not in df.columns:
        raise KeyError("缺少列 'ddm_timestamp_utc'")
    
    print("使用【绝对时间】进行 8:1:1 划分...")
    
    # 按时间排序
    t = pd.to_datetime(df["ddm_timestamp_utc"])
    sorted_idx = t.sort_values().index
    
    n = len(sorted_idx)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    
    train_idx = sorted_idx[:n_train]
    val_idx = sorted_idx[n_train:n_train + n_val]
    test_idx = sorted_idx[n_train + n_val:]
    
    # 打印时间范围
    print(f"  训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}, 测试样本: {len(test_idx)}")
    print(f"  训练集时间范围: {t.loc[train_idx].min()} ~ {t.loc[train_idx].max()}")
    print(f"  验证集时间范围: {t.loc[val_idx].min()} ~ {t.loc[val_idx].max()}")
    print(f"  测试集时间范围: {t.loc[test_idx].min()} ~ {t.loc[test_idx].max()}")
    
    # 验证时间不重叠
    train_max = t.loc[train_idx].max()
    val_min = t.loc[val_idx].min()
    val_max = t.loc[val_idx].max()
    test_min = t.loc[test_idx].min()
    
    print(f"  ✅ 训练集结束: {train_max}")
    print(f"  ✅ 验证集开始: {val_min}")
    print(f"  ✅ 验证集结束: {val_max}")
    print(f"  ✅ 测试集开始: {test_min}")
    
    return train_idx, val_idx, test_idx


# 模块 8：归一化统计计算与持久化（Normalization Statistics Service）
@dataclass
class NormalizationStats:
    """
    存储 Z-Score 归一化所需的均值和标准差。
    Z-Score 公式: x_norm = (x - mean) / std
    """
    ddm_mean: np.ndarray  # (4,)  DDM 四个通道的均值
    ddm_std: np.ndarray   # (4,)  DDM 四个通道的标准差
    aux_mean: np.ndarray  # (F,)  辅助特征的均值
    aux_std: np.ndarray   # (F,)  辅助特征的标准差

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ddm_mean=self.ddm_mean, ddm_std=self.ddm_std, aux_mean=self.aux_mean, aux_std=self.aux_std)

    @staticmethod
    def load(path: Path) -> "NormalizationStats":
        z = np.load(path)
        return NormalizationStats(ddm_mean=z["ddm_mean"], ddm_std=z["ddm_std"], aux_mean=z["aux_mean"], aux_std=z["aux_std"])

def compute_normalization_stats(
    df: pd.DataFrame,
    windows: Sequence[np.ndarray],
    aux_cols: Sequence[str],
    ddm_store: Optional[DDMExternalStore] = None,
    ddm_index: Optional[DDMIndexResolver] = None,
    use_log_transform: bool = True,
    max_samples: int = 100000,  # 增大采样数
) -> NormalizationStats:
    # 设计目标：
    # 1) 仅基于训练窗口涉及的观测点计算统计量，避免数据泄漏。
    # 2) 同时统计 DDM 提取成功率/来源，防止静默失败导致统计失真。
    # 3) 失败率达到硬阈值时中止，避免不可信统计量进入训练流程。
    """
    计算归一化统计量
    修复：使用随机采样而非分层采样，保持原始分布
    """
    
    print("  提取唯一样本...")
    # 去重后的观测索引集合。
    # 同一观测点可能出现在多个滑窗中，这里只统计一次以减少冗余。
    all_idxs = sorted(set(int(x) for w in windows for x in w))
    
    # ✅ 修复：使用随机采样，保持原始分布
    if len(all_idxs) > max_samples:
        print(f"  样本数 {len(all_idxs)} 超过上限 {max_samples}，随机采样...")
        rng = np.random.default_rng(42)
        all_idxs = rng.choice(all_idxs, size=max_samples, replace=False).tolist()
    
    n_samples = len(all_idxs)
    print(f"  使用 {n_samples} 个样本计算统计量")
    
    # 辅助特征统计
    # 辅助特征统计采用“finite 掩码 + NaN 填 0”的方式，保证数值稳定。
    # 最终得到每个辅助特征维度的均值和标准差（用于 Z-Score 归一化）。
    aux_all = df.loc[all_idxs, aux_cols].to_numpy(dtype=np.float64)
    m = np.isfinite(aux_all)
    aux_clean = np.nan_to_num(aux_all, nan=0.0)
    aux_mean = (aux_clean * m).sum(axis=0) / np.maximum(m.sum(axis=0), 1)
    aux_var = ((aux_clean ** 2) * m).sum(axis=0) / np.maximum(m.sum(axis=0), 1) - aux_mean ** 2
    aux_std = np.sqrt(np.maximum(aux_var, 1e-12))
    
    # DDM 统计（对数变换后）
    print(f"  计算 DDM 统计量 (log_transform={use_log_transform})...")
    ddm_values = [[] for _ in range(4)]
    ddm_ok_count = 0
    ddm_fail_count = 0
    ddm_source_df = 0
    ddm_source_store = 0
    # 计数器说明：
    # ddm_ok_count: 成功参与统计的样本数
    # ddm_fail_count: 提取失败或形状不合法而跳过的样本数
    # ddm_source_df/store: 成功样本来自 DataFrame / 外部存储的数量
    
    report_interval = max(1, n_samples // 10)
    
    for i, ridx in enumerate(all_idxs):
        if (i + 1) % report_interval == 0:
            print(f"    {100*(i+1)//n_samples}%")
        
        row = df.loc[ridx]
        arr = None
        source = "none"
        # 提取优先级：DataFrame > 外部存储。
        # 这样可兼容“元数据在 pkl、DDM 大数组在外部文件”的双存储模式。
        
        # 从 DataFrame 中提取
        for key in DDM_ARRAY_KEYS:
            if key in row.index and isinstance(row[key], np.ndarray):
                arr = row[key].astype(np.float32)
                source = "df"
                break
        
        # 从外部存储读取
        if arr is None and ddm_store is not None and ddm_index is not None:
            try:
                arr = ddm_store.get(ddm_index.index(ridx, df))
                if arr is not None:
                    source = "store"
            except (IndexError, KeyError, ValueError) as e:
                print(f"      ⚠️ 警告：样本 {ridx} 的外部 DDM 读取失败: {e}")
        
        if arr is None:
            ddm_fail_count += 1
            continue
        
        # 先做一层严格校验：异常形状直接计入失败并跳过，避免静默进入全 0 风险。
        # 形状规范化策略：
        # - 期望最终形状恒为 (4, 17, 11)
        # - 若输入为 (4, 11, 17) 则做转置
        # - 其它形状直接记为失败并跳过
        if arr.ndim != 3 or arr.shape[0] != 4:
            print(f"      ⚠️ 警告：样本 {ridx} 的 DDM 维度异常，shape={getattr(arr, 'shape', None)}，已跳过")
            ddm_fail_count += 1
            continue
        if arr.shape[1:] == (11, 17):
            arr = np.transpose(arr, (0, 2, 1))
        elif arr.shape[1:] != (17, 11):
            print(f"      ⚠️ 警告：样本 {ridx} 的 DDM 形状异常，shape={arr.shape}，已跳过")
            ddm_fail_count += 1
            continue
        
        ddm_ok_count += 1
        if source == "df":
            ddm_source_df += 1
        elif source == "store":
            ddm_source_store += 1
        
        for ch in range(4):
            ch_data = np.abs(arr[ch]).flatten()
            
            if use_log_transform:
                # ✅ 与 Dataset 中的变换保持一致
                if ch == 0:
                    ch_data = np.maximum(ch_data, 1e-25)
                else:
                    ch_data = np.maximum(ch_data, 1.0)
                ch_data = np.log10(ch_data)
            
            valid = ch_data[np.isfinite(ch_data)]
            ddm_values[ch].extend(valid.tolist())
    
    print("  DDM 统计阶段加载报告:")
    print(f"    ✅ 成功: {ddm_ok_count}/{n_samples} ({100*ddm_ok_count/max(n_samples,1):.1f}%)")
    print(f"       - 来自 DataFrame: {ddm_source_df}")
    print(f"       - 来自外部存储: {ddm_source_store}")
    print(f"    ❌ 失败: {ddm_fail_count}/{n_samples} ({100*ddm_fail_count/max(n_samples,1):.1f}%)")
    # 失败率策略：
    # >=50%: 统计量高风险，直接中止；
    # (5%, 50%): 允许继续但明确告警；
    # (0, 5%]: 仅提示少量缺失。
    fail_ratio = ddm_fail_count / max(n_samples, 1)
    if fail_ratio >= 0.5:
        raise RuntimeError(
            f"❌ DDM 统计失败率过高: {ddm_fail_count}/{n_samples} ({fail_ratio*100:.1f}%)\n"
            f"超过 50% 的样本无法提取有效 DDM，归一化统计不可信，已中止。\n"
            f"可能原因:\n"
            f"  1. DataFrame 中没有 DDM 数组列 ({DDM_ARRAY_KEYS})\n"
            f"  2. 外部 DDM 存储未正确配置 (ddm_store={ddm_store is not None})\n"
            f"  3. DDM 数组形状不匹配 (期望 (4, 17, 11) 或 (4, 11, 17))"
        )
    elif fail_ratio > 0.05:
        print(
            f"  ⚠️ 警告: DDM 统计阶段有 {ddm_fail_count} 个样本 ({fail_ratio*100:.1f}%) 提取失败，"
            f"可能影响归一化稳定性。"
        )
    elif ddm_fail_count > 0:
        print(f"  ℹ️ 少量样本 ({ddm_fail_count}) DDM 缺失，对统计影响较小。")
    
    if ddm_ok_count == 0:
        raise RuntimeError("❌ 没有任何有效 DDM 样本可用于归一化统计。")
    
    # 每个通道在全体像素上的统计量（flatten 后聚合）。
    # ddm_std 设置最小下界，避免后续除零。
    ddm_mean = np.array([np.mean(v) for v in ddm_values], dtype=np.float32)
    ddm_std = np.array([np.std(v) for v in ddm_values], dtype=np.float32)
    ddm_std = np.maximum(ddm_std, 1e-6)
    
    print(f"  DDM mean (log): {ddm_mean}")
    print(f"  DDM std (log):  {ddm_std}")
    
    return NormalizationStats(
        ddm_mean=ddm_mean,
        ddm_std=ddm_std,
        aux_mean=aux_mean.astype(np.float32),
        aux_std=aux_std.astype(np.float32),
    )


# 模块 9：PyTorch 数据集类（STGWindowDataset）
class STGWindowDataset(Dataset):
    """修复版数据集：正确实现对数变换 + 外部存储回退"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        windows: Sequence[np.ndarray],
        aux_cols: Sequence[str],
        stats: NormalizationStats,
        ddm_store=None,
        ddm_index=None,
        *,
        target_col: str = "WS",
        preload: bool = True,
        use_log_transform: bool = True,
    ) -> None:
        self.windows = [w.astype(np.int64) for w in windows]
        self.stats = stats
        self.use_log_transform = use_log_transform
        # 预加载模式说明：
        # 训练/验证阶段统一从缓存读取 ddm/aux/y/sp_idx，
        # 避免在 __getitem__ 中重复访问 DataFrame 造成性能抖动。
        # 🔴 修复1：保存外部存储引用，供 _extract_ddm_raw 使用
        self.ddm_store = ddm_store
        self.ddm_index = ddm_index
        self.df = df  # 保存 DataFrame 引用（外部存储回退需要 ddm_index.index(ridx, df)）
        
        if preload:
            print(f"  预加载数据到内存 (log_transform={use_log_transform})...")
            
            # 收集唯一索引
            # idx_to_pos 用于把“原始观测索引”映射到“缓存数组下标”。
            all_idxs = sorted(set(int(x) for w in windows for x in w))
            n_samples = len(all_idxs)
            idx_to_pos = {idx: i for i, idx in enumerate(all_idxs)}
            self.idx_to_pos = idx_to_pos
            
            # 辅助特征
            print(f"    加载辅助特征 ({n_samples} 样本)...")
            aux_all = df.loc[all_idxs, aux_cols].to_numpy(dtype=np.float32)
            aux_all = (aux_all - stats.aux_mean[None, :]) / stats.aux_std[None, :]
            self.aux_cache = np.nan_to_num(aux_all, nan=0.0)
            
            # 标签
            print(f"    加载标签...")
            y_all = df.loc[all_idxs, target_col].to_numpy(dtype=np.float32)
            self.y_cache = np.nan_to_num(y_all, nan=0.0)
            
            # 🔴 新增：加载SP点索引（强制要求）
            print(f"    加载SP点索引...")
            if "sp_index" not in df.columns:
                raise KeyError(
                    "❌ 数据集缺少 'sp_index' 列！\n"
                    "请确保在调用 build_aux_columns() 时已正确生成该列。\n"
                    "该列用于指定每个DDM的真实镜面反射点(SP)位置。"
                )
            
            sp_idx_all = df.loc[all_idxs, "sp_index"].to_numpy(dtype=np.int64)
            
            # 验证SP索引有效性
            if np.any((sp_idx_all < 0) | (sp_idx_all > 186)):
                raise ValueError(
                    f"❌ SP点索引包含无效值！\n"
                    f"索引范围应为 0-186, 但发现: {sp_idx_all.min()}-{sp_idx_all.max()}\n"
                    f"请检查数据集中的 sp_delay_bin 和 sp_doppler_bin 值。"
                )
            
            self.sp_idx_cache = sp_idx_all
            print(f"      ✅ SP索引范围: {sp_idx_all.min()}-{sp_idx_all.max()} (共{len(np.unique(sp_idx_all))}个不同位置)")
            
            # DDM（含对数变换）
            print(f"    加载 DDM 数据 ({n_samples} 样本)...")
            # 先分配全 0 缓存，再对成功样本逐个覆盖写入。
            # 因此必须监控失败率，否则大量失败会残留全 0 输入。
            ddm_cache = np.zeros((n_samples, 4, 17, 11), dtype=np.float32)
            
            # 🔴 修复2：增加失败计数器
            ddm_ok_count = 0
            ddm_fail_count = 0
            ddm_source_df = 0      # 从 DataFrame 提取成功的计数
            ddm_source_store = 0   # 从外部存储提取成功的计数
            
            report_interval = max(1, n_samples // 10)
            for i, ridx in enumerate(all_idxs):
                if (i + 1) % report_interval == 0:
                    print(f"      进度: {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")
                
                # 🔴 修复3：使用带回退的提取方法
                # source 记录该样本来源（df/store/none），用于后续健康度统计。
                arr, source = self._extract_ddm_with_fallback(ridx)
                
                if arr is not None:
                    # ✅ 步骤1：先对数变换，再归一化
                    if self.use_log_transform:
                        arr = self._apply_log_transform(arr)
                    
                    # ✅ 步骤2：归一化
                    arr = (arr - stats.ddm_mean[:, None, None]) / stats.ddm_std[:, None, None]
                    ddm_cache[i] = arr
                    ddm_ok_count += 1
                    if source == "df":
                        ddm_source_df += 1
                    elif source == "store":
                        ddm_source_store += 1
                else:
                    ddm_fail_count += 1
            
            self.ddm_cache = ddm_cache
            
            # 🔴 修复4：详细的加载统计报告
            # 该报告用于快速判断是否存在“DDM 大面积缺失 -> 输入退化 -> 输出塌缩”风险。
            print(f"  DDM 加载统计:")
            print(f"    ✅ 成功: {ddm_ok_count}/{n_samples} ({100*ddm_ok_count/max(n_samples,1):.1f}%)")
            print(f"       - 从 DataFrame 提取: {ddm_source_df}")
            print(f"       - 从外部存储提取: {ddm_source_store}")
            print(f"    ❌ 失败(全0填充): {ddm_fail_count}/{n_samples} ({100*ddm_fail_count/max(n_samples,1):.1f}%)")
            
            # 🔴 修复5：根据失败率抛出错误或警告
            fail_ratio = ddm_fail_count / max(n_samples, 1)
            if fail_ratio > 0.5:
                raise RuntimeError(
                    f"❌ DDM 加载失败率过高: {ddm_fail_count}/{n_samples} ({fail_ratio*100:.1f}%)\n"
                    f"超过 50% 的样本没有有效 DDM 数据，模型将无法正常训练。\n"
                    f"可能原因:\n"
                    f"  1. DataFrame 中没有 DDM 数组列 ({DDM_ARRAY_KEYS})\n"
                    f"  2. 外部 DDM 存储未正确配置 (ddm_store={ddm_store is not None})\n"
                    f"  3. DDM 数组形状不匹配 (期望 (4, 17, 11) 或 (4, 11, 17))"
                )
            elif fail_ratio > 0.05:
                print(
                    f"  ⚠️ 警告: {ddm_fail_count} 个样本 ({fail_ratio*100:.1f}%) DDM 为全 0！\n"
                    f"     这些样本的空间特征将退化为常数，可能影响模型性能。"
                )
            elif ddm_fail_count > 0:
                print(f"  ℹ️ 少量样本 ({ddm_fail_count}) DDM 缺失，影响可忽略。")
            
            print(f"  ✅ 预加载完成，内存占用: {ddm_cache.nbytes / 1e9:.2f} GB")
            self.preloaded = True
    
    def _extract_ddm_with_fallback(self, ridx: int) -> Tuple[Optional[np.ndarray], str]:
        """
        🔴 修复核心：带外部存储回退的 DDM 提取方法
        
        提取优先级：
        1. 先从 DataFrame 行中查找 DDM 数组列
        2. 如果 DataFrame 中没有，回退到外部 DDM 存储 (ddm_store)
        
        Returns:
            (arr, source): arr 为 (4, 17, 11) 的 float32 数组或 None
                           source 为 "df" / "store" / "none"
        """
        # 该函数是 DDM 抽取的统一入口：
        # - 封装双路径读取（DataFrame + 外部存储）
        # - 统一返回来源标签，便于日志和统计
        # - 任何异常均降级为失败返回，不在此处中断训练流程
        row = self.df.loc[ridx]
        
        # 优先级 1：从 DataFrame 提取
        arr = self._extract_ddm_raw(row)
        if arr is not None:
            return arr, "df"
        
        # 优先级 2：从外部存储提取
        if self.ddm_store is not None and self.ddm_index is not None:
            try:
                store_idx = self.ddm_index.index(ridx, self.df)
                arr = self.ddm_store.get(store_idx)  # 返回 (4, 17, 11) float32
                if arr is not None and arr.ndim == 3 and arr.shape[0] == 4:
                    if arr.shape[1:] == (17, 11):
                        return arr.astype(np.float32).copy(), "store"
                    elif arr.shape[1:] == (11, 17):
                        return np.transpose(arr, (0, 2, 1)).astype(np.float32).copy(), "store"
                    else:
                        print(f"      ⚠️ 外部存储样本 {ridx} 形状异常: {arr.shape}，已跳过")
            except (IndexError, KeyError, ValueError) as e:
                print(f"      ⚠️ 外部存储读取样本 {ridx} 失败: {e}")
        
        return None, "none"
    
    def _extract_ddm_raw(self, row) -> Optional[np.ndarray]:
        """提取原始 DDM 数据（不做任何变换）—— 仅从 DataFrame 行提取"""
        # 只负责“从行对象提取 + 形状规范化”，不处理外部存储回退。
        # 返回约定：
        # - 成功：float32, shape=(4, 17, 11)
        # - 失败：None
        for key in DDM_ARRAY_KEYS:
            if key in row.index and isinstance(row[key], np.ndarray):
                arr = row[key]
                if arr.ndim == 3 and arr.shape[0] == 4:
                    if arr.shape[1:] == (17, 11):
                        return arr.astype(np.float32).copy()
                    elif arr.shape[1:] == (11, 17):
                        return np.transpose(arr, (0, 2, 1)).astype(np.float32).copy()
        return None
    
    def _apply_log_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        对 DDM 进行对数变换（与 compute_normalization_stats 保持一致！）
        """
        # 数值稳定策略：
        # 1) 先取绝对值，抑制符号噪声；
        # 2) 分通道设置下界，避免 log10(0)；
        # 3) 最后用 nan_to_num 清理异常值，保证缓存无 NaN/Inf。
        result = arr.copy()
        for ch in range(4):
            ch_data = np.abs(result[ch])
            
            # 设置下限避免 log(0)，与统计量计算保持一致
            if ch == 0:  # power analog，值极小 (1e-25 ~ 1e-17)
                ch_data = np.maximum(ch_data, 1e-25)
            else:  # 其他通道
                ch_data = np.maximum(ch_data, 1.0)
            
            result[ch] = np.log10(ch_data)
        
        # 处理可能的 NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=-30.0)
        return result
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, i):
        idxs = self.windows[i]
        # 将窗口里的原始观测索引映射到预加载缓存下标。
        positions = [self.idx_to_pos[int(idx)] for idx in idxs]
        
        # 返回一个时间窗口对应的完整序列：
        # ddm:   (T, 4, 17, 11)
        # aux:   (T, aux_dim)
        # y:     (T,)
        # sp_idx:(T,)
        return {
            "ddm": torch.from_numpy(self.ddm_cache[positions].copy()),
            "aux": torch.from_numpy(self.aux_cache[positions].copy()),
            # Paper-aligned supervision: one target per time step in the 10-step window.
            "y": torch.from_numpy(self.y_cache[positions].copy()).float(),
            "sp_idx": torch.from_numpy(self.sp_idx_cache[positions].copy()),  # 🔴 新增：返回SP索引序列 (T,)
        }

# 模块 10：通用神经网络基元
class MLP(nn.Module):
    """多层感知机 (Multi-Layer Perceptron)，最基础的神经网络结构"""
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float = 0.0, act=nn.ReLU) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 模块 11：图卷积层实现（基于 STG-DNN 论文第 III-B 节）
def knn_cosine(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    使用余弦相似度计算 K 近邻 (KNN)。
    将 DDM 的 187 个像素视为图节点，通过特征空间中的相似度动态构建图结构。
    
    Args:
        x: (B, N, C) - B 个样本，每个样本 N 个节点，每个节点 C 维特征
        k: 每个节点连接的邻居数量
    
    Returns:
        idx: (B, N, k) - 每个节点的 k 个最近邻的索引
    
    修复：使用 FP16 安全的 mask 值，避免 AMP 混合精度训练时溢出
    """
    B, N, C = x.shape
    # L2 归一化后做内积 = 余弦相似度
    x_n = F.normalize(x, p=2, dim=-1, eps=1e-12)
    sim = torch.bmm(x_n, x_n.transpose(1, 2))  # (B, N, N) 相似度矩阵
    # 屏蔽自身（对角线），避免节点选自己为邻居
    # 使用 diagonal in-place 填充，避免每次分配 (N, N) 的 eye 张量
    mask_value = -65000.0 if sim.dtype == torch.float16 else -1e9
    sim.diagonal(dim1=-2, dim2=-1).fill_(mask_value)
    # 选择相似度最高的 k 个邻居
    _, idx = sim.topk(k=k, dim=-1, largest=True, sorted=False)
    return idx

def gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    根据 KNN 索引收集邻居特征。
    用于图卷积中聚合邻居信息。
    
    Args:
        x: (B, N, C) - 节点特征
        idx: (B, N, k) - 每个节点的 k 个邻居索引
    
    Returns:
        (B, N, k, C) - 每个节点的 k 个邻居的特征
    """
    B, N, C = x.shape
    k = idx.shape[-1]
    # 生成 batch 偏移索引，将 (B, N) 展平为 (B*N) 进行索引
    idx_base = torch.arange(B, device=x.device).view(B, 1, 1) * N
    idx_flat = (idx + idx_base).reshape(-1)
    x_flat = x.reshape(B * N, C)
    return x_flat[idx_flat].reshape(B, N, k, C)

# 模块 12：空间图特征提取（Dynamic EdgeConv Graph Encoder）
class EdgeConvMultiHead(nn.Module):
    """
    标准全节点 EdgeConv 图卷积层：
    - 对所有 N 个节点构建 KNN 图
    - 对所有节点做 EdgeConv 聚合更新
    - 输入输出都是 (B, N, dim)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        k: int = 10,
        phi_hidden: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim 必须能被 num_heads 整除")
        self.k = k
        head_dim = out_dim // num_heads

        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
        )
        self.head_linears = nn.ModuleList([
            nn.Linear(in_dim + phi_hidden, head_dim) for _ in range(num_heads)
        ])
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_dim) - 全体节点特征
        返回: (B, N, out_dim) - 更新后的全体节点特征
        """
        B, N, C = x.shape
        k_eff = min(self.k, N - 1)
        
        # 1. 动态构建全局 KNN 图
        idx = knn_cosine(x, k_eff)                         # (B, N, k)
        
        # 2. 收集邻居特征
        neigh = gather_neighbors(x, idx)                    # (B, N, k, C)
        
        # 3. EdgeConv: diff = neighbor - self
        diff = neigh - x.unsqueeze(2).expand_as(neigh)      # (B, N, k, C)
        
        # 4. 边函数变换
        h = self.phi(diff)                                  # (B, N, k, phi_hidden)
        
        # 5. 最大池化聚合
        h_max = h.max(dim=2).values                         # (B, N, phi_hidden)
        
        # 6. 拼接自身 + 聚合特征
        concat = torch.cat([x, h_max], dim=-1)              # (B, N, C + phi_hidden)
        
        # 7. 多头处理并拼接
        out = torch.cat([lin(concat) for lin in self.head_linears], dim=-1)  # (B, N, out_dim)
        out = F.relu(out)
        out = self.drop(out)
        out = self.norm(out)
        return out

class GraphModule(nn.Module):
    """
    修复版图模块：全节点 GCN 更新 + SP 点特征提取
    """
    def __init__(
        self,
        *,
        k: int = 10,
        num_layers: int = 3,
        in_dim: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 128,
        phi_hidden: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # 堆叠多层图卷积（全节点更新）
        layers: List[nn.Module] = []
        cur = in_dim
        for li in range(num_layers):
            nxt = hidden_dim if li < num_layers - 1 else out_dim
            layers.append(EdgeConvMultiHead(
                cur, nxt, k=k, phi_hidden=phi_hidden,
                num_heads=num_heads, dropout=dropout
            ))
            cur = nxt
        self.layers = nn.ModuleList(layers)

        # 局部特征嵌入：原始 SP 4 维特征 → out_dim
        self.local_embed = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.LayerNorm(out_dim)
        )
        # 融合：原始 SP 局部 + 图卷积后 SP 全局
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.ReLU(), nn.LayerNorm(out_dim)
        )

    def forward(self, ddm: torch.Tensor, sp_indices: torch.Tensor) -> torch.Tensor:
        """
        ddm: (B, 4, 17, 11)
        sp_indices: (B,) - 每个样本的真实 SP 点索引 (0-186)
        返回: (B, out_dim)
        """
        B = ddm.shape[0]
        # (B, 4, 17, 11) → (B, 187, 4)
        x = ddm.permute(0, 2, 3, 1).contiguous().view(B, 17 * 11, 4)

        if sp_indices is None:
            raise ValueError("❌ 缺少 sp_indices 参数！")

        batch_idx = torch.arange(B, device=x.device)
        
        # 提取原始 SP 特征用于局部嵌入
        x0_sp = x[batch_idx, sp_indices]  # (B, 4)

        # 全节点图卷积：所有 187 个节点都参与更新
        for layer in self.layers:
            x = layer(x)  # (B, 187, dim) → (B, 187, next_dim)

        # 图卷积后提取 SP 点特征（此时 SP 已汇聚多跳信息）
        x_sp = x[batch_idx, sp_indices]  # (B, out_dim)

        # 融合原始局部 + 图卷积全局
        local = self.local_embed(x0_sp)     # (B, out_dim)
        return self.fuse(torch.cat([local, x_sp], dim=-1))  # (B, out_dim)
    
# 模块 13：时序位置编码（Sin/Cos Positional Encoding）
class SinCosPositionalEncoding(nn.Module):
    """
    标准的正弦余弦位置编码，用于 Transformer。
    因为 Transformer 本身不包含序列顺序信息，需要通过位置编码注入位置信息。
    """
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, Dim)
        return x + self.pe[: x.size(1)].unsqueeze(0)

# ⭐模块 14：STG-DNN 主网络（GCN + Transformer Sequence Model）
class STGDNN(nn.Module):
    """
    时空图深度神经网络 (STG-DNN)
    ✅ 已修改：使用 GCN（动态图卷积）进行空间特征提取 + Transformer 进行时序建模
    
    与 CNN 版本的区别：
    - CNN 版本：使用 CNNModule（多层卷积 + CBAM 注意力）提取空间特征
    - GCN 版本：使用 GraphModule（动态 KNN + EdgeConv）提取空间特征
    - GCN 能捕获 DDM 像素间的非局部依赖关系，更符合 STG-DNN 论文的设计
    """
    def __init__(
        self,
        aux_dim: int,
        *,
        time_steps: int = 10,
        # ✅ 修改：CNN 参数替换为 GCN 参数
        k: int = 10,                  # KNN 邻居数（原 CNN 版无此参数）
        gcn_layers: int = 3,          # 图卷积层数（对应原 CNN 的 len(cnn_channels)）
        gcn_hidden: int = 64,         # GCN 隐藏维度（对应原 CNN 的 cnn_channels 中间值）
        gcn_out: int = 128,           # GCN 输出维度（对应原 CNN 的 cnn_out）
        gcn_phi_hidden: int = 32,     # 边函数隐藏维度（GCN 特有参数）
        gcn_heads: int = 4,           # 多头数量（GCN 特有参数）
        # Transformer 参数（保持不变）
        d_model: int = 64,
        nhead: int = 8,
        num_transformer_layers: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        
        # ✅ 修改：将 CNNModule 替换为 GraphModule
        # 原代码：
        #   self.spatial_encoder = CNNModule(
        #       in_channels=4, hidden_channels=cnn_channels,
        #       out_dim=cnn_out, use_attention=use_attention, dropout=dropout,
        #   )
        # 新代码：使用 GraphModule 进行空间特征提取
        print(f"🔹 使用 GCN（动态图卷积）进行空间特征提取 (k={k}, layers={gcn_layers})")
        self.spatial_encoder = GraphModule(
            k=k,
            num_layers=gcn_layers,
            in_dim=4,                              # DDM 4 个通道
            hidden_dim=gcn_hidden,
            out_dim=gcn_out,
            phi_hidden=gcn_phi_hidden,
            num_heads=gcn_heads,
            dropout=dropout,
        )
        spatial_out_dim = gcn_out  # GraphModule 输出维度，与原 cnn_out 对应
        
        # 以下代码与 CNN 版本完全一致，无需修改
        # 特征投影层
        self.aux_proj = nn.Sequential(nn.Linear(aux_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.ddm_proj = nn.Sequential(nn.Linear(spatial_out_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.token_fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.LayerNorm(d_model))

        # 聚合 Token (类似于 BERT 的 CLS token)
        self.agg = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.agg, std=0.02)

        self.pos = SinCosPositionalEncoding(d_model=d_model, max_len=time_steps + 1)

        # Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        self.out_norm = nn.LayerNorm(d_model)
        
        # 预测头：按论文 Eq.(12) 输出每个时间步的风速序列 (T=10)。
        self.head = MLP(
            in_dim=d_model * (time_steps + 1),
            hidden_dims=[dim_feedforward, dim_feedforward],
            out_dim=time_steps,
            dropout=dropout,
            act=nn.ReLU,
        )

    def forward(self, ddm_seq: torch.Tensor, aux_seq: torch.Tensor, sp_indices: torch.Tensor) -> torch.Tensor:
        """
        前向传播（与 CNN 版本逻辑一致，仅空间编码器不同）
        Args:
            ddm_seq: (B, T, 4, 17, 11) - DDM 序列
            aux_seq: (B, T, aux_dim) - 辅助特征序列
            sp_indices: (B, T) - 每个时间步的SP点索引 🔴 必需参数
        Returns:
            (B, T) - 每个时间步的预测风速
        """
        B, T = ddm_seq.shape[0], ddm_seq.shape[1]
        if T != self.time_steps:
            raise ValueError(f"预期时间步 T={self.time_steps}, 实际输入 {T}")
        
        # 验证SP索引参数
        if sp_indices is None:
            raise ValueError(
                "❌ 缺少 sp_indices 参数！\n"
                "STGDNN.forward() 必须接收每个时间步的SP点索引 (B, T)。\n"
                "请确保从 DataLoader 中正确传递 'sp_idx' 字段。"
            )

        # 1. 空间特征提取
        # ✅ 修改说明：这里调用的是 GraphModule.forward()
        # GraphModule 内部会将 (B*T, 4, 17, 11) 重排为 (B*T, 187, 4) 的图节点
        # 然后通过动态 KNN + EdgeConv 提取空间特征
        # 🔴 新增：传递每个时间步的SP索引
        ddm_flat = ddm_seq.reshape(B * T, 4, 17, 11)
        sp_idx_flat = sp_indices.reshape(B * T)  # (B*T,) - 强制要求,不再可选
        spatial_features = self.spatial_encoder(ddm_flat, sp_idx_flat)  # (B*T, gcn_out)
        spatial_features = self.ddm_proj(spatial_features).reshape(B, T, -1)  # (B, T, d_model)

        # 2. 辅助特征处理（与 CNN 版本完全一致）
        aux_features = self.aux_proj(aux_seq.reshape(B * T, -1)).reshape(B, T, -1)  # (B, T, d_model)

        # 3. 特征融合（与 CNN 版本完全一致）
        fused_tokens = self.token_fuse(torch.cat([spatial_features, aux_features], dim=-1))  # (B, T, d_model)

        # 4. Transformer 时序建模（与 CNN 版本完全一致）
        x = torch.cat([self.agg.expand(B, 1, -1), fused_tokens], dim=1)  # (B, T+1, d_model)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.out_norm(x)
        
        # 5. 按论文 Eq.(12): 先 flatten LN(xL)，再经 MLP 输出 10 个风速反演结果。
        flat_tokens = x.reshape(B, -1)  # (B, (T+1)*d_model)
        return self.head(flat_tokens)  # (B, T)


def _forward_transformer_layer_detailed(
    layer: nn.TransformerEncoderLayer, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    以公开模块重构单层 Transformer（Pre-LN）前向，返回：
    - y: 注意力残差后的中间态
    - m: FFN 分支输出（未加残差）
    - l: 该层最终输出
    """
    if not layer.norm_first:
        raise ValueError("ReplaceMe 剪枝仅支持 norm_first=True 的 TransformerEncoderLayer。")

    x_norm = layer.norm1(x)
    try:
        attn_out = layer.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
            is_causal=False,
        )[0]
    except TypeError:
        attn_out = layer.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
        )[0]
    y = x + layer.dropout1(attn_out)

    ff_in = layer.norm2(y)
    ff_hidden = layer.linear1(ff_in)
    ff_hidden = layer.activation(ff_hidden)
    ff_hidden = layer.dropout(ff_hidden)
    m = layer.dropout2(layer.linear2(ff_hidden))
    l = y + m
    return y, m, l


def _build_tokens_before_transformer(
    model: STGDNN, ddm_seq: torch.Tensor, aux_seq: torch.Tensor, sp_indices: torch.Tensor
) -> torch.Tensor:
    """
    复用 STGDNN 的 token 构建路径，得到进入 Transformer 前（含位置编码）的序列表示。
    """
    bsz, tlen = ddm_seq.shape[0], ddm_seq.shape[1]
    if tlen != model.time_steps:
        raise ValueError(f"预期时间步 T={model.time_steps}, 实际输入 {tlen}")

    ddm_flat = ddm_seq.reshape(bsz * tlen, 4, 17, 11)
    sp_idx_flat = sp_indices.reshape(bsz * tlen)
    spatial_features = model.spatial_encoder(ddm_flat, sp_idx_flat)
    spatial_features = model.ddm_proj(spatial_features).reshape(bsz, tlen, -1)
    aux_features = model.aux_proj(aux_seq.reshape(bsz * tlen, -1)).reshape(bsz, tlen, -1)
    fused_tokens = model.token_fuse(torch.cat([spatial_features, aux_features], dim=-1))
    x = torch.cat([model.agg.expand(bsz, 1, -1), fused_tokens], dim=1)
    return model.pos(x)


def _sync_transformer_encoder_metadata(model: STGDNN) -> None:
    """
    强制同步 TransformerEncoder 的层数元数据，避免包装/复制后出现属性不一致。
    """
    n_layers = int(len(model.transformer.layers))
    if hasattr(model.transformer, "num_layers"):
        model.transformer.num_layers = n_layers
    if hasattr(model.transformer, "_num_layers"):
        model.transformer._num_layers = n_layers


def apply_replaceme_depth_pruning(
    model: STGDNN,
    calibration_loader: DataLoader,
    device: str,
    *,
    prune_layers: int = 2,
    calibration_batches: int = 8,
    objective: str = "cosine",
    cosine_steps: int = 300,
    cosine_lr: float = 1e-2,
    cosine_batch_size: int = 4096,
    cosine_l2_reg: float = 0.0,
    cosine_early_stop_patience: int = 20,
    cosine_early_stop_min_delta: float = 1e-5,
    cosine_eval_interval: int = 10,
) -> Dict[str, Any]:
    """
    按 ReplaceMe 思路执行训练后（无 healing）深度剪枝（block-level depth pruning）：
    1) 用 cosine 距离选择连续剪枝区间；
    2) 估计线性变换 T：
       - objective='ls'：最小二乘闭式/数值稳定解
       - objective='cosine'：优化 Eq.(8) 近似目标 cos(M_i T, L_{i+n}-Y_i)
    3) 将 T 融合到第 i 层 FFN 下投影，并删除 i+1..i+n 层。
    """
    if prune_layers <= 0:
        raise ValueError("prune_layers 必须为正整数。")
    if calibration_batches <= 0:
        raise ValueError("calibration_batches 必须为正整数。")
    if cosine_steps <= 0:
        raise ValueError("cosine_steps 必须为正整数。")
    if cosine_lr <= 0:
        raise ValueError("cosine_lr 必须大于 0。")
    if cosine_batch_size <= 0:
        raise ValueError("cosine_batch_size 必须为正整数。")
    if cosine_l2_reg < 0:
        raise ValueError("cosine_l2_reg 不能小于 0。")
    if cosine_early_stop_patience <= 0:
        raise ValueError("cosine_early_stop_patience 必须为正整数。")
    if cosine_early_stop_min_delta < 0:
        raise ValueError("cosine_early_stop_min_delta 不能小于 0。")
    if cosine_eval_interval <= 0:
        raise ValueError("cosine_eval_interval 必须为正整数。")
    objective = objective.strip().lower()
    if objective not in ("ls", "cosine"):
        raise ValueError("objective 仅支持 'ls' 或 'cosine'。")

    layers = model.transformer.layers
    num_layers = len(layers)
    if num_layers < 2:
        raise ValueError("Transformer 层数过少，无法执行深度剪枝。")
    if prune_layers >= num_layers:
        raise ValueError(f"prune_layers={prune_layers} 必须小于当前层数 {num_layers}。")

    candidate_starts = list(range(0, num_layers - prune_layers))
    if not candidate_starts:
        raise ValueError("没有可用的连续剪枝候选区间。")

    model.eval()
    # 固定同一批 calibration 数据，保证剪枝前后评估与各步骤可严格对齐。
    calibration_batches_cache: List[Dict[str, torch.Tensor]] = []
    for batch in calibration_loader:
        calibration_batches_cache.append(
            {
                "ddm": batch["ddm"].to(device, non_blocking=True),
                "aux": batch["aux"].to(device, non_blocking=True),
                "sp_idx": batch["sp_idx"].to(device, non_blocking=True),
                "y": batch["y"].to(device, non_blocking=True),
            }
        )
        if len(calibration_batches_cache) >= calibration_batches:
            break
    if not calibration_batches_cache:
        raise RuntimeError("剪枝失败：校准数据为空或未成功读取批次。")

    def _eval_rmse_on_calibration() -> float:
        ys: List[np.ndarray] = []
        yh: List[np.ndarray] = []
        with torch.no_grad():
            for b in calibration_batches_cache:
                pred = model(b["ddm"], b["aux"], b["sp_idx"])
                ys.append(b["y"].detach().cpu().numpy())
                yh.append(pred.detach().cpu().numpy())
        y_true = np.concatenate(ys, axis=0)
        y_pred = np.concatenate(yh, axis=0)
        return float(compute_metrics(y_true, y_pred)["rmse"])

    calib_rmse_before = _eval_rmse_on_calibration()

    dist_sums = np.zeros(len(candidate_starts), dtype=np.float64)
    dist_counts = np.zeros(len(candidate_starts), dtype=np.int64)

    # Step-1: 按论文思想，用 cosine(L_i, L_{i+n}) 选择最优剪枝起点 i。
    with torch.no_grad():
        for b in calibration_batches_cache:
            x = _build_tokens_before_transformer(model, b["ddm"], b["aux"], b["sp_idx"])

            layer_outputs: List[torch.Tensor] = []
            cur = x
            for layer in layers:
                _, _, cur = _forward_transformer_layer_detailed(layer, cur)
                layer_outputs.append(cur)

            for pos, start in enumerate(candidate_starts):
                left = layer_outputs[start]
                right = layer_outputs[start + prune_layers]
                left_n = F.normalize(left, dim=-1)
                right_n = F.normalize(right, dim=-1)
                cos_dist = 1.0 - (left_n * right_n).sum(dim=-1)
                dist_sums[pos] += float(cos_dist.mean().item())
                dist_counts[pos] += 1

    if int(dist_counts.max()) == 0:
        raise RuntimeError("剪枝失败：校准数据为空或未成功读取批次。")

    mean_dist = dist_sums / np.maximum(dist_counts, 1)
    best_pos = int(np.argmin(mean_dist))
    cut_idx = int(candidate_starts[best_pos])

    # Step-2: 固定 cut_idx 后估计线性变换 T。
    m_list: List[torch.Tensor] = []
    rhs_list: List[torch.Tensor] = []
    with torch.no_grad():
        for b in calibration_batches_cache:
            cur = _build_tokens_before_transformer(model, b["ddm"], b["aux"], b["sp_idx"])

            yi = None
            mi = None
            l_target = None
            for li, layer in enumerate(layers):
                y, m, l = _forward_transformer_layer_detailed(layer, cur)
                if li == cut_idx:
                    yi = y
                    mi = m
                if li == cut_idx + prune_layers:
                    l_target = l
                cur = l

            if yi is None or mi is None or l_target is None:
                raise RuntimeError("剪枝失败：关键激活张量未能正确收集。")

            m_flat = mi.reshape(-1, mi.size(-1)).detach()
            rhs_flat = (l_target - yi).reshape(-1, yi.size(-1)).detach()
            m_list.append(m_flat)
            rhs_list.append(rhs_flat)

    optimization_loss = float("nan")
    early_stopped = False
    effective_steps = 0
    best_step = 0
    if objective == "ls":
        # 在 CPU 上做最小二乘，避免不同 CUDA 版本下 lstsq 可用性差异。
        m_mat = torch.cat(m_list, dim=0).to(device="cpu", dtype=torch.float64)
        rhs_mat = torch.cat(rhs_list, dim=0).to(device="cpu", dtype=torch.float64)
        try:
            t_mat = torch.linalg.lstsq(m_mat, rhs_mat).solution
        except RuntimeError:
            t_mat = torch.linalg.pinv(m_mat) @ rhs_mat
        with torch.no_grad():
            pred = m_mat @ t_mat
            optimization_loss = float(torch.mean(1.0 - F.cosine_similarity(pred, rhs_mat, dim=-1, eps=1e-8)).item())
    else:
        # Cosine 目标（论文 Eq.(8) 近似）：min_T cos(M_i T, L_{i+n} - Y_i)
        if device.startswith("cuda") and torch.cuda.is_available():
            opt_device = torch.device(device)
        else:
            opt_device = torch.device("cpu")
        m_mat = torch.cat(m_list, dim=0).to(device=opt_device, dtype=torch.float32)
        rhs_mat = torch.cat(rhs_list, dim=0).to(device=opt_device, dtype=torch.float32)
        d_model = int(m_mat.size(1))
        t_param = nn.Parameter(torch.eye(d_model, device=opt_device, dtype=torch.float32))
        opt_t = torch.optim.Adam([t_param], lr=cosine_lr)
        n_tokens = int(m_mat.size(0))
        mb_size = min(max(1, int(cosine_batch_size)), n_tokens)
        best_monitor = float("inf")
        best_cosine_loss = float("inf")
        best_t = t_param.detach().clone()
        no_improve = 0
        eval_interval = max(1, int(cosine_eval_interval))
        for step in range(int(cosine_steps)):
            if mb_size == n_tokens:
                m_batch = m_mat
                rhs_batch = rhs_mat
            else:
                idx = torch.randint(0, n_tokens, (mb_size,), device=opt_device)
                m_batch = m_mat.index_select(0, idx)
                rhs_batch = rhs_mat.index_select(0, idx)
            pred = m_batch @ t_param
            loss = torch.mean(1.0 - F.cosine_similarity(pred, rhs_batch, dim=-1, eps=1e-8))
            if cosine_l2_reg > 0:
                loss = loss + float(cosine_l2_reg) * torch.mean(t_param.pow(2))
            opt_t.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([t_param], max_norm=1.0)
            opt_t.step()
            effective_steps = step + 1
            need_eval = (
                effective_steps == 1
                or effective_steps % eval_interval == 0
                or effective_steps == int(cosine_steps)
            )
            if need_eval:
                with torch.no_grad():
                    pred_all = m_mat @ t_param
                    cos_loss_full = torch.mean(
                        1.0 - F.cosine_similarity(pred_all, rhs_mat, dim=-1, eps=1e-8)
                    )
                    monitor = cos_loss_full
                    if cosine_l2_reg > 0:
                        monitor = monitor + float(cosine_l2_reg) * torch.mean(t_param.pow(2))
                    monitor_v = float(monitor.item())
                    cos_v = float(cos_loss_full.item())
                if monitor_v < best_monitor - float(cosine_early_stop_min_delta):
                    best_monitor = monitor_v
                    best_cosine_loss = cos_v
                    best_t = t_param.detach().clone()
                    no_improve = 0
                    best_step = effective_steps
                else:
                    no_improve += 1
                    if no_improve >= int(cosine_early_stop_patience):
                        early_stopped = True
                        break
        with torch.no_grad():
            t_param.data.copy_(best_t)
            optimization_loss = best_cosine_loss
            t_mat = t_param.detach().to(device="cpu", dtype=torch.float32)

    # Step-3: 将 T 融合进 cut_idx 层 FFN 的 down-proj（linear2）。
    target_layer = layers[cut_idx]
    with torch.no_grad():
        t_use = t_mat.to(dtype=target_layer.linear2.weight.dtype, device=target_layer.linear2.weight.device)
        w2 = target_layer.linear2.weight.data
        b2 = target_layer.linear2.bias.data if target_layer.linear2.bias is not None else None
        fused_w2 = t_use.t().matmul(w2)
        target_layer.linear2.weight.data.copy_(fused_w2)
        if b2 is not None:
            fused_b2 = b2.matmul(t_use)
            target_layer.linear2.bias.data.copy_(fused_b2)

    # Step-4: 删除被替换的连续层 [cut_idx+1, cut_idx+prune_layers]。
    # 使用“原地删除”而非整体替换 ModuleList，保持对象引用稳定（对 DataParallel 更安全）。
    pruned_indices = list(range(cut_idx + 1, cut_idx + prune_layers + 1))
    for li in sorted(pruned_indices, reverse=True):
        del model.transformer.layers[li]
    _sync_transformer_encoder_metadata(model)
    layers_after = int(len(model.transformer.layers))

    calib_rmse_after = _eval_rmse_on_calibration()

    return {
        "method": f"ReplaceMe-{objective.upper()}",
        "objective": objective,
        "cut_index": cut_idx,
        "pruned_indices": pruned_indices,
        "layers_before": num_layers,
        "layers_after": layers_after,
        "selected_cosine_distance": float(mean_dist[best_pos]),
        "calibration_batches": calibration_batches,
        "optimization_loss": optimization_loss,
        "calibration_rmse_before": calib_rmse_before,
        "calibration_rmse_after": calib_rmse_after,
        "calibration_rmse_delta": float(calib_rmse_after - calib_rmse_before),
        "early_stopped": early_stopped,
        "effective_steps": effective_steps,
        "best_step": best_step,
        "cosine_early_stop_patience": cosine_early_stop_patience,
        "cosine_early_stop_min_delta": cosine_early_stop_min_delta,
    }


# ⭐模块 15：训练配置与特征选择（Training Config + Aux Feature Schema）
@dataclass
class TrainConfig:
    """训练参数配置类"""
    model: str = "stg_dnn"  # 固定为 stg_dnn
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deterministic: bool = True

    dataset_pkl: Path = Path("stg_dnn_dataset_full_batch.pkl")
    ddm_store: Optional[Path] = None

    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True  # 加速数据传输到 GPU

    time_steps: int = 10

    # GCN 参数：
    k: int = 10                  # KNN 的 K 值（每个节点连接的邻居数）
    gcn_layers: int = 3          # 图卷积层数
    gcn_hidden: int = 64         # GCN 中间层隐藏维度
    gcn_out: int = 128           # GCN 输出特征维度
    gcn_phi_hidden: int = 32     # EdgeConv 边函数的隐藏层维度
    gcn_heads: int = 4           # EdgeConv 多头数量

    # Transformer 参数
    d_model: int = 64
    nhead: int = 8
    transformer_layers: int = 8
    ff_dim: int = 256
    dropout: float = 0.1

    # ReplaceMe 风格深度剪枝（训练后、无 healing）
    enable_replaceme_pruning: bool = True
    replaceme_prune_layers: int = 2
    replaceme_calibration_batches: int = 8
    replaceme_objective: str = "cosine"  # "cosine" 或 "ls"
    replaceme_cosine_steps: int = 300
    replaceme_cosine_lr: float = 1e-2
    replaceme_cosine_batch_size: int = 4096
    replaceme_cosine_l2_reg: float = 0.0
    replaceme_cosine_early_stop_patience: int = 20
    replaceme_cosine_early_stop_min_delta: float = 1e-5
    replaceme_cosine_eval_interval: int = 10

    # 优化参数
    lr: float = 5e-4
    max_epochs: int = 1
    weight_decay: float = 0.0
    patience: int = 20

    # 学习率调度器参数
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    min_lr: float = 1e-6
    step_size: int = 30
    step_gamma: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    max_lr: float = 1e-3
    pct_start: float = 0.3

    out_dir: Path = Path("runs/stg_dnn_repro")
    save_best: bool = True
    norm_stats_path: Optional[Path] = None

def build_aux_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "sp_lon", "sp_lat", "sp_inc_angle",
        "prn_code",
        "range_corr_gain", "log_rcg", "sqrt_rcg",
        "ddm_les", "log_les",
        "ddm_nbrcs", "log_nbrcs",
        "sc_vel_x", "sc_vel_y", "sc_vel_z"
    ]
    
    # 🔴 1. 计算动态SP点索引（基于数据集中的sp_delay_bin和sp_doppler_bin）
    if "sp_delay_bin" not in df.columns or "sp_doppler_bin" not in df.columns:
        raise KeyError(
            "❌ 数据集必须包含 'sp_delay_bin' 和 'sp_doppler_bin' 列！\n"
            "这两列用于确定每个DDM的真实镜面反射点(SP)位置。\n"
            "请检查数据预处理流程是否正确生成了这些列。"
        )
    
    # 计算SP点在展平后的索引: sp_index = delay_bin * 11 + doppler_bin
    # 添加调试信息确认维度
    delay_max = df["sp_delay_bin"].max()
    doppler_max = df["sp_doppler_bin"].max()
    print(f"  sp_delay_bin 范围: 0-{delay_max}")
    print(f"  sp_doppler_bin 范围: 0-{doppler_max}")

    # 1. 检查数据类型和范围
    delay_vals = df["sp_delay_bin"].dropna()
    doppler_vals = df["sp_doppler_bin"].dropna()

    print(f"  sp_delay_bin 范围: {delay_vals.min():.4f} - {delay_vals.max():.4f}")
    print(f"  sp_doppler_bin 范围: {doppler_vals.min():.4f} - {doppler_vals.max():.4f}")

    # 2. 四舍五入为整数 (亚像素 bin 位置 → 最近的整数 bin)
    delay_int = df["sp_delay_bin"].round().astype(np.int64)
    doppler_int = df["sp_doppler_bin"].round().astype(np.int64)

    # 3. 钳位到有效范围 (delay: 0-16, doppler: 0-10)
    delay_int = delay_int.clip(0, 16)
    doppler_int = doppler_int.clip(0, 10)

    # 4. 计算展平后的索引
    # DDM 形状: (4, 17, 11) → 展平为 (187,)
    # permute(0,2,3,1) → view(B, 17*11, 4) → 索引 = delay * 11 + doppler
    df["sp_index"] = (delay_int * 11 + doppler_int).astype(np.int64)
    
    # 验证索引范围
    sp_min, sp_max = df["sp_index"].min(), df["sp_index"].max()
    if sp_min < 0 or sp_max > 186:
        raise ValueError(
            f"❌ SP点索引超出有效范围！\n"
            f"当前范围: {sp_min}-{sp_max}, 期望范围: 0-186 (17×11-1)\n"
            f"请检查 sp_delay_bin 和 sp_doppler_bin 的值是否正确。"
        )
    
    print(f"  ✅ 已计算动态SP点索引 (范围: {sp_min}-{sp_max}, 共{len(df['sp_index'].unique())}个不同位置)")
    
    # 🔴 2. RCG 对数变换（解决 RCG 分叉问题）
    if "range_corr_gain" in df.columns:
        rcg = df["range_corr_gain"].fillna(1.0).clip(lower=0.01)
        df["log_rcg"] = np.log10(rcg)
        df["sqrt_rcg"] = np.sqrt(rcg)
        print("  ✅ 已添加 RCG 变换特征: log10(RCG), sqrt(RCG)")
    
    # 🔴 3. LES 对数变换（LES 范围很宽，对数化后更均匀）
    if "ddm_les" in df.columns:
        les = df["ddm_les"].fillna(1.0).clip(lower=0.01)
        df["log_les"] = np.log10(les)
        print("  ✅ 已添加 LES 变换特征: log10(LES)")
    
    # 🔴 4. NBRCS 对数变换
    if "ddm_nbrcs" in df.columns:
        nbrcs = df["ddm_nbrcs"].fillna(1.0).clip(lower=0.01)
        df["log_nbrcs"] = np.log10(nbrcs)
        print("  ✅ 已添加 NBRCS 变换特征: log10(NBRCS)")
    
    cols = [c for c in preferred if c in df.columns]
    
    # 防止标签泄露（保持原有逻辑）
    forbidden = {'WS', 'ERA5', 'wind_speed', 'target'}
    leaked = set(cols) & forbidden
    if leaked:
        raise ValueError(f"❌ 特征列包含标签泄露！{leaked}")
    
    # 填充 NaN（保持原有逻辑）
    for c in cols:
        if df[c].isna().sum() > len(df) * 0.5:
            print(f"⚠️ 警告: {c} 列有 {df[c].isna().sum()/len(df)*100:.1f}% NaN")
        df[c] = df[c].fillna(0.0)
    
    print(f"✅ 使用 {len(cols)} 个辅助特征: {cols}")
    return cols

# 模块 16：训练与验证函数（支持 AMP 混合精度训练）
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """????/??????? AMP?"""
    model.eval()
    ys: List[np.ndarray] = []
    yh: List[np.ndarray] = []

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    amp_device_type = "cuda" if use_amp else "cpu"

    for batch in loader:
        ddm = batch["ddm"].to(device, non_blocking=True)
        aux = batch["aux"].to(device, non_blocking=True)
        y = batch["y"]
        sp_idx = batch["sp_idx"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(ddm, aux, sp_idx)

        ys.append(y.cpu().numpy())
        yh.append(pred.cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yh, axis=0)
    return compute_metrics(y_true, y_pred)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """???? Epoch??? MSE ????? AMP?"""
    model.train()
    losses: List[float] = []
    use_amp = scaler is not None and device.startswith("cuda") and torch.cuda.is_available()
    amp_device_type = "cuda" if use_amp else "cpu"

    for batch in loader:
        ddm = batch["ddm"].to(device, non_blocking=True)
        aux = batch["aux"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        sp_idx = batch["sp_idx"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(ddm, aux, sp_idx)
            loss = F.mse_loss(pred, y, reduction='mean')

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")

def make_model(cfg: TrainConfig, aux_dim: int) -> nn.Module:
    """
    根据配置实例化 STG-DNN 模型
    """
    return STGDNN(
        aux_dim=aux_dim,
        time_steps=cfg.time_steps,
        # ✅ 修改：传入 GCN 参数（替代原来的 CNN 参数）
        # 原代码：
        #   use_attention=cfg.use_attention,
        #   cnn_channels=cfg.cnn_channels,
        #   cnn_out=cfg.cnn_out,
        # 新代码：
        k=cfg.k,                          # KNN 邻居数
        gcn_layers=cfg.gcn_layers,         # 图卷积层数
        gcn_hidden=cfg.gcn_hidden,         # GCN 隐藏维度
        gcn_out=cfg.gcn_out,               # GCN 输出维度
        gcn_phi_hidden=cfg.gcn_phi_hidden, # 边函数隐藏维度
        gcn_heads=cfg.gcn_heads,           # 多头数量
        # Transformer 参数（保持不变）
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_transformer_layers=cfg.transformer_layers,
        dim_feedforward=cfg.ff_dim,
        dropout=cfg.dropout,
    )

# 模块 18：结果可视化工具类（支持高斯核密度估计的散点图）
class WindSpeedVisualizer:
    """
    风速预测结果可视化工具类
    所有图表自动保存到 save_dir，并支持中文字体
    """
    def __init__(self, save_dir: Path, model_name: str = "Model"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # 颜色方案
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.cmap = 'viridis'

    def plot_learning_curves(self, train_losses: list, val_rmse: list, train_rmse: list):
        """绘制学习曲线：训练损失 + 训练/验证 RMSE"""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=self.colors[0])
        ax1.plot(train_losses, label='Train Loss', color=self.colors[0])
        ax1.tick_params(axis='y', labelcolor=self.colors[0])
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('RMSE (m/s)', color=self.colors[1])
        ax2.plot(train_rmse, label='Train RMSE', color=self.colors[1], linestyle='--')
        ax2.plot(val_rmse, label='Val RMSE', color=self.colors[2])
        ax2.tick_params(axis='y', labelcolor=self.colors[1])

        fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
        plt.title(f'{self.model_name} Learning Curves')
        plt.tight_layout()
        plt.savefig(self.save_dir / "learning_curves.png", dpi=600, bbox_inches='tight')
        plt.close()

    def plot_scatter_density_kde(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              set_name: str = "Test",
                              save_name: Optional[str] = None):
        """
        ??????????
        - ?? 2D ?????
        - ????????
        - ? LogNorm ????????????
        """
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12

        fig, ax = plt.subplots(figsize=(8, 7))

        # ??????
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        x = y_true[mask]
        y = y_pred[mask]

        # ?????
        if len(x) < 10:
            print(f"  ?? ??: {set_name} ?????? ({len(x)} ???)??????")
            plt.close()
            return

        # ??????
        rmse = np.sqrt(np.mean((y - x) ** 2))
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        bias = np.mean(y - x)

        # ????
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 1.0, 0.0

        # ????
        min_val = 0.0
        max_val = float(max(x.max(), y.max()) + 1.0)

        # ????????2D histogram + gaussian smoothing + LogNorm
        print(f"  ?? {set_name} ???????hist2d + LogNorm?...")
        bins = 260
        hist, x_edges, y_edges = np.histogram2d(
            x,
            y,
            bins=bins,
            range=[[min_val, max_val], [min_val, max_val]],
        )

        # ??????????????/???
        hist_smooth = gaussian_filter(hist, sigma=1.0, mode='nearest')
        hist_smooth = np.where(hist_smooth > 0, hist_smooth, np.nan)
        positive = hist_smooth[np.isfinite(hist_smooth)]
        if positive.size == 0:
            print(f"  ?? {set_name} ??????????")
            plt.close()
            return

        vmin = float(np.nanpercentile(positive, 2.0))
        vmax = float(np.nanpercentile(positive, 99.8))
        if vmax <= vmin:
            vmax = max(vmin * 10.0, vmin + 1e-6)

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            hist_smooth.T,
            cmap='jet',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading='auto',
            zorder=1,
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label('Density (log scale)', fontsize=12)

        # 1:1 ?????????
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='1:1 Line')

        # ?????????
        fit_x = np.linspace(min_val, max_val, 100)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, 'r-', lw=1.5,
                label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

        # ???????
        ax.set_xlabel('ERA5 Wind Speed (m/s)', fontsize=14)
        ax.set_ylabel('Predicted Wind Speed (m/s)', fontsize=14)
        ax.set_title(f'Scatter Density Plot\nRMSE={rmse:.3f}, R={corr:.3f}, Bias={bias:.3f}',
                    fontsize=14)

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        if save_name is None:
            save_name = f"scatter_density_kde_{set_name.lower()}.png"
        plt.savefig(self.save_dir / save_name, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  ? ??????: {self.save_dir / save_name}")

    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, set_name: str):
        """误差分布直方图"""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        error = y_pred[mask] - y_true[mask]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(error, bins=80, range=(-15, 15), density=True, color='steelblue', alpha=0.8)
        ax.axvline(0, color='r', linestyle='--', lw=1.5)
        ax.axvline(np.mean(error), color='orange', linestyle='-', lw=1.5, 
                   label=f'Mean={np.mean(error):.3f}')
        ax.set_xlabel('Prediction Error (m/s)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'{set_name} Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / f"error_distribution_{set_name.lower()}.png", dpi=600, bbox_inches='tight')
        plt.close()

    def plot_interval_metrics(self, interval_metrics: list):
        """分风速区间 RMSE 柱状图"""
        intervals = [(d['lo'], d['hi']) for d in interval_metrics]
        labels = [f"{lo}-{hi if hi != float('inf') else '∞'}" for lo, hi in intervals]
        rmses = [d['rmse'] for d in interval_metrics]
        ns = [d['n'] for d in interval_metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, rmses, color='steelblue', alpha=0.8)
        ax.set_xlabel('Wind Speed Range (m/s)', fontsize=12)
        ax.set_ylabel('RMSE (m/s)', fontsize=12)
        ax.set_title('RMSE by Wind Speed Interval', fontsize=14)

        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'n={n}', ha='center', va='bottom', fontsize=10)

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.save_dir / "interval_rmse.png", dpi=600, bbox_inches='tight')
        plt.close()

    def generate_all_figures(self, y_true: np.ndarray, y_pred: np.ndarray, set_name: str,
                             lat: Optional[np.ndarray] = None, lon: Optional[np.ndarray] = None,
                             interval_metrics: Optional[list] = None):
        """
        生成所有图表（精简版）
        主要生成：KDE 散点图、误差分布图、分区间 RMSE
        """
        print(f"正在生成 {set_name} 集图表...")
        
        # 核心图表：KDE 着色散点图（论文标准样式）
        self.plot_scatter_density_kde(y_true, y_pred, set_name)
        
        # 误差分布
        self.plot_error_distribution(y_true, y_pred, set_name)
        
        # 分区间指标
        if interval_metrics is not None:
            self.plot_interval_metrics(interval_metrics)
        
        print(f"  ✅ {set_name} 集所有图表生成完成")

# 模块 19：学习率调度器构建函数（支持多种调度策略）
def build_scheduler(
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> Tuple[Optional[Any], str]:
    """
    根据配置构建学习率调度器。
    
    Args:
        cfg: 训练配置
        optimizer: 优化器
        steps_per_epoch: 每个 epoch 的迭代步数（用于 OneCycleLR）
    
    Returns:
        scheduler: 学习率调度器（如果为 None 则不使用）
        scheduler_type: 调度器更新时机 ("epoch" | "step" | "plateau")
    """
    if cfg.scheduler == "none" or cfg.scheduler is None:
        print("学习率调度: 不使用（固定学习率）")
        return None, "none"
    
    elif cfg.scheduler == "step":
        # StepLR: 每隔固定 epoch 衰减学习率
        # 适用于：简单场景，需要手动调整 step_size
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.step_gamma,
        )
        print(f"学习率调度: StepLR (每 {cfg.step_size} 轮衰减 {cfg.step_gamma}x)")
        return scheduler, "epoch"
    
    elif cfg.scheduler == "cosine":
        # CosineAnnealingLR: 余弦退火，从初始 lr 平滑降到 min_lr
        # 适用于：训练轮数固定的场景
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_epochs,
            eta_min=cfg.min_lr,
        )
        print(f"学习率调度: CosineAnnealing (T_max={cfg.max_epochs}, min_lr={cfg.min_lr})")
        return scheduler, "epoch"
    
    elif cfg.scheduler == "cosine_warmup":
        # Cosine with Warmup: 先线性预热，再余弦退火（推荐用于 Transformer）
        # 这是目前最流行的调度策略之一
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=cfg.warmup_epochs,
            max_epochs=cfg.max_epochs,
            warmup_lr=cfg.warmup_lr,
            base_lr=cfg.lr,
            min_lr=cfg.min_lr,
        )
        print(f"学习率调度: CosineWarmup (warmup={cfg.warmup_epochs}轮, "
              f"lr: {cfg.warmup_lr} → {cfg.lr} → {cfg.min_lr})")
        return scheduler, "epoch"
    
    elif cfg.scheduler == "onecycle":
        # OneCycleLR: 超收敛策略，lr 先升后降，配合动量变化
        # 适用于：快速训练，通常能在更少 epoch 内收敛
        total_steps = steps_per_epoch * cfg.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=total_steps,
            pct_start=cfg.pct_start,
            anneal_strategy="cos",
            div_factor=cfg.max_lr / cfg.warmup_lr,      # 初始 lr = max_lr / div_factor
            final_div_factor=cfg.max_lr / cfg.min_lr,   # 最终 lr = max_lr / final_div_factor
        )
        print(f"学习率调度: OneCycleLR (max_lr={cfg.max_lr}, total_steps={total_steps})")
        return scheduler, "step"  # OneCycleLR 需要每个 step 更新
    
    elif cfg.scheduler == "reduce_plateau":
        # ReduceLROnPlateau: 当指标停止改善时自动降低学习率
        # 适用于：不确定需要多少轮训练的场景
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            min_lr=cfg.min_lr,
            verbose=True,
        )
        print(f"学习率调度: ReduceLROnPlateau (factor={cfg.plateau_factor}, "
              f"patience={cfg.plateau_patience})")
        return scheduler, "plateau"
    
    else:
        raise ValueError(f"未知的学习率调度器: {cfg.scheduler}")

class CosineAnnealingWarmupScheduler:
    """
    带 Warmup 的余弦退火调度器（自定义实现）。
    
    学习率变化曲线：
    1. Warmup 阶段 (0 ~ warmup_epochs): lr 从 warmup_lr 线性增加到 base_lr
    2. Cosine 阶段 (warmup_epochs ~ max_epochs): lr 从 base_lr 余弦退火到 min_lr
    
    这是 Transformer 模型训练的黄金标准调度策略。
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_lr: float,
        base_lr: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # 初始化为 warmup 起始学习率
        self._set_lr(warmup_lr)
    
    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup 阶段：线性增加
            if self.warmup_epochs > 0:
                alpha = self.current_epoch / self.warmup_epochs
                lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * alpha
            else:
                lr = self.base_lr
        else:
            # Cosine 阶段：余弦退火
            cosine_epochs = self.max_epochs - self.warmup_epochs
            progress = (self.current_epoch - self.warmup_epochs) / max(cosine_epochs, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        self._set_lr(lr)
    
    def state_dict(self) -> Dict[str, Any]:
        return {"current_epoch": self.current_epoch}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_epoch = state_dict["current_epoch"]

# 模块 20：诊断工具函数（检查 RCG 分布和高风速区域的双峰现象）
def _diagnose_rcg_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """
    诊断 RCG 分布：检查是否存在双模态（混合口径的标志）
    """
    if 'range_corr_gain' not in df.columns:
        print("⚠️ 缺少 range_corr_gain 列，跳过 RCG 诊断")
        return
    
    rcg = df['range_corr_gain'].dropna()
    if len(rcg) == 0:
        return
    
    print("\n" + "=" * 50)
    print("📊 RCG 分布诊断")
    print("=" * 50)
    print(f"  样本数: {len(rcg)}")
    print(f"  范围: [{rcg.min():.4e}, {rcg.max():.4e}]")
    print(f"  均值: {rcg.mean():.4e}")
    print(f"  中位数: {rcg.median():.4e}")
    print(f"  标准差: {rcg.std():.4e}")
    
    # 检查 log(RCG) 是否双峰
    log_rcg = np.log10(rcg.clip(lower=1e-10).values)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：log10(RCG) 分布
    axes[0].hist(log_rcg, bins=100, density=True, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('log10(RCG)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('RCG Distribution (log scale)\nIf bimodal → mixed calibration!')
    axes[0].axvline(np.median(log_rcg), color='r', linestyle='--', label=f'Median={np.median(log_rcg):.2f}')
    axes[0].legend()
    
    # 右图：RCG vs WS 散点（检查是否 RCG 的两个模态对应不同的预测行为）
    if 'WS' in df.columns:
        ws = df['WS'].values
        valid = np.isfinite(ws) & np.isfinite(log_rcg[:len(ws)])
        if valid.sum() > 100:
            # 采样避免过多点
            n_plot = min(20000, valid.sum())
            idx = np.random.choice(np.where(valid)[0], n_plot, replace=False)
            axes[1].scatter(ws[idx], log_rcg[idx], s=0.5, alpha=0.3, c='steelblue')
            axes[1].set_xlabel('ERA5 Wind Speed (m/s)')
            axes[1].set_ylabel('log10(RCG)')
            axes[1].set_title('RCG vs Wind Speed\nVertical spread = calibration inconsistency')
    
    # 检查 rcg_source 列
    if 'rcg_source' in df.columns:
        sources = df['rcg_source'].value_counts()
        print(f"\n  RCG 来源分布:")
        for src, cnt in sources.items():
            print(f"    {src}: {cnt} ({100*cnt/len(df):.1f}%)")
    
    plt.tight_layout()
    save_path = out_dir / "rcg_diagnosis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存 RCG 诊断图: {save_path}")
    print("=" * 50 + "\n")

def _diagnose_high_wind_bimodality(
    y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path
) -> None:
    """
    高风速区域双峰诊断：
    在窄 bin 内检查预测值是否呈双峰分布
    如果是双峰 → 真正的数据口径问题
    如果是单峰偏移 → 只是回归收缩
    """
    print("\n📊 高风速双峰诊断...")
    
    bins_to_check = [
        (10.0, 10.5), (11.0, 11.5), (12.0, 12.5),
        (13.0, 13.5), (14.0, 14.5), (15.0, 15.5),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (lo, hi) in enumerate(bins_to_check):
        mask = (y_true >= lo) & (y_true < hi)
        preds_in_bin = y_pred[mask]
        
        if len(preds_in_bin) < 10:
            axes[i].set_title(f'WS=[{lo},{hi}) n={len(preds_in_bin)}\nToo few samples')
            continue
        
        axes[i].hist(preds_in_bin, bins=30, density=True, alpha=0.7, color='steelblue')
        axes[i].axvline(lo, color='g', linestyle='--', alpha=0.5, label='True range')
        axes[i].axvline(hi, color='g', linestyle='--', alpha=0.5)
        axes[i].axvline(np.mean(preds_in_bin), color='r', linestyle='-', 
                       label=f'Mean={np.mean(preds_in_bin):.2f}')
        axes[i].set_title(f'WS=[{lo},{hi}) n={len(preds_in_bin)}')
        axes[i].set_xlabel('Predicted WS (m/s)')
        axes[i].legend(fontsize=7)
        
        # 简单双峰检测：用 Hartigan's dip test 思路
        # 如果 std > 2.0 且分布偏离正态，可能是双峰
        std_pred = np.std(preds_in_bin)
        skew = float(np.mean(((preds_in_bin - np.mean(preds_in_bin)) / (std_pred + 1e-6)) ** 3))
        kurt = float(np.mean(((preds_in_bin - np.mean(preds_in_bin)) / (std_pred + 1e-6)) ** 4)) - 3
        
        if kurt < -1.0 and std_pred > 1.5:
            axes[i].set_facecolor('#fff0f0')  # 浅红背景：可能双峰
            print(f"  ⚠️ WS=[{lo},{hi}): std={std_pred:.2f}, kurtosis={kurt:.2f} → 可能双峰!")
        else:
            print(f"  ✅ WS=[{lo},{hi}): std={std_pred:.2f}, kurtosis={kurt:.2f} → 单峰")
    
    plt.suptitle('High Wind Speed Bimodality Diagnosis\n'
                 'Red bg = possible bimodal (mixed calibration)', fontsize=13)
    plt.tight_layout()
    save_path = out_dir / "high_wind_bimodality_diagnosis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存双峰诊断图: {save_path}\n")

# 模块 21：训练/验证/测试主流程（End-to-End Training Orchestration）
def run_training(cfg: TrainConfig) -> None:
    """
    主训练循环（含可视化）
    融合了 run_training 和 run_training_with_visualization 的功能
    """
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ 新增：GPU 性能优化
    setup_gpu_optimizations(deterministic=cfg.deterministic)
    
    # ✅ 新增：初始化 AMP 混合精度训练
    use_amp = cfg.device.startswith("cuda") and torch.cuda.is_available()
    if use_amp:
        try:
            scaler = torch.amp.GradScaler(device="cuda", enabled=True)
        except TypeError:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = None
    if use_amp:
        print("✅ 启用 AMP 混合精度训练 (FP16)")

    # ==================== 数据加载 ====================
    print(f"加载数据集: {cfg.dataset_pkl}")
    df: pd.DataFrame = pd.read_pickle(cfg.dataset_pkl)
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("数据集 pickle 必须包含 pandas DataFrame。")
    
    # 🔴 新增：RCG 口径诊断
    _diagnose_rcg_distribution(df, cfg.out_dir)

    # ==================== 特征构建 ====================
    aux_cols = build_aux_columns(df)
    print(f"辅助特征 ({len(aux_cols)}): {aux_cols}")

    # 设置外部 DDM 存储 (如果需要)
    ddm_store = DDMExternalStore(cfg.ddm_store) if cfg.ddm_store is not None else None
    ddm_index = DDMIndexResolver(df, ddm_store) if ddm_store is not None else None

    # ==================== 数据集切分 ====================
    train_idx, val_idx, test_idx = split_by_paper_time(df, seed=cfg.seed)
    print(f"切分大小 (观测点数): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 构建滑动窗口
        # 构建滑动窗口
    all_windows = make_windows_from_observations(df, window=cfg.time_steps)
    
    # 检查是否有足够的窗口
    if len(all_windows) == 0:
        raise RuntimeError(
            f"无法构建任何滑动窗口！请检查：\n"
            f"  1. 数据集是否包含 'final_seq_group' 和 'ddm_timestamp_utc' 列\n"
            f"  2. 每个序列组是否有至少 {cfg.time_steps} 个连续观测点\n"
            f"  3. 时空约束条件是否过于严格"
        )
    
    print(f"共构建 {len(all_windows)} 个滑动窗口")
    
    train_set = set(map(int, train_idx.values))
    val_set = set(map(int, val_idx.values))
    test_set = set(map(int, test_idx.values))

    train_windows: List[np.ndarray] = []
    val_windows: List[np.ndarray] = []
    test_windows: List[np.ndarray] = []

    # 由于切分是按序列组进行的，同一窗口的所有点必然属于同一集合
    # 因此使用 issubset 检查是安全的，不会丢失窗口
    for w in all_windows:
        w_set = set(map(int, w))
        if w_set.issubset(train_set):
            train_windows.append(w)
        elif w_set.issubset(val_set):
            val_windows.append(w)
        elif w_set.issubset(test_set):
            test_windows.append(w)
        else:
            # 理论上不应该发生，因为同一序列组的点都在同一集合中
            # 如果发生，说明窗口跨越了不同序列组（make_windows 函数有 bug）
            pass

    print(f"窗口分配: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")

    # ==================== 回退处理 ====================
    if len(train_windows) == 0 or len(val_windows) == 0 or len(test_windows) == 0:
        print("⚠️ 警告: 按序列组切分后某个集合窗口为空。回退到基于窗口的随机切分 (8:1:1)。")
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(all_windows))
        n = len(all_windows)
        n_train = max(1, int(round(0.8 * n)))
        n_val = max(1, int(round(0.1 * n)))
        
        train_windows = [all_windows[i] for i in perm[:n_train]]
        val_windows = [all_windows[i] for i in perm[n_train:n_train + n_val]]
        test_windows = [all_windows[i] for i in perm[n_train + n_val:]]
        
        print(f"随机切分后: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")

    # 最终检查
    if len(train_windows) == 0:
        raise RuntimeError("训练集窗口为空，无法训练！请检查数据集。")
    
    if len(val_windows) == 0:
        print("⚠️ 警告: 验证集为空；使用 10% 训练集作为验证集。")
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(train_windows))
        n_val = max(1, int(round(0.1 * len(train_windows))))
        val_windows = [train_windows[i] for i in perm[:n_val]]
        train_windows = [train_windows[i] for i in perm[n_val:]]

    if len(test_windows) == 0:
        print("⚠️ 警告: 测试集为空；从训练集划分 10%。")
        rng = np.random.default_rng(cfg.seed + 1)
        perm = rng.permutation(len(train_windows))
        n_test = max(1, int(round(0.1 * len(train_windows))))
        test_windows = [train_windows[i] for i in perm[:n_test]]
        train_windows = [train_windows[i] for i in perm[n_test:]]

    print(f"窗口数量: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")

    # ==================== 归一化统计量 ====================
    if cfg.norm_stats_path is not None:
        stats_path = cfg.norm_stats_path
    else:
        stats_path = cfg.out_dir / "norm_stats.npz"

    if stats_path.exists():
        stats = NormalizationStats.load(stats_path)
        print(f"✅ 加载归一化统计量: {stats_path}")
    else:
        print("⚠️ 未找到归一化统计量，重新计算...")
        stats = compute_normalization_stats(df, train_windows, aux_cols, ddm_store=ddm_store, ddm_index=ddm_index)
        save_path = cfg.out_dir / "norm_stats.npz"
        stats.save(save_path)
        print(f"已保存归一化统计量: {save_path}")

    # ==================== 构建 Dataset 和 DataLoader ====================
    print("构建数据集...")
    
    # 使用预加载模式（关键改动！）
    train_ds = STGWindowDataset(
        df, train_windows, aux_cols, stats, 
        ddm_store=ddm_store, ddm_index=ddm_index, 
        target_col="WS", 
        preload=True,  # 预加载到内存
        use_log_transform=True,
    )
    val_ds = STGWindowDataset(
        df, val_windows, aux_cols, stats,
        ddm_store=ddm_store, ddm_index=ddm_index,
        target_col="WS",
        preload=True,
        use_log_transform=True,
    )
    test_ds = STGWindowDataset(
        df, test_windows, aux_cols, stats,
        ddm_store=ddm_store, ddm_index=ddm_index,
        target_col="WS",
        preload=True,
        use_log_transform=True,
    )
    

    pin = cfg.pin_memory and cfg.device.startswith("cuda")
    
    # num_workers=0 因为数据已在内存，不需要多进程
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,   
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,  # ✅ 保持 worker 进程
        prefetch_factor=4 if cfg.num_workers > 0 else None,         # ✅ 预取更多 batch
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size * 2,    # ✅ 验证时可以用更大 batch（不需要梯度）
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.batch_size * 2,    # ✅ 测试时可以用更大 batch
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    # ==================== 初始化模型和优化器 ====================
    model = make_model(cfg, aux_dim=len(aux_cols)).to(cfg.device)
    
    # ✅ PyTorch 2.0+ 编译加速（与结构化剪枝互斥，避免图缓存与层结构变更冲突）
    if cfg.enable_replaceme_pruning:
        print("ℹ️ 跳过 torch.compile()：启用了结构化深度剪枝，避免编译图与剪枝后结构不一致。")
    elif hasattr(torch, 'compile') and cfg.device.startswith("cuda"):
        print("✅ 启用 torch.compile() 编译优化...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("   编译模式: reduce-overhead")
        except Exception as e:
            print(f"   ⚠️ 编译失败，使用原始模型: {e}")
    
    # 多 GPU 支持（如果你有多张卡）
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个 GPU，正在使用 DataParallel 模式...")
        model = nn.DataParallel(model)

    # 将模型移至主设备 (cuda:0)
    model = model.to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 构建学习率调度器
    steps_per_epoch = len(train_loader)
    scheduler, scheduler_type = build_scheduler(cfg, optim, steps_per_epoch)

    # 训练集全局目标标准差（用于稳定方差约束，减少 mini-batch 噪声）

    best_val = float("inf")
    best_path = cfg.out_dir / f"best_{cfg.model}.pt"
    bad = 0

    # 记录训练历史（用于绘制学习曲线）
    history = {
        'train_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'test_rmse': [],
        'lr': []
    }

    print("=" * 90)
    print(f"开始训练 {cfg.model.upper()} @ {cfg.device}")
    print(f"AdamW lr={cfg.lr:g}, batch={cfg.batch_size}, patience={cfg.patience}, max_epochs={cfg.max_epochs}")
    print(f"学习率调度: {cfg.scheduler}")
    print("=" * 90)

    # ==================== 训练循环 ====================
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        # ✅ 传入 scaler 参数
        train_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            cfg.device,
            scaler=scaler,
        )

        # 评估
        train_met = evaluate(model, train_loader, cfg.device)
        val_met = evaluate(model, val_loader, cfg.device)
        test_met = evaluate(model, test_loader, cfg.device)

        # 获取当前学习率
        current_lr = optim.param_groups[0]["lr"]

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_rmse'].append(train_met['rmse'])
        history['val_rmse'].append(val_met['rmse'])
        history['test_rmse'].append(test_met['rmse'])
        history['lr'].append(current_lr)

        dt_s = time.time() - t0
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.6f} | lr={current_lr:.2e} | "
            f"RMSE train/val/test={train_met['rmse']:.4f}/{val_met['rmse']:.4f}/{test_met['rmse']:.4f} | "
            f"Bias val={val_met['bias']:+.4f} | Corr val={val_met['corr']:.4f} | {dt_s:.1f}s"
        )

        # 更新学习率调度器
        if scheduler is not None and scheduler_type != "step":
            if scheduler_type == "epoch":
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(val_met["rmse"])

        # 早停检查
        if val_met["rmse"] + 1e-9 < best_val:
            best_val = val_met["rmse"]
            bad = 0
            if cfg.save_best:
                torch.save({
                    "model": model.state_dict(),
                    "cfg": dataclasses.asdict(cfg),
                    "epoch": epoch,
                    "best_val_rmse": best_val,
                }, best_path)
        else:
            bad += 1
            if cfg.patience > 0 and bad >= cfg.patience:
                print(f"早停机制触发于 epoch {epoch} (最佳验证集 RMSE={best_val:.4f}).")
                break

    # ==================== 加载最佳模型 ====================
    if cfg.save_best and best_path.exists():
        ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint: {best_path}")

    # ==================== ReplaceMe-style depth pruning (post-training, no healing) ====================
    if cfg.enable_replaceme_pruning:
        prune_target = model
        if isinstance(prune_target, nn.DataParallel):
            prune_target = prune_target.module
        if hasattr(prune_target, "_orig_mod"):
            prune_target = prune_target._orig_mod

        if isinstance(prune_target, STGDNN):
            prune_info = apply_replaceme_depth_pruning(
                prune_target,
                train_loader,
                cfg.device,
                prune_layers=cfg.replaceme_prune_layers,
                calibration_batches=cfg.replaceme_calibration_batches,
                objective=cfg.replaceme_objective,
                cosine_steps=cfg.replaceme_cosine_steps,
                cosine_lr=cfg.replaceme_cosine_lr,
                cosine_batch_size=cfg.replaceme_cosine_batch_size,
                cosine_l2_reg=cfg.replaceme_cosine_l2_reg,
                cosine_early_stop_patience=cfg.replaceme_cosine_early_stop_patience,
                cosine_early_stop_min_delta=cfg.replaceme_cosine_early_stop_min_delta,
                cosine_eval_interval=cfg.replaceme_cosine_eval_interval,
            )
            print("ReplaceMe depth pruning finished (block-level, not global weight pruning):")
            print(
                f"  method={prune_info['method']} | objective={prune_info['objective']} | "
                f"cut={prune_info['cut_index']} | "
                f"pruned={prune_info['pruned_indices']} | "
                f"layers {prune_info['layers_before']} -> {prune_info['layers_after']} | "
                f"cos_dist={prune_info['selected_cosine_distance']:.6f} | "
                f"opt_loss={prune_info['optimization_loss']:.6f} | "
                f"calib_batches={prune_info['calibration_batches']}"
            )
            print(
                f"  calibration RMSE: {prune_info['calibration_rmse_before']:.6f} -> "
                f"{prune_info['calibration_rmse_after']:.6f} "
                f"(delta={prune_info['calibration_rmse_delta']:+.6f})"
            )
            if prune_info["objective"] == "cosine":
                print(
                    f"  cosine early-stop: triggered={prune_info['early_stopped']} | "
                    f"best_step={prune_info['best_step']} | steps_run={prune_info['effective_steps']} | "
                    f"patience={prune_info['cosine_early_stop_patience']} | "
                    f"min_delta={prune_info['cosine_early_stop_min_delta']:.1e}"
                )
        else:
            print("Skip ReplaceMe pruning: model is not STGDNN.")

    # ==================== 最终评估 ====================
    final_val = evaluate(model, val_loader, cfg.device)
    final_test = evaluate(model, test_loader, cfg.device)
    
    print("=" * 90)
    print(f"最终结果 | val : rmse={final_val['rmse']:.4f}, bias={final_val['bias']:+.4f}, corr={final_val['corr']:.4f}")
    print(f"最终结果 | test: rmse={final_test['rmse']:.4f}, bias={final_test['bias']:+.4f}, corr={final_test['corr']:.4f}")
    print("=" * 90)

    # ==================== 收集预测结果 ====================
    model.eval()
    
    # 测试集预测
    test_ys: List[np.ndarray] = []
    test_yh: List[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            ddm = batch["ddm"].to(cfg.device, non_blocking=True)
            aux = batch["aux"].to(cfg.device, non_blocking=True)
            y = batch["y"]
            sp_idx = batch["sp_idx"].to(cfg.device, non_blocking=True)  # 🔴 新增：加载SP索引
            pred = model(ddm, aux, sp_idx)  # 🔴 传递SP索引
            test_ys.append(y.cpu().numpy())      # (B,) 直接使用
            test_yh.append(pred.cpu().numpy())  # (B,)
    
    y_true_test = np.concatenate(test_ys, axis=0)
    y_pred_test = np.concatenate(test_yh, axis=0)
    raw_test_window_preds = y_pred_test

    # 验证集预测
    val_ys: List[np.ndarray] = []
    val_yh: List[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            ddm = batch["ddm"].to(cfg.device, non_blocking=True)
            aux = batch["aux"].to(cfg.device, non_blocking=True)
            y = batch["y"]
            sp_idx = batch["sp_idx"].to(cfg.device, non_blocking=True)
            pred = model(ddm, aux, sp_idx)
            val_ys.append(y.cpu().numpy())
            val_yh.append(pred.cpu().numpy())

    y_true_val = np.concatenate(val_ys, axis=0)
    y_pred_val = np.concatenate(val_yh, axis=0)
    raw_val_window_preds = y_pred_val

    
    test_obs_indices, y_true_test, y_pred_test, test_vote_counts = aggregate_window_predictions_to_observations(
        test_windows, y_pred_test, df, target_col="WS"
    )
    val_obs_indices, y_true_val, y_pred_val, val_vote_counts = aggregate_window_predictions_to_observations(
        val_windows, y_pred_val, df, target_col="WS"
    )

    assert_no_window_striping_artifacts(
        raw_window_preds=raw_test_window_preds,
        obs_indices=test_obs_indices,
        y_true_obs=y_true_test,
        y_pred_obs=y_pred_test,
        vote_counts=test_vote_counts,
        time_steps=cfg.time_steps,
        set_name="Test",
    )
    assert_no_window_striping_artifacts(
        raw_window_preds=raw_val_window_preds,
        obs_indices=val_obs_indices,
        y_true_obs=y_true_val,
        y_pred_obs=y_pred_val,
        vote_counts=val_vote_counts,
        time_steps=cfg.time_steps,
        set_name="Validation",
    )

    final_val_obs = compute_metrics(y_true_val, y_pred_val)
    final_test_obs = compute_metrics(y_true_test, y_pred_test)
    print(
        f"观测点级聚合后: val : rmse={final_val_obs['rmse']:.4f}, "
        f"bias={final_val_obs['bias']:+.4f}, corr={final_val_obs['corr']:.4f}"
    )
    print(
        f"观测点级聚合后: test: rmse={final_test_obs['rmse']:.4f}, "
        f"bias={final_test_obs['bias']:+.4f}, corr={final_test_obs['corr']:.4f}"
    )
    print(
        f"test 聚合: 窗口{len(test_windows)} -> 观测点 {len(test_obs_indices)}, "
        f"平均投票数{float(np.mean(test_vote_counts)):.2f}"
    )
    print(
        f"val  聚合: 窗口 {len(val_windows)} -> 观测点 {len(val_obs_indices)}, "
        f"平均投票数{float(np.mean(val_vote_counts)):.2f}"
    )

    # ==================== 保存预测结果和指标 ====================
    # 分风速区间指标
    intervals = compute_interval_metrics(y_true_test, y_pred_test)
    out_csv = cfg.out_dir / f"interval_metrics_{cfg.model}.csv"
    pd.DataFrame(intervals).to_csv(out_csv, index=False)
    print(f"已保存分区间指标: {out_csv}")

    # 预测值
    # ========== 修改开始 ==========
    # 使用观测点级聚合后的唯一索引（跨窗口平均后）
    test_indices = test_obs_indices

    # 预测值
    out_npz = cfg.out_dir / f"preds_{cfg.model}.npz"
    np.savez(out_npz, 
            y_true=y_true_test, 
            y_pred=y_pred_test,
            test_indices=test_indices,
            test_vote_counts=test_vote_counts)
    print(f"已保存测试集预测（含对齐索引）: {out_npz}")
# ========== 修改结束 ==========

    # 训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(cfg.out_dir / "training_history.csv", index=False)
    print(f"已保存训练历史: {cfg.out_dir / 'training_history.csv'}")

    # ==================== 可视化 ====================
    visualizer = WindSpeedVisualizer(save_dir=cfg.out_dir, model_name=cfg.model.upper())

    # 1. 学习曲线
    if len(history['train_loss']) > 1:
        visualizer.plot_learning_curves(
            train_losses=history['train_loss'],
            val_rmse=history['val_rmse'],
            train_rmse=history['train_rmse']
        )

    # 2. 测试集图表（KDE 散点图 + 误差分布 + 分区间指标）
    visualizer.generate_all_figures(
        y_true=y_true_test,
        y_pred=y_pred_test,
        set_name="Test",
        interval_metrics=intervals
    )

    # 3. 验证集 KDE 散点图
    visualizer.plot_scatter_density_kde(y_true_val, y_pred_val, "Validation")
    # 🔴 新增：高风速双峰诊断图
    _diagnose_high_wind_bimodality(y_true_test, y_pred_test, cfg.out_dir)

    print(f"\n{'='*60}")
    print(f"✅ 训练完成！所有结果已保存至: {cfg.out_dir}")
    print(f"生成的图表:")
    print(f"  - scatter_density_kde_test.png (KDE 着色散点图 - 主图)")
    print(f"  - scatter_density_kde_validation.png (验证集 KDE 散点图)")
    print(f"  - error_distribution_test.png (误差分布)")
    print(f"  - interval_rmse.png (分区间 RMSE)")
    print(f"  - learning_curves.png (学习曲线)")
    print(f"{'='*60}")

# 模块 22：程序入口与运行参数固化（Entrypoint / Runner）
def main() -> None:
    """程序入口 - RTX 5090 优化版"""
    
    cfg = TrainConfig(
        # ============ 数据路径（保持不变）============
        dataset_pkl=Path(r"E:\1\stg_dnn_dataset_15days.pkl"),
        out_dir=Path(r"E:\1\output2"),
        
        # ============ 性能优化参数（保持不变）============
        batch_size=512,
        num_workers=8,
        pin_memory=True,
        
        # ============ 模型参数 ============
        model="stg_dnn",
        time_steps=10,
        # ✅ 修改：将 CNN 参数替换为 GCN 参数
        # 原代码：
        #   （无 k, gcn_layers 等参数，使用 CNN 默认值）
        # 新代码：
        k=10,                  # KNN 邻居数
        gcn_layers=3,          # 图卷积层数
        gcn_hidden=64,         # GCN 隐藏维度
        gcn_out=128,           # GCN 输出维度
        gcn_phi_hidden=32,     # 边函数隐藏维度
        gcn_heads=4,           # 多头数量
        # Transformer 参数（保持不变）
        d_model=64,
        nhead=8,
        transformer_layers=8,
        ff_dim=256,
        dropout=0.15,
        
        # ============ 训练参数（保持不变）============
        max_epochs=200,
        patience=30,
        lr=3e-4,
        weight_decay=1e-4,
        
        # ============ 学习率调度（保持不变）============
        scheduler="cosine_warmup",
        warmup_epochs=10,
        min_lr=1e-6,

        # ============ ReplaceMe 深度剪枝（更贴近论文主推 cosine 版本） ============
        enable_replaceme_pruning=True,
        replaceme_prune_layers=2,
        replaceme_calibration_batches=8,
        replaceme_objective="cosine",
        replaceme_cosine_steps=300,
        replaceme_cosine_lr=1e-2,
        replaceme_cosine_batch_size=4096,
        replaceme_cosine_l2_reg=0.0,
        replaceme_cosine_early_stop_patience=20,
        replaceme_cosine_early_stop_min_delta=1e-5,
        replaceme_cosine_eval_interval=10,
    )
    
    print("=" * 60)
    print("RTX 5090 优化配置")
    print("=" * 60)
    print(f"  数据集: {cfg.dataset_pkl}")
    print(f"  输出目录: {cfg.out_dir}")
    print(f"  设备: {cfg.device}")
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  Num Workers: {cfg.num_workers}")
    print(f"  学习率: {cfg.lr}")
    print(f"  调度器: {cfg.scheduler}")
    print(f"  早停耐心: {cfg.patience}")
    # ✅ 新增：打印 GCN 参数
    print(f"  GCN: k={cfg.k}, layers={cfg.gcn_layers}, hidden={cfg.gcn_hidden}, out={cfg.gcn_out}")
    print("=" * 60)
    
    run_training(cfg)

if __name__ == "__main__":
    main()
