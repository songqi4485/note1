from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde


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


class WindSpeedPlotter:
    def __init__(self, model_name: str = "Model", dpi: int = 600) -> None:
        self.model_name = model_name
        self.dpi = int(dpi)
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 12

    def plot_learning_curves(
        self,
        train_losses: List[float],
        val_rmse: List[float],
        train_rmse: List[float],
        out_path: Path,
    ) -> None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=self.colors[0])
        ax1.plot(train_losses, label="Train Loss", color=self.colors[0])
        ax1.tick_params(axis="y", labelcolor=self.colors[0])
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("RMSE (m/s)", color=self.colors[1])
        ax2.plot(train_rmse, label="Train RMSE", color=self.colors[1], linestyle="--")
        ax2.plot(val_rmse, label="Val RMSE", color=self.colors[2])
        ax2.tick_params(axis="y", labelcolor=self.colors[1])

        fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
        plt.title(f"{self.model_name} Learning Curves")
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_scatter_density(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        out_path: Path,
        set_name: str = "Test",
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        x = y_true[mask]
        y = y_pred[mask]

        if len(x) < 10:
            print(f"[WARN] {set_name}: too few samples ({len(x)}), skip scatter density")
            plt.close()
            return

        rmse = np.sqrt(np.mean((y - x) ** 2))
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        bias = np.mean(y - x)

        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 1.0, 0.0

        # KDE on too many points can be slow; downsample only for visualization.
        max_points_for_kde = 80000
        if len(x) > max_points_for_kde:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x), size=max_points_for_kde, replace=False)
            x_plot = x[idx]
            y_plot = y[idx]
            print(f"[INFO] {set_name}: KDE scatter uses {max_points_for_kde} sampled points")
        else:
            x_plot = x
            y_plot = y

        xy = np.vstack([x_plot, y_plot])
        try:
            density = gaussian_kde(xy, bw_method="scott")(xy)
        except np.linalg.LinAlgError:
            # Fallback for near-singular covariance in KDE.
            jitter = np.random.default_rng(42).normal(0.0, 1e-3, size=xy.shape)
            density = gaussian_kde(xy + jitter, bw_method="scott")(xy + jitter)

        positive = density[density > 0]
        if positive.size == 0:
            print(f"[WARN] {set_name}: empty KDE density, skip scatter density")
            plt.close()
            return

        order = np.argsort(density)
        x_plot = x_plot[order]
        y_plot = y_plot[order]
        density = density[order]

        vmin = float(np.percentile(positive, 2.0))
        vmax = float(np.percentile(positive, 99.8))
        vmin = max(vmin, 1e-12)
        if vmax <= vmin:
            vmax = vmin * 10.0

        min_val = 0.0
        max_val = float(max(x.max(), y.max()) + 1.0)

        mesh = ax.scatter(
            x_plot,
            y_plot,
            c=density,
            s=2.5,
            cmap="jet",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            linewidths=0.0,
            alpha=0.9,
            rasterized=True,
            zorder=1,
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("KDE Density (log scale)", fontsize=12)

        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="1:1 Line")
        fit_x = np.linspace(min_val, max_val, 100)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, "r-", lw=1.5, label=f"Fit: y={slope:.2f}x+{intercept:.2f}")

        ax.set_xlabel("ERA5 Wind Speed (m/s)", fontsize=14)
        ax.set_ylabel("Predicted Wind Speed (m/s)", fontsize=14)
        ax.set_title(
            f"Scatter Density Plot\nRMSE={rmse:.3f}, R={corr:.3f}, Bias={bias:.3f}",
            fontsize=14,
        )
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {out_path}")

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        out_path: Path,
        set_name: str = "Test",
    ) -> None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        error = y_pred[mask] - y_true[mask]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(error, bins=80, range=(-15, 15), density=True, color="steelblue", alpha=0.8)
        ax.axvline(0, color="r", linestyle="--", lw=1.5)
        ax.axvline(np.mean(error), color="orange", linestyle="-", lw=1.5, label=f"Mean={np.mean(error):.3f}")
        ax.set_xlabel("Prediction Error (m/s)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title(f"{set_name} Error Distribution", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {out_path}")

    def plot_interval_metrics(self, interval_metrics: List[Dict[str, float]], out_path: Path) -> None:
        labels = [f"{d['lo']}-{d['hi'] if d['hi'] != float('inf') else 'inf'}" for d in interval_metrics]
        rmses = [d["rmse"] for d in interval_metrics]
        ns = [d["n"] for d in interval_metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, rmses, color="steelblue", alpha=0.8)
        ax.set_xlabel("Wind Speed Range (m/s)", fontsize=12)
        ax.set_ylabel("RMSE (m/s)", fontsize=12)
        ax.set_title("RMSE by Wind Speed Interval", fontsize=14)

        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {out_path}")

    def generate_all_figures(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scatter_path: Path,
        error_path: Path,
        interval_path: Optional[Path],
        set_name: str,
        interval_metrics: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        self.plot_scatter_density(y_true, y_pred, out_path=scatter_path, set_name=set_name)
        self.plot_error_distribution(y_true, y_pred, out_path=error_path, set_name=set_name)
        if interval_metrics is not None and interval_path is not None:
            self.plot_interval_metrics(interval_metrics, out_path=interval_path)


def _load_preds(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if "y_true" not in data.files or "y_pred" not in data.files:
        raise KeyError(f"{npz_path} missing 'y_true' or 'y_pred'. keys={list(data.files)}")
    y_true = np.asarray(data["y_true"], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(data["y_pred"], dtype=np.float64).reshape(-1)
    return y_true, y_pred


def _try_plot_history(history_csv: Optional[Path], plotter: WindSpeedPlotter, out_path: Path) -> None:
    if history_csv is None:
        return
    if not history_csv.exists():
        print(f"[WARN] history csv not found: {history_csv}, skip learning curves")
        return

    df = pd.read_csv(history_csv)
    required = {"train_loss", "val_rmse", "train_rmse"}
    if not required.issubset(df.columns):
        print(f"[WARN] history csv missing columns {required}, skip learning curves")
        return

    if len(df) <= 1:
        print("[WARN] history csv has <=1 row, skip learning curves")
        return

    plotter.plot_learning_curves(
        train_losses=df["train_loss"].tolist(),
        val_rmse=df["val_rmse"].tolist(),
        train_rmse=df["train_rmse"].tolist(),
        out_path=out_path,
    )
    print(f"[OK] saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone plotting script from preds_*.npz")
    parser.add_argument(
        "--preds",
        type=str,
        default=None,
        help="Path to preds_*.npz (optional; if omitted, use path configured in main)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (optional; if omitted, use path configured in main)",
    )
    parser.add_argument("--set-name", type=str, default="Test", help="Label used in titles and file names")
    parser.add_argument("--model-name", type=str, default="STG_DNN", help="Model name used in learning curves title")
    parser.add_argument("--history-csv", type=str, default=None, help="Optional training_history.csv path")
    parser.add_argument(
        "--interval-csv-name",
        type=str,
        default="interval_metrics_from_npz.csv",
        help="File name for interval metrics csv",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output figure dpi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ===== User-editable path configuration (edit these values in one place) =====
    # CLI args still work and take precedence over this block.
    script_dir = Path(__file__).resolve().parent
    default_paths = {
        "preds_npz": script_dir / "preds_test.npz",
        "out_dir": script_dir / "figures_from_npz",
        "history_csv": None,  # Example: script_dir / "training_history.csv"
    }
    output_names = {
        "interval_csv": args.interval_csv_name,
        "scatter_png": f"scatter_density_kde_{args.set_name.lower()}.png",
        "error_png": f"error_distribution_{args.set_name.lower()}.png",
        "interval_png": "interval_rmse.png",
        "learning_curve_png": "learning_curves.png",
    }

    preds_src = args.preds if args.preds is not None else default_paths["preds_npz"]
    out_dir_src = args.out_dir if args.out_dir is not None else default_paths["out_dir"]
    history_src = args.history_csv if args.history_csv is not None else default_paths["history_csv"]

    npz_path = Path(preds_src).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(
            f"preds file not found: {npz_path}\n"
            "Pass --preds, or edit default_paths['preds_npz'] in main()."
        )

    out_dir = Path(out_dir_src).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history_csv = Path(history_src).expanduser().resolve() if history_src is not None else None

    # All save paths are explicitly constructed in main for easy future edits.
    interval_csv = (out_dir / output_names["interval_csv"]).resolve()
    scatter_png = (out_dir / output_names["scatter_png"]).resolve()
    error_png = (out_dir / output_names["error_png"]).resolve()
    interval_png = (out_dir / output_names["interval_png"]).resolve()
    learning_curve_png = (out_dir / output_names["learning_curve_png"]).resolve()

    y_true, y_pred = _load_preds(npz_path)
    metrics = compute_metrics(y_true, y_pred)
    intervals = compute_interval_metrics(y_true, y_pred)

    pd.DataFrame(intervals).to_csv(interval_csv, index=False)

    print(f"[INFO] input npz: {npz_path}")
    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] scatter png: {scatter_png}")
    print(f"[INFO] error png: {error_png}")
    print(f"[INFO] interval png: {interval_png}")
    print(f"[INFO] interval csv: {interval_csv}")
    if history_csv is not None:
        print(f"[INFO] history csv: {history_csv}")
        print(f"[INFO] learning curve png: {learning_curve_png}")
    print(
        f"[INFO] metrics: RMSE={metrics['rmse']:.4f}, "
        f"Bias={metrics['bias']:+.4f}, Corr={metrics['corr']:.4f}"
    )
    print(f"[OK] saved: {interval_csv}")

    plotter = WindSpeedPlotter(model_name=args.model_name, dpi=args.dpi)
    plotter.generate_all_figures(
        y_true=y_true,
        y_pred=y_pred,
        scatter_path=scatter_png,
        error_path=error_png,
        interval_path=interval_png,
        set_name=args.set_name,
        interval_metrics=intervals,
    )
    _try_plot_history(history_csv, plotter, out_path=learning_curve_png)

    print("[DONE] plotting finished")


if __name__ == "__main__":
    main()
