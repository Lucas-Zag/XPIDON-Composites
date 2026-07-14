#!/usr/bin/env python3
"""
Simple Colab entry point for XPIDON.

Features
--------
- Cleans old local training outputs before a new run.
- Uses the current training settings.
- Saves models, loss files, code/configuration and environment information
  to Google Drive every 30 minutes.
- Runs plot_training_results.py only after training finishes.
- Performs one final save after completion, interruption, or Python error.

Before running in Colab
-----------------------
from google.colab import drive
drive.mount("/content/drive")

Then:
%cd /content/XPIDON-Composites
!python -u main_colab_30min.py
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import runpy
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from utils.phy_params import PhyParams
from utils.exp_params import ExpParams
from utils.train_params import TrainParams
from utils.air_temp import Temp_air
from utils.data_utils import DataGenerator, generate_training_data
from trainer.train import train
from models.pidon import XPIDON
from loss.loss import XPIDONLoss


PROJECT_ROOT = Path(__file__).resolve().parent
DRIVE_ROOT = Path("/content/drive/MyDrive/XPIDON_Colab")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = DRIVE_ROOT / "runs" / RUN_ID

SAVE_INTERVAL_SECONDS = 30 * 60
STOP_EVENT = threading.Event()


# Current settings used in your recent run.
INIT_SUB_DOMAIN = 5
TOLERANCE_LEVEL = 5e-4
M_INP = 9

BRANCH_LAYERS = [M_INP, 50, 50, 50, 100]
TRUNK_LAYERS = [2, 50, 50, 50, 50, 50, 100]
NOMAD_LAYERS_T = [100, 50, 50, 50, 50, 1]


def run_git_command(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return f"Unavailable: {exc}"


def readable_pickle(path: Path) -> bool:
    """Avoid copying a pickle while train.py is still writing it."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False

        size_before = path.stat().st_size
        modified_before = path.stat().st_mtime_ns

        with path.open("rb") as file:
            pickle.load(file)

        return (
            path.stat().st_size == size_before
            and path.stat().st_mtime_ns == modified_before
        )
    except Exception:
        return False


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def copy_file_if_ready(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False

    if source.suffix == ".pkl" and not readable_pickle(source):
        return False

    if destination.exists():
        same_size = destination.stat().st_size == source.stat().st_size
        destination_newer = destination.stat().st_mtime >= source.stat().st_mtime
        if same_size and destination_newer:
            return False

    atomic_copy(source, destination)
    return True


def clean_old_local_outputs() -> None:
    """Clean the current /content run so old model tags cannot mix with new ones."""

    folders = [
        PROJECT_ROOT / "outputs" / "models",
        PROJECT_ROOT / "outputs" / "loss_data",
        PROJECT_ROOT / "outputs" / "figures",
        PROJECT_ROOT / "outputs" / "predictions",
    ]

    root_patterns = [
        "T_temp.pkl",
        "a_temp.pkl",
        "xpidon_class_*_T.pkl",
        "xpidon_class_*_a.pkl",
    ]

    for folder in folders:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    for pattern in root_patterns:
        for file in PROJECT_ROOT.glob(pattern):
            if file.is_file():
                file.unlink()

    print("Old local training outputs cleaned.")


def save_configuration() -> None:
    """Save the exact code/configuration used by this run."""

    config_dir = RUN_DIR / "configuration"
    config_dir.mkdir(parents=True, exist_ok=True)

    important_files = [
        PROJECT_ROOT / Path(__file__).name,
        PROJECT_ROOT / "trainer" / "train.py",
        PROJECT_ROOT / "models" / "pidon.py",
        PROJECT_ROOT / "loss" / "loss.py",
        PROJECT_ROOT / "utils" / "data_utils.py",
        PROJECT_ROOT / "phy_params.json",
        PROJECT_ROOT / "exp_params.json",
        PROJECT_ROOT / "train_params.json",
        PROJECT_ROOT / "plot_training_results.py",
        PROJECT_ROOT / "training_book.ipynb",
    ]

    for source in important_files:
        if source.is_file():
            relative_name = str(source.relative_to(PROJECT_ROOT)).replace("/", "__")
            shutil.copy2(source, config_dir / relative_name)

    environment = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": run_git_command("rev-parse", "HEAD"),
        "git_branch": run_git_command("branch", "--show-current"),
        "git_status": run_git_command("status", "--short"),
        "training_settings": {
            "init_sub_domain": INIT_SUB_DOMAIN,
            "tolerance_level": TOLERANCE_LEVEL,
            "m_inp": M_INP,
            "branch_layers": BRANCH_LAYERS,
            "trunk_layers": TRUNK_LAYERS,
            "nomad_layers_T": NOMAD_LAYERS_T,
            "periodic_save_minutes": SAVE_INTERVAL_SECONDS // 60,
        },
    }

    try:
        import jax
        import jaxlib
        import numpy
        import optax

        environment.update(
            {
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "numpy": numpy.__version__,
                "optax": optax.__version__,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    except Exception as exc:
        environment["environment_error"] = repr(exc)

    (config_dir / "run_environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sync_training_files(include_final_outputs: bool = False) -> int:
    """
    Periodic save:
      - root temporary checkpoints
      - accepted subdomain models
      - per-subdomain loss histories

    Final save additionally includes:
      - figures
      - predictions
    """

    mappings = [
        (
            PROJECT_ROOT,
            RUN_DIR / "checkpoints",
            ["T_temp.pkl", "a_temp.pkl", "xpidon_class_*.pkl"],
        ),
        (
            PROJECT_ROOT / "outputs" / "models",
            RUN_DIR / "outputs" / "models",
            ["*.pkl"],
        ),
        (
            PROJECT_ROOT / "outputs" / "loss_data",
            RUN_DIR / "outputs" / "loss_data",
            ["*.pkl"],
        ),
    ]

    if include_final_outputs:
        mappings.extend(
            [
                (
                    PROJECT_ROOT / "outputs" / "figures",
                    RUN_DIR / "outputs" / "figures",
                    ["*"],
                ),
                (
                    PROJECT_ROOT / "outputs" / "predictions",
                    RUN_DIR / "outputs" / "predictions",
                    ["*"],
                ),
            ]
        )

    copied = 0

    for source_dir, destination_dir, patterns in mappings:
        if not source_dir.exists():
            continue

        for pattern in patterns:
            for source in source_dir.glob(pattern):
                if not source.is_file():
                    continue

                destination = destination_dir / source.name

                try:
                    if copy_file_if_ready(source, destination):
                        copied += 1
                except OSError as exc:
                    print(f"[Save warning] Could not copy {source}: {exc}")

    manifest = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "include_final_outputs": include_final_outputs,
        "models": sorted(
            file.name
            for file in (RUN_DIR / "outputs" / "models").glob("*.pkl")
        ),
        "loss_files": sorted(
            file.name
            for file in (RUN_DIR / "outputs" / "loss_data").glob("*.pkl")
        ),
        "temporary_checkpoints": sorted(
            file.name
            for file in (RUN_DIR / "checkpoints").glob("*.pkl")
        ),
    }

    (RUN_DIR / "latest_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return copied


def periodic_save_worker() -> None:
    while not STOP_EVENT.wait(SAVE_INTERVAL_SECONDS):
        copied = sync_training_files(include_final_outputs=False)
        print(
            f"\n[30-minute save] {datetime.now():%Y-%m-%d %H:%M:%S}: "
            f"{copied} updated file(s) copied to Google Drive."
        )


def final_model_is_complete() -> bool:
    model_dir = PROJECT_ROOT / "outputs" / "models"
    return (
        (model_dir / "xpidon_class_10000_T.pkl").exists()
        and (model_dir / "xpidon_class_10000_a.pkl").exists()
    )


def write_run_summary(
    status: str,
    started_at: datetime,
    error: str | None = None,
) -> None:
    ended_at = datetime.now()

    model_dir = RUN_DIR / "outputs" / "models"
    loss_dir = RUN_DIR / "outputs" / "loss_data"

    summary = {
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "run_directory": str(RUN_DIR),
        "full_domain_model_saved": (
            (model_dir / "xpidon_class_10000_T.pkl").exists()
            and (model_dir / "xpidon_class_10000_a.pkl").exists()
        ),
        "model_files": sorted(file.name for file in model_dir.glob("*.pkl")),
        "loss_files": sorted(file.name for file in loss_dir.glob("*.pkl")),
        "error": error,
    }

    (RUN_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError(
            "Google Drive is not mounted.\n"
            "Run this first in Colab:\n"
            "from google.colab import drive\n"
            'drive.mount("/content/drive")'
        )

    os.chdir(PROJECT_ROOT)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

    (DRIVE_ROOT / "latest_run.txt").write_text(
        str(RUN_DIR),
        encoding="utf-8",
    )

    clean_old_local_outputs()
    save_configuration()

    params = PhyParams("phy_params.json")
    exp_params = ExpParams("exp_params.json")
    train_params = TrainParams("train_params.json")

    print("\nGoogle Drive run directory:")
    print(RUN_DIR)

    print("\nTraining settings:")
    print(" init_sub_domain =", INIT_SUB_DOMAIN)
    print(" tolerance_level =", TOLERANCE_LEVEL)
    print(" branch_layers   =", BRANCH_LAYERS)
    print(" trunk_layers    =", TRUNK_LAYERS)
    print(" nomad_layers_T  =", NOMAD_LAYERS_T)
    print(" save interval   = 30 minutes")

    started_at = datetime.now()
    status = "running"
    error_text = None

    save_thread = threading.Thread(
        target=periodic_save_worker,
        daemon=True,
    )
    save_thread.start()

    try:
        train(
            XPIDON,
            XPIDONLoss,
            generate_training_data,
            DataGenerator,
            Temp_air,
            params,
            exp_params,
            train_params,
            INIT_SUB_DOMAIN,
            TOLERANCE_LEVEL,
            M_INP,
            BRANCH_LAYERS,
            TRUNK_LAYERS,
            NOMAD_LAYERS_T,
        )

        status = "training_completed"

        # Figures are generated only after training.
        plot_script = PROJECT_ROOT / "plot_training_results.py"

        if final_model_is_complete() and plot_script.exists():
            print("\nFull-domain model found. Generating final figures...")
            runpy.run_path(str(plot_script), run_name="__main__")
            status = "completed"
        elif not final_model_is_complete():
            print(
                "\nThe final 10000 model pair was not found. "
                "Plotting was skipped, but available models will be saved."
            )
            status = "completed_incomplete_domain"
        else:
            print(
                "\nplot_training_results.py was not found. "
                "Models and loss data will still be saved."
            )
            status = "completed_without_plots"

    except KeyboardInterrupt:
        status = "interrupted"
        error_text = "Training interrupted by user."
        print("\nTraining interrupted. Saving current checkpoints...")

    except Exception:
        status = "failed"
        error_text = traceback.format_exc()
        print("\nTraining failed. Saving current checkpoints...")
        traceback.print_exc()

    finally:
        STOP_EVENT.set()
        save_thread.join(timeout=5)

        copied = sync_training_files(include_final_outputs=True)
        write_run_summary(status, started_at, error_text)

        print(f"\n[Final save] {copied} updated file(s) copied.")
        print("Saved to:")
        print(RUN_DIR)


if __name__ == "__main__":
    main()
