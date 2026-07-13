#!/usr/bin/env python3
"""
Colab T4 launcher and persistent saver for XPIDON-Composites.

What this script does
---------------------
1. Mounts Google Drive.
2. Restores XPIDON-Composites into /content from:
  a) an existing /content project,
  b) Google Drive project_backup, or
  c) a GitHub repository.
3. Runs XPIDON training with the parameters currently used in training_book.ipynb:
      init_sub_domain = 3
      Tolerance_level = 2e-3
      m_inp = 9
      branch_layers = [9, 50, 50, 50, 100]
      trunk_layers = [2, 50, 50, 50, 50, 50, 100]
      nomad_layers_T = [100, 50, 50, 50, 50, 1]
4. Saves models, temporary checkpoints, losses, figures, predictions,
  configuration files, environment information, and logs under:
      /content/drive/MyDrive/XPIDON_Colab/runs/<timestamp>/
5. Periodically copies only readable/complete pickle checkpoints to Drive,
  reducing loss if Colab disconnects.

Recommended Colab command
--------------------------
Upload this file to /content, then run one of:

A. Restore code from Drive backup:
  %run /content/colab_t4_train_and_save.py --source drive

B. Clone your GitHub branch:
  %run /content/colab_t4_train_and_save.py \
      --source github \
      --repo-url https://github.com/USERNAME/XPIDON-Composites.git \
      --branch YOUR_BRANCH

C. Use a project that already exists in /content:
  %run /content/colab_t4_train_and_save.py --source existing

Important
---------
- Select a T4 GPU runtime before running.
- GitHub/Drive must contain your latest modified trainer/train.py and related files.
- /content is temporary; the run directory in Google Drive is persistent.
"""

from __future__ import annotations

import argparse
import importlib
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
from typing import Iterable, Optional


DEFAULT_PROJECT_DIR = Path("/content/XPIDON-Composites")
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/XPIDON_Colab")
REQUIRED_PROJECT_FILES = (
    "main.py",
    "phy_params.json",
    "exp_params.json",
    "train_params.json",
)


class Tee:
    """Write text to both the Colab console and a persistent log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def run_command(command: list[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def mount_google_drive() -> None:
    """Mount Google Drive if it is not already mounted."""
    drive_my = Path("/content/drive/MyDrive")
    if drive_my.exists():
        print(f"Google Drive already mounted: {drive_my}")
        return

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "This script is intended for Google Colab. "
            "Mount Google Drive manually or run it in Colab."
        ) from exc

    drive.mount("/content/drive")
    if not drive_my.exists():
        raise RuntimeError("Google Drive mount did not produce /content/drive/MyDrive.")


def is_valid_project(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_PROJECT_FILES)


def safe_remove_project(path: Path) -> None:
    resolved = path.resolve()
    if resolved != DEFAULT_PROJECT_DIR.resolve():
        raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def restore_project(
    source: str,
    project_dir: Path,
    drive_root: Path,
    repo_url: Optional[str],
    branch: Optional[str],
    force_refresh: bool,
) -> str:
    """
    Restore the project to /content.

    Priority for --source auto:
      1. Existing valid /content project
      2. Drive project_backup
      3. GitHub repo
    """
    backup_dir = drive_root / "project_backup"

    if force_refresh and project_dir.exists():
        print(f"Removing existing temporary project: {project_dir}")
        safe_remove_project(project_dir)

    if source in ("existing", "auto") and is_valid_project(project_dir):
        print(f"Using existing temporary project: {project_dir}")
        return "existing"

    if source in ("drive", "auto") and is_valid_project(backup_dir):
        if project_dir.exists():
            safe_remove_project(project_dir)
        print(f"Restoring project from Drive: {backup_dir}")
        shutil.copytree(backup_dir, project_dir)
        return "drive"

    if source in ("github", "auto"):
        if not repo_url:
            if source == "github":
                raise ValueError("--repo-url is required when --source github is selected.")
        else:
            if project_dir.exists():
                safe_remove_project(project_dir)

            command = ["git", "clone"]
            if branch:
                command += ["--branch", branch, "--single-branch"]
            command += [repo_url, str(project_dir)]
            run_command(command)

            if not is_valid_project(project_dir):
                raise RuntimeError(
                    f"Cloned repository is missing required XPIDON files: {project_dir}"
                )
            return "github"

    raise RuntimeError(
        "Could not restore XPIDON-Composites.\n"
        "Use one of:\n"
        "  --source drive\n"
        "  --source existing\n"
        "  --source github --repo-url <URL> [--branch <BRANCH>]"
    )


def create_run_directories(drive_root: Path) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = drive_root / "runs" / timestamp

    paths = {
        "run": run_dir,
        "models": run_dir / "models",
        "loss_data": run_dir / "loss_data",
        "figures": run_dir / "figures",
        "predictions": run_dir / "predictions",
        "config": run_dir / "config",
        "source": run_dir / "source_snapshot",
        "logs": run_dir / "logs",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    (drive_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    return paths


def copy_source_snapshot(project_dir: Path, destination: Path) -> None:
    """Save the code/config state used for this training run."""
    ignored_names = {
        ".git",
        "__pycache__",
        ".ipynb_checkpoints",
        "outputs",
    }

    for item in project_dir.iterdir():
        if item.name in ignored_names:
            continue
        if item.is_file() and item.suffix in {".pkl", ".npy", ".npz"}:
            continue

        target = destination / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    "*.pkl",
                    "*.npy",
                    "*.npz",
                ),
            )
        elif item.is_file():
            shutil.copy2(item, target)


def pickle_is_readable(path: Path) -> bool:
    """Check that a pickle is complete before copying it to Drive."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False

        size_before = path.stat().st_size
        mtime_before = path.stat().st_mtime_ns

        with path.open("rb") as file:
            pickle.load(file)

        size_after = path.stat().st_size
        mtime_after = path.stat().st_mtime_ns
        return size_before == size_after and mtime_before == mtime_after
    except (EOFError, pickle.UnpicklingError, OSError, ValueError, TypeError):
        return False
    except Exception:
        # Some JAX objects may raise another exception while unpickling if a
        # dependency is temporarily unavailable. Do not copy a questionable file.
        return False


def atomic_copy(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".partial")
    try:
        shutil.copy2(source, temp)
        os.replace(temp, destination)
        return True
    except OSError:
        if temp.exists():
            temp.unlink(missing_ok=True)
        return False


def model_candidates(project_dir: Path) -> Iterable[Path]:
    patterns = (
        "T_temp.pkl",
        "a_temp.pkl",
        "xpidon_class_*_T.pkl",
        "xpidon_class_*_a.pkl",
    )

    seen: set[Path] = set()
    for pattern in patterns:
        for path in project_dir.glob(pattern):
            if path not in seen:
                seen.add(path)
                yield path

    outputs_models = project_dir / "outputs" / "models"
    if outputs_models.exists():
        for path in outputs_models.glob("*.pkl"):
            if path not in seen:
                seen.add(path)
                yield path


def sync_valid_models(project_dir: Path, model_dir: Path, verbose: bool = True) -> int:
    copied = 0
    for source in model_candidates(project_dir):
        if not pickle_is_readable(source):
            continue

        destination = model_dir / source.name
        source_stat = source.stat()

        if destination.exists():
            dest_stat = destination.stat()
            if (
                dest_stat.st_size == source_stat.st_size
                and dest_stat.st_mtime_ns >= source_stat.st_mtime_ns
            ):
                continue

        if atomic_copy(source, destination):
            copied += 1
            if verbose:
                print(f"[checkpoint] Saved: {destination}")

    return copied


def copy_directory_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        return

    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, target)
        except OSError as exc:
            print(f"[warning] Could not copy {item}: {exc}")


class PeriodicCheckpointSync:
    """Periodically preserve complete model pickle files while training."""

    def __init__(
        self,
        project_dir: Path,
        model_dir: Path,
        interval_minutes: float,
    ):
        self.project_dir = project_dir
        self.model_dir = model_dir
        self.interval_seconds = max(interval_minutes * 60.0, 30.0)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        print(
            f"Periodic checkpoint backup enabled: "
            f"every {self.interval_seconds / 60:.1f} minutes"
        )
        self.thread.start()

    def _worker(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            try:
                sync_valid_models(self.project_dir, self.model_dir, verbose=True)
            except Exception as exc:
                print(f"[warning] Periodic checkpoint sync failed: {exc}")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        sync_valid_models(self.project_dir, self.model_dir, verbose=True)


def get_runtime_metadata(
    project_dir: Path,
    project_source: str,
    training_parameters: dict,
) -> dict:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir),
        "project_source": project_source,
        "platform": platform.platform(),
        "python": sys.version,
        "training_parameters": training_parameters,
    }

    try:
        import jax
        import jaxlib
        import numpy
        import optax

        metadata.update(
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
        metadata["environment_error"] = repr(exc)

    return metadata


def save_final_artifacts(project_dir: Path, paths: dict[str, Path]) -> None:
    """Collect all expected outputs after normal completion or interruption."""
    sync_valid_models(project_dir, paths["models"], verbose=True)

    output_map = {
        project_dir / "outputs" / "models": paths["models"],
        project_dir / "outputs" / "loss_data": paths["loss_data"],
        project_dir / "outputs" / "figures": paths["figures"],
        project_dir / "outputs" / "predictions": paths["predictions"],
    }

    for source, destination in output_map.items():
        copy_directory_contents(source, destination)

    # Also collect common root-level result files.
    for pattern, destination in (
        ("loss_history_*.pkl", paths["loss_data"]),
        ("*.png", paths["figures"]),
        ("*.csv", paths["predictions"]),
        ("*.npy", paths["predictions"]),
        ("*.npz", paths["predictions"]),
    ):
        for source in project_dir.glob(pattern):
            try:
                shutil.copy2(source, destination / source.name)
            except OSError as exc:
                print(f"[warning] Could not copy {source}: {exc}")


def maybe_run_plot_script(project_dir: Path) -> None:
    plot_script = project_dir / "plot_training_results.py"
    if not plot_script.exists():
        print("plot_training_results.py not found; skipping final plotting.")
        return

    print("\nRunning plot_training_results.py ...")
    try:
        runpy.run_path(str(plot_script), run_name="__main__")
    except Exception:
        print("[warning] Plot script failed. Training models will still be saved.")
        traceback.print_exc()


def run_training(project_dir: Path, run_paths: dict[str, Path]) -> None:
    """Run the exact network/subdomain settings from the uploaded notebook."""
    os.chdir(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    # Avoid stale imports when rerunning in the same Colab notebook session.
    importlib.invalidate_caches()

    from utils.phy_params import PhyParams
    from utils.exp_params import ExpParams
    from utils.train_params import TrainParams
    from utils.air_temp import Temp_air
    from utils.data_utils import DataGenerator, generate_training_data
    from trainer.train import train
    from models.pidon import XPIDON
    from loss.loss import XPIDONLoss

    params = PhyParams(str(project_dir / "phy_params.json"))
    exp_params = ExpParams(str(project_dir / "exp_params.json"))
    train_params = TrainParams(str(project_dir / "train_params.json"))

    # Current parameters from training_book.ipynb
    init_sub_domain = 3
    tolerance_level = 2e-3
    m_inp = 9
    branch_layers = [m_inp, 50, 50, 50, 100]
    trunk_layers = [2, 50, 50, 50, 50, 50, 100]
    nomad_layers_T = [100, 50, 50, 50, 50, 1]

    training_parameters = {
        "init_sub_domain": init_sub_domain,
        "Tolerance_level": tolerance_level,
        "m_inp": m_inp,
        "branch_layers": branch_layers,
        "trunk_layers": trunk_layers,
        "nomad_layers_T": nomad_layers_T,
        "phy_params_file": "phy_params.json",
        "exp_params_file": "exp_params.json",
        "train_params_file": "train_params.json",
    }

    metadata = get_runtime_metadata(
        project_dir=project_dir,
        project_source="resolved-before-training",
        training_parameters=training_parameters,
    )
    (run_paths["config"] / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTraining configuration:")
    print(json.dumps(training_parameters, indent=2))
    print(f"\nPersistent run directory:\n{run_paths['run']}\n")

    train(
        XPIDON,
        XPIDONLoss,
        generate_training_data,
        DataGenerator,
        Temp_air,
        params,
        exp_params,
        train_params,
        init_sub_domain,
        tolerance_level,
        m_inp,
        branch_layers,
        trunk_layers,
        nomad_layers_T,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore, train, and persist XPIDON in a Colab T4 runtime."
    )
    parser.add_argument(
        "--source",
        choices=("auto", "existing", "drive", "github"),
        default="auto",
        help="Where the project should be restored from.",
    )
    parser.add_argument(
        "--repo-url",
        default=None,
        help="GitHub repository URL. Required for --source github.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional GitHub branch containing your latest modified code.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Temporary Colab project path.",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=DEFAULT_DRIVE_ROOT,
        help="Persistent Google Drive root.",
    )
    parser.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=5.0,
        help="Minutes between safe checkpoint backups.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Remove the current /content project and restore a fresh copy.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not run plot_training_results.py after training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    mount_google_drive()
    args.drive_root.mkdir(parents=True, exist_ok=True)

    project_source = restore_project(
        source=args.source,
        project_dir=args.project_dir,
        drive_root=args.drive_root,
        repo_url=args.repo_url,
        branch=args.branch,
        force_refresh=args.force_refresh,
    )

    run_paths = create_run_directories(args.drive_root)
    copy_source_snapshot(args.project_dir, run_paths["source"])

    # Save the three JSON files in a compact config directory as well.
    for filename in ("phy_params.json", "exp_params.json", "train_params.json"):
        source = args.project_dir / filename
        if source.exists():
            shutil.copy2(source, run_paths["config"] / filename)

    log_path = run_paths["logs"] / "training.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    checkpoint_sync = PeriodicCheckpointSync(
        project_dir=args.project_dir,
        model_dir=run_paths["models"],
        interval_minutes=args.checkpoint_minutes,
    )

    status = "unknown"
    exit_code = 0
    started_at = datetime.now()

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print("=" * 72)
            print("XPIDON Colab T4 training")
            print("=" * 72)
            print(f"Project source: {project_source}")
            print(f"Project directory: {args.project_dir}")
            print(f"Drive run directory: {run_paths['run']}")
            print(f"Started at: {started_at.isoformat(timespec='seconds')}")

            checkpoint_sync.start()
            run_training(args.project_dir, run_paths)

            if not args.skip_plots:
                maybe_run_plot_script(args.project_dir)

            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            exit_code = 130
            print("\nTraining interrupted by the user. Saving available checkpoints...")
        except Exception:
            status = "failed"
            exit_code = 1
            print("\nTraining failed. Saving available checkpoints and traceback...")
            traceback.print_exc()
        finally:
            try:
                checkpoint_sync.stop()
            except Exception:
                traceback.print_exc()

            try:
                save_final_artifacts(args.project_dir, run_paths)
            except Exception:
                print("[warning] Final artifact backup encountered an error.")
                traceback.print_exc()

            ended_at = datetime.now()
            summary = {
                "status": status,
                "project_source": project_source,
                "started_at": started_at.isoformat(timespec="seconds"),
                "ended_at": ended_at.isoformat(timespec="seconds"),
                "elapsed_seconds": (ended_at - started_at).total_seconds(),
                "run_directory": str(run_paths["run"]),
                "model_directory": str(run_paths["models"]),
                "log_file": str(log_path),
            }

            (run_paths["run"] / "run_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print("\n" + "=" * 72)
            print("Run summary")
            print("=" * 72)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("\nPersistent models:")
            for model in sorted(run_paths["models"].glob("*.pkl")):
                print(" ", model.name)

            sys.stdout = original_stdout
            sys.stderr = original_stderr

    return exit_code


if __name__ == "__main__":
    main()

