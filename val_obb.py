import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.utils import DEFAULT_CFG


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DroneVehicle OBB model.")
    parser.add_argument("--model", default="auto", help="Checkpoint path. Use 'auto' to resolve from --run-dir.")
    parser.add_argument("--run-dir", default=r"./runs/DroneVehicle_smoke", help="Run dir containing weights/best.pt.")
    parser.add_argument("--data", default=r"./data/DroneVehicle_local.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--save-dir",
        default=r"./runs/recheck",
        help="Output directory root. A unique subdir is auto-created unless --save-dir is explicitly set.",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Dataset split to evaluate.")
    parser.add_argument("--save", action="store_true", help="Save visualized validation predictions.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular inference.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing non-empty --save-dir.")
    return parser.parse_args()


def resolve_model_path(model_arg: str, run_dir: str) -> str:
    """Resolve checkpoint path for validation with a user-friendly fallback strategy."""
    if model_arg and model_arg.lower() != "auto":
        model_path = Path(model_arg)
        if model_path.exists():
            return str(model_path)
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. "
            "Please pass a valid --model path or use --model auto with --run-dir."
        )

    run_path = Path(run_dir)
    candidates = [run_path / "weights" / "best.pt", run_path / "weights" / "last.pt"]
    for checkpoint in candidates:
        if checkpoint.exists():
            return str(checkpoint)

    raise FileNotFoundError(
        f"No checkpoint found under {run_path / 'weights'}. "
        "Expected best.pt or last.pt. Please run training first or pass --model explicitly."
    )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_save_dir(cli_args, model_path: str) -> Path:
    save_dir = Path(cli_args.save_dir)
    default_root = Path("./runs/recheck")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save_dir == default_root:
        run_name = Path(cli_args.run_dir).name
        model_name = Path(model_path).stem
        save_dir = default_root / f"{run_name}_{model_name}_{cli_args.split}_{now}"

    if save_dir.exists():
        has_files = any(save_dir.iterdir())
        if has_files and not cli_args.exist_ok:
            raise FileExistsError(
                f"Save dir already exists and is not empty: {save_dir}. "
                "Use a new --save-dir or pass --exist-ok."
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def write_run_meta(save_dir: Path, cli_args, model_path: str):
    model_file = Path(model_path)
    data_file = Path(cli_args.data)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "resolved_model_path": str(model_file.resolve()),
        "resolved_run_dir": str(Path(cli_args.run_dir).resolve()),
        "model_sha256": _sha256_of_file(model_file) if model_file.exists() else None,
        "data_path": str(data_file.resolve()) if data_file.exists() else str(data_file),
        "data_sha256": _sha256_of_file(data_file) if data_file.exists() else None,
        "split": cli_args.split,
        "imgsz": cli_args.imgsz,
        "batch": cli_args.batch,
        "device": cli_args.device,
        "save": cli_args.save,
        "rect": cli_args.rect,
    }
    (save_dir / "val_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == '__main__':
    cli_args = parse_args()
    model_path = resolve_model_path(cli_args.model, cli_args.run_dir)
    save_dir = prepare_save_dir(cli_args, model_path)
    write_run_meta(save_dir, cli_args, model_path)
    DEFAULT_CFG.save_dir = str(save_dir)

    args = dict(
        model=model_path,
        data=cli_args.data,
        split=cli_args.split,
        device=cli_args.device,
        imgsz=cli_args.imgsz,
        batch=cli_args.batch,
        save=cli_args.save,
        rect=cli_args.rect,
    )
    validator = OBBValidator(args=args, save_dir=save_dir)
    validator(model=args["model"])


