import os

import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG


def parse_args():
    parser = argparse.ArgumentParser(description="Train DroneVehicle OBB model with optional distillation.")
    parser.add_argument("--model", default=r"./model_yaml_obb/yolov8_LIF_obb.yaml")
    parser.add_argument("--data", default=r"./data/DroneVehicle_local.yaml")
    parser.add_argument("--teacher-rgb", default="", help="Optional RGB teacher checkpoint path.")
    parser.add_argument("--teacher-ir", default="", help="Optional IR teacher checkpoint path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-dir", default=r"./runs/DroneVehicle")
    parser.add_argument("--distill-weight", type=float, default=0.8)
    parser.add_argument("--loss-type", default="CWD")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability (0-1).")
    parser.add_argument(
        "--auto-augment",
        default=None,
        help="Auto-augment policy: randaugment|autoaugment|augmix|none/null to disable.",
    )
    parser.add_argument("--erasing", type=float, default=None, help="Random erasing probability (0-1).")
    parser.add_argument("--cls", type=float, default=None, help="Classification loss gain.")
    parser.add_argument("--rare-sampler", action="store_true", help="Enable rare-class-aware sampling.")
    parser.add_argument("--rare-sampler-base-weight", type=float, default=None)
    parser.add_argument("--rare-sampler-empty-weight", type=float, default=None)
    parser.add_argument("--rare-sampler-bonus-truck", type=float, default=None)
    parser.add_argument("--rare-sampler-bonus-freight", type=float, default=None)
    parser.add_argument("--rare-sampler-bonus-bus", type=float, default=None)
    parser.add_argument("--rare-sampler-bonus-van", type=float, default=None)
    return parser.parse_args()


def maybe_load_teacher(weight_path):
    if not weight_path:
        return None
    _, checkpoint = attempt_load_one_weight(weight_path)
    checkpoint["model"].info()
    return checkpoint["model"]


if __name__ == '__main__':
    cli_args = parse_args()

    teacher_rgb = maybe_load_teacher(cli_args.teacher_rgb)
    teacher_ir = maybe_load_teacher(cli_args.teacher_ir)

    if (teacher_rgb is None) != (teacher_ir is None):
        raise ValueError("Please provide both --teacher-rgb and --teacher-ir, or provide neither of them.")

    args = dict(
        model=cli_args.model,
        data=cli_args.data,
        amp=True,
        imgsz=cli_args.imgsz,
        epochs=cli_args.epochs,
        batch=cli_args.batch,
        device=cli_args.device,
        lr0=cli_args.lr0,
        online=cli_args.online,
        augment=not cli_args.no_augment,
        workers=cli_args.workers
    )
    if cli_args.mosaic is not None:
        args["mosaic"] = cli_args.mosaic
    if cli_args.auto_augment is not None:
        auto_aug = cli_args.auto_augment.strip().lower()
        args["auto_augment"] = None if auto_aug in ("none", "null") else cli_args.auto_augment
    if cli_args.erasing is not None:
        args["erasing"] = cli_args.erasing
    if cli_args.cls is not None:
        args["cls"] = cli_args.cls
    if cli_args.rare_sampler:
        args["rare_sampler"] = True
    if cli_args.rare_sampler_base_weight is not None:
        args["rare_sampler_base_weight"] = cli_args.rare_sampler_base_weight
    if cli_args.rare_sampler_empty_weight is not None:
        args["rare_sampler_empty_weight"] = cli_args.rare_sampler_empty_weight
    if cli_args.rare_sampler_bonus_truck is not None:
        args["rare_sampler_bonus_truck"] = cli_args.rare_sampler_bonus_truck
    if cli_args.rare_sampler_bonus_freight is not None:
        args["rare_sampler_bonus_freight"] = cli_args.rare_sampler_bonus_freight
    if cli_args.rare_sampler_bonus_bus is not None:
        args["rare_sampler_bonus_bus"] = cli_args.rare_sampler_bonus_bus
    if cli_args.rare_sampler_bonus_van is not None:
        args["rare_sampler_bonus_van"] = cli_args.rare_sampler_bonus_van

    if teacher_rgb is not None and teacher_ir is not None:
        args.update(
            Distillation="MultiDistillation",
            distill_weight=cli_args.distill_weight,
            Teacher_Model_RGB=teacher_rgb,
            Teacher_Model_IR=teacher_ir,
            loss_type=cli_args.loss_type,
        )

    DEFAULT_CFG.save_dir = cli_args.save_dir

    model_s = OBBTrainer(overrides=args)
    model_s.train()
