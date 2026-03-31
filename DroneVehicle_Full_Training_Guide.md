# DroneVehicle 完整训练指南（M2D-LIF）

## 1. 文档目标

本文档用于说明如何在本仓库中，把 DroneVehicle 数据集完整跑通到可复现实验结果，覆盖以下内容：

1. 原始 XML 数据转换为训练可用格式。
2. 双模态 OBB 正式训练（RGB+IR）。
3. 验证与结果查看。
4. 训练相关代码文件说明与调用关系。

当数据/模型配置中的 `ch: 6` 生效时，本项目会按 RGB+IR 双模态流程运行。

## 2. 环境与工作目录

请在仓库根目录执行命令：

```bash
cd /root/shared-nvme/M2D-LIF-master/M2D-LIF-master
```

建议环境：

1. Python 3.10+
2. CUDA 版本与 PyTorch 匹配
3. 单卡训练（以下命令用 `--device 0`）

## 3. 数据流程总览

原始 DroneVehicle 数据（`trainimg/trainimgr + XML`）不能被训练脚本直接读取，必须先转换。

训练期望的数据结构：

1. `images/{train,val,test}`
2. `images_ir/{train,val,test}`
3. `labels/{train,val,test}`，标签为 YOLO-OBB 格式：
   `class x1 y1 x2 y2 x3 y3 x4 y4`（归一化）

本仓库的转换脚本已增强，可处理目录层级差异与标签脏数据（如 `feright car`、`truvk`、`*`）。

## 4. 第一步：原始数据转换

执行：

```bash
python scripts/prepare_dronevehicle.py \
  --source-root "/root/shared-nvme/data" \
  --multimodal-root "/root/shared-nvme/data/DroneVehicle_yolo_obb_multimodal" \
  --rgb-root "/root/shared-nvme/data/DroneVehicle_rgb_obb" \
  --ir-root "/root/shared-nvme/data/DroneVehicle_ir_obb" \
  --multimodal-label-source rgb
```

参数说明：

1. `--source-root`：原始数据根目录（含 train/val/test）。
2. `--multimodal-root`：双模态训练数据输出目录。
3. `--rgb-root`：RGB 单模态输出目录（可选保留）。
4. `--ir-root`：IR 单模态输出目录（可选保留）。
5. `--multimodal-label-source`：双模态标签来源，`rgb` 或 `ir`。

预期输出：

1. 控制台打印 train/val/test 三个 split 的转换统计。
2. 最终打印三套输出目录路径。

产物目录：

1. `/root/shared-nvme/data/DroneVehicle_yolo_obb_multimodal`
2. `/root/shared-nvme/data/DroneVehicle_rgb_obb`
3. `/root/shared-nvme/data/DroneVehicle_ir_obb`

## 5. 第二步：确认 Linux 数据配置

使用 `data/DroneVehicle_linux.yaml`：

```yaml
path: /root/shared-nvme/data/DroneVehicle_yolo_obb_multimodal

train: images/train
val: images/val
test: images/test

nc: 5
ch: 6

names:
  0: car
  1: truck
  2: feright_car
  3: bus
  4: van
```

注意事项：

1. `nc: 5` 是 DroneVehicle 正确类别数。
2. `ch: 6` 表示双模态输入（RGB+IR）。
3. 模型 YAML 若写了其它 `nc`，训练时会被数据 YAML 自动覆盖，这是正常行为。

## 6. 第三步：正式训练（先不蒸馏）

推荐命令：

```bash
python train_dist_obb.py \
  --model "./model_yaml_obb/yolov8_LIF_obb.yaml" \
  --data "./data/DroneVehicle_linux.yaml" \
  --imgsz 640 \
  --epochs 100 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --save-dir "./runs/DroneVehicle_full"
```

- 可以生成日志的命令：

```bash
mkdir -p logs
python train_dist_obb.py \
  --model "./model_yaml_obb/yolov8_LIF_obb.yaml" \
  --data "./data/DroneVehicle_linux.yaml" \
  --imgsz 640 \
  --epochs 100 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --save-dir "./runs/DroneVehicle_full" 2>&1 | tee -a logs/dronevehicle_full_train.log
```

输入参数解释：

1. `--model`：模型结构定义文件。
2. `--data`：数据配置文件。
3. `--imgsz`：训练输入分辨率。
4. `--epochs`：总训练轮次。
5. `--batch`：批大小。
6. `--device`：GPU 编号。
7. `--workers`：DataLoader 线程数。
8. `--save-dir`：训练输出目录。

训练输出（核心产物）：

1. `runs/DroneVehicle_full/weights/last.pt`
2. `runs/DroneVehicle_full/weights/best.pt`
3. `runs/DroneVehicle_full/results.csv`
4. `runs/DroneVehicle_full/labels.jpg`
5. `runs/DroneVehicle_full` 下 TensorBoard 日志

## 7. 第四步：验证

按 run 目录自动找权重（优先 best，其次 last）：

```bash
python val_obb.py \
  --run-dir "./runs/DroneVehicle_full" \
  --data "./data/DroneVehicle_linux.yaml" \
  --save --rect
```

或直接指定权重：

```bash
python val_obb.py \
  --model "./runs/DroneVehicle_full/weights/best.pt" \
  --data "./data/DroneVehicle_linux.yaml" \
  --save --rect
```

验证输出：

1. 终端指标：`P`、`R`、`mAP50`、`mAP50-95`
2. 若加 `--save`，保存可视化预测结果
3. 输出目录为 `--save-dir`（默认 `./runs/DroneVehicle_val`）

## 8.第五步：测试

```bash
python val_obb.py \
  --model ./runs/DroneVehicle_full/weights/best.pt \
  --data ./data/DroneVehicle_linux.yaml \
  --split test \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --save
```

## 8. 可选：蒸馏训练

如果你已有 RGB 与 IR 两个教师模型，可使用：

```bash
python train_dist_obb.py \
  --model "./model_yaml_obb/yolov8_LIF_obb.yaml" \
  --data "./data/DroneVehicle_linux.yaml" \
  --teacher-rgb "/path/to/rgb_teacher.pt" \
  --teacher-ir "/path/to/ir_teacher.pt" \
  --distill-weight 0.8 \
  --loss-type CWD \
  --imgsz 640 \
  --epochs 100 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --save-dir "./runs/DroneVehicle_distill"
```

规则：

1. `--teacher-rgb` 与 `--teacher-ir` 必须同时提供。
2. 只给一个会触发脚本报错，这是预期保护逻辑。

## 9. 正式训练涉及的核心代码文件

1. `scripts/prepare_dronevehicle.py`
   作用：原始 XML 转 YOLO-OBB，自动识别 split 根目录，处理脏标签别名并跳过非法标签。

2. `data/DroneVehicle_linux.yaml`
   作用：定义 Linux 环境下的数据路径、类别数和通道数。

3. `train_dist_obb.py`
   作用：OBB 训练主入口，负责组装训练参数，按需接入双教师蒸馏。

4. `model_yaml_obb/yolov8_LIF_obb.yaml`
   作用：模型结构定义；该文件使用双模态相关模块，配合 `ch: 6` 运行。

5. `ultralytics/data/base.py`
   作用：当 `ch > 3` 时按规则加载 `images_ir` 并与 RGB 配对输入。

6. `val_obb.py`
   作用：验证入口，支持 `--run-dir` 自动解析 `best.pt/last.pt`。

7. `ultralytics/utils/torch_utils.py`
   作用：`strip_optimizer()`；已修复 torch 2.6+ 默认 `weights_only=True` 的兼容问题。

8. `ultralytics/nn/tasks.py`
   作用：`torch_safe_load()`；已修复 torch 2.6+ checkpoint 加载兼容问题。

## 10. 输入与输出总表（正式训练）

输入：

1. 原始数据根目录：`/root/shared-nvme/data`
2. 模型 YAML：`./model_yaml_obb/yolov8_LIF_obb.yaml`
3. 数据 YAML：`./data/DroneVehicle_linux.yaml`

输出：

1. 转换后双模态数据：`/root/shared-nvme/data/DroneVehicle_yolo_obb_multimodal`
2. 训练目录：`./runs/DroneVehicle_full`
3. 核心权重：`./runs/DroneVehicle_full/weights/best.pt`、`./runs/DroneVehicle_full/weights/last.pt`
4. 验证输出：`./runs/DroneVehicle_val`（或自定义 `--save-dir`）

## 11. 常见问题与快速排查

问题 1：`No checkpoint found under .../weights`

1. 原因：训练未完成或 `--run-dir` 指错。
2. 检查：`ls -l ./runs/DroneVehicle_full/weights`

问题 2：`UnpicklingError ... weights_only`

1. 原因：torch 2.6+ 默认加载行为变更。
2. 状态：仓库已在 `ultralytics/utils/torch_utils.py` 与 `ultralytics/nn/tasks.py` 做兼容修复。

问题 3：XML 类别名异常（如 `feright`、`truvk`、`*`）

1. 原因：原始标注噪声。
2. 状态：转换脚本已支持别名归一化并跳过非法标签。

## 12. 一套最小可复现命令

按顺序执行：

```bash
python scripts/prepare_dronevehicle.py \
  --source-root "/root/shared-nvme/data" \
  --multimodal-root "/root/shared-nvme/data/DroneVehicle_yolo_obb_multimodal" \
  --rgb-root "/root/shared-nvme/data/DroneVehicle_rgb_obb" \
  --ir-root "/root/shared-nvme/data/DroneVehicle_ir_obb" \
  --multimodal-label-source rgb

python train_dist_obb.py \
  --model "./model_yaml_obb/yolov8_LIF_obb.yaml" \
  --data "./data/DroneVehicle_linux.yaml" \
  --imgsz 640 --epochs 100 --batch 8 --device 0 --workers 4 \
  --save-dir "./runs/DroneVehicle_full"

python val_obb.py \
  --run-dir "./runs/DroneVehicle_full" \
  --data "./data/DroneVehicle_linux.yaml" \
  --save --rect
```
