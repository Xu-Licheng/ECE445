# ECE445 手势分类训练项目

本项目用于将手势视频转为图片数据集，并训练可部署到 K230 的双模型链路：

1) 手部检测模型（先做 hand boxing）
2) 手势分类模型（对 hand box 做分类）

当前仓库已包含：

1. 数据抽帧脚本。
2. 训练与评估脚本。
3. 一键全流程脚本。
4. 离线图片分类 demo。
5. 实时摄像头分类 demo。
6. 自动标注脚本（GroundingDINO）。
7. 手部检测训练脚本（YOLO）。

## 1. 项目结构

```text
ECE445/
├─ video/                              # 原始视频（每个视频对应一个类别）
├─ scripts/
│  ├─ extract_frames_dataset.py        # 抽帧建数据集
│  ├─ train_gesture_classifier.py      # 训练与评估
│  ├─ export_gesture_onnx_static.py    # 手势分类静态 ONNX 导出
│  ├─ auto_label_hand_boxes.py         # GroundingDINO 自动标注 hand box
│  ├─ build_hand_det_yolo_dataset.py   # 构建手检 YOLO 数据集
│  ├─ train_hand_detector_yolo.py      # 训练手部检测模型
│  ├─ run_full_pipeline.py             # 一键全流程脚本
│  ├─ run_k230_dual_model_pipeline.py  # 双模型总控脚本
│  ├─ demo_classify_images.py          # 离线图片分类 demo
│  └─ live_camera_gesture_demo.py      # 实时摄像头分类 demo
├─ requirements.txt                    # 基础依赖（不含 torch/torchvision）
├─ test_demo/                          # 离线 demo 输入与输出目录
└─ README.md
```

## 2. 环境与安装

- 操作系统：Windows（其他系统也可）
- Python：建议 3.10-3.13
- NVIDIA GPU + CUDA（推荐）
- 直接在当前本地 Python 环境安装即可

先安装基础依赖：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果你要跑自动标注和手检训练（推荐），额外安装：

```bash
pip install transformers ultralytics accelerate
```

如果你还想保留 MediaPipe 做 PC 对照测试，可选安装：

```bash
pip install mediapipe
```

再安装 PyTorch（按显卡选择）：

方案 A：RTX 50 系列建议使用 cu128

```bash
pip uninstall -y torch torchvision
pip install --upgrade --force-reinstall "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

方案 B：常规显卡可使用 cu124

```bash
pip uninstall -y torch torchvision
pip install --upgrade --force-reinstall "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

安装后验证 CUDA：

```bash
python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__);print('cuda',torch.version.cuda);print('avail',torch.cuda.is_available());print('count',torch.cuda.device_count());print('arch',torch.cuda.get_arch_list())"
```

## 3. 数据准备

把视频放到 video 目录。每个视频文件会被当作一个类别，类别名来自文件名（去后缀）。

示例：

```text
video/
├─ ok_up.mp4
├─ ok_down.mp4
├─ thumb_up.mp4
└─ thumb_down.mp4
```

## 4. 一键全流程（推荐）

命令：

```bash
python scripts/run_full_pipeline.py --resolution 160 --overwrite_dataset --pretrained --epochs 25 --device cuda:0
```

这条命令会自动完成：

1. 抽帧生成 dataset/raw_frames。
2. 训练模型。
3. 测试评估。
4. 输出日志与图表到 training_runs/run_xxx。

如果只想重训（跳过抽帧）：

```bash
python scripts/run_full_pipeline.py --skip_extract --resolution 160 --pretrained --epochs 25 --device cuda:0
```

CPU 回退：

```bash
python scripts/run_full_pipeline.py --resolution 160 --overwrite_dataset --pretrained --epochs 25 --device cpu
```

查看参数：

```bash
python scripts/run_full_pipeline.py -h
```

## 5. 分步流程（可选）

1) 抽帧建数据集

```bash
python scripts/extract_frames_dataset.py --video_dir video --output_dir dataset/raw_frames --sample_every_n_frames 1 --resize_width 160 --resize_height 160 --overwrite
```

2) 训练与评估

```bash
python scripts/train_gesture_classifier.py --dataset_dir dataset/raw_frames --output_dir training_runs --image_size 160 --epochs 25 --batch_size 32 --device cuda:0 --pretrained
```

## 6. 离线图片分类 demo（先 hand box 再分类）

把待分类图片放到 test_demo 根目录，然后运行：

```bash
python scripts/demo_classify_images.py --model_path training_runs/run_xxx/best_model.pth --demo_dir test_demo --device cuda:0 --hand_detector yolo --hand_det_model_path training_runs_hand_det/run_xxx/weights/best.pt
```

默认行为是复制原图到预测类别目录。若希望移动文件：

```bash
python scripts/demo_classify_images.py --model_path training_runs/run_xxx/best_model.pth --demo_dir test_demo --device cuda:0 --hand_detector yolo --hand_det_model_path training_runs_hand_det/run_xxx/weights/best.pt --move_files
```

如果要求必须检测到手框才分类：

```bash
python scripts/demo_classify_images.py --model_path training_runs/run_xxx/best_model.pth --demo_dir test_demo --device cuda:0 --hand_detector yolo --hand_det_model_path training_runs_hand_det/run_xxx/weights/best.pt --require_hand_box
```

输出说明：

1. test_demo/ok_down
2. test_demo/ok_up
3. test_demo/thumb_down
4. test_demo/thumb_up
5. test_demo/predictions.csv

## 7. 实时摄像头分类 demo（先 hand box 再分类）

```bash
python scripts/live_camera_gesture_demo.py --gesture_model_path training_runs/run_gesture/best_model.pth --hand_det_model_path training_runs_hand_det/run_hand_det/weights/best.pt --device cuda:0
```

如果你已经训练好两个模型，也可以省略模型路径自动选最新：

```bash
python scripts/live_camera_gesture_demo.py --device cuda:0
```

常用参数：

1. --camera_index 0（摄像头编号）
2. --smoothing_window 5（平滑窗口）
3. --min_confidence 0.6（低于阈值显示 uncertain）
4. --no-mirror（关闭镜像）
5. --box_expand_ratio 1.25（手框扩张比例）
6. --gesture_model_path xxx.pth（手势分类模型路径）
7. --hand_det_model_path xxx.pt（手部检测模型路径）

退出方式：按 q 或 ESC。

提示：实时脚本默认走双模型流程（YOLO 手检 + 手势分类），不再依赖 skin 回退。

## 8. 自动标注与手检训练（K230 推荐流程）

1) 从手势分类数据自动生成 hand box 伪标签（GroundingDINO）

```bash
python scripts/auto_label_hand_boxes.py --input_dir dataset/raw_frames --output_images_dir dataset/hand_det/images_all --output_labels_dir dataset/hand_det/labels_all --summary_csv dataset/hand_det/autolabel_summary.csv --device cuda:0
```

2) 构建 YOLO 手检数据集划分

```bash
python scripts/build_hand_det_yolo_dataset.py --images_dir dataset/hand_det/images_all --labels_dir dataset/hand_det/labels_all --output_dir dataset/hand_det/yolo --overwrite
```

3) 训练手部检测模型并导出 ONNX

```bash
python scripts/train_hand_detector_yolo.py --data_yaml dataset/hand_det/yolo/dataset.yaml --imgsz 320 --epochs 80 --batch 32 --device cuda:0 --project training_runs_hand_det --run_name run_hand_det --export_onnx
```

4) 训练手势分类模型

```bash
python scripts/train_gesture_classifier.py --dataset_dir dataset/raw_frames --output_dir training_runs --run_name run_gesture --image_size 160 --epochs 25 --batch_size 32 --device cuda:0 --pretrained
```

5) 导出手势分类静态 ONNX（K230 友好）

```bash
python scripts/export_gesture_onnx_static.py --model_path training_runs/run_gesture/best_model.pth --output_onnx training_runs/run_gesture/gesture_classifier_static.onnx --image_size 160 --opset 12 --device cpu
```

6) 一条命令跑完整双模型流水线

```bash
python scripts/run_k230_dual_model_pipeline.py --overwrite_dataset --resolution 160 --epochs_hand_det 80 --epochs_gesture 25 --device cuda:0
```

## 9. K230 转换与部署模板

以下为 nncase 工具链模板命令，请按你本地 K230 SDK 版本调整参数名：

1) 手部检测 ONNX -> kmodel

```bash
nncase compile --target k230 --input-format onnx --input-file training_runs_hand_det/run_hand_det/weights/best.onnx --output-file deploy/hand_detector.kmodel --dataset deploy/calib_hand_det --input-shape 1,3,320,320
```

2) 手势分类 ONNX -> kmodel

```bash
nncase compile --target k230 --input-format onnx --input-file training_runs/run_gesture/gesture_classifier_static.onnx --output-file deploy/gesture_classifier.kmodel --dataset deploy/calib_gesture --input-shape 1,3,160,160
```

3) 板端推理顺序

相机帧 -> hand_detector.kmodel 出框 -> ROI 裁剪/resize -> gesture_classifier.kmodel 分类 -> 叠加显示结果

## 10. 结果文件说明

每次训练输出到 training_runs/run_xxx，常见文件：

1. train.log
2. metrics.csv
3. training_curves.png
4. confusion_matrix.png
5. classification_report.json
6. summary.json
7. best_model.pth
8. final_model.pth

快速查看测试准确率：

```bash
python -c "import json;print(json.load(open('training_runs/run_xxx/summary.json','r',encoding='utf-8'))['test_acc'])"
```

## 11. 常见问题

1) 报错 Torch not compiled with CUDA enabled

- 说明：安装到了 CPU 版 torch。
- 处理：按第 2 节重新安装 CUDA 版 torch/torchvision。

2) 报错 no kernel image is available for execution on the device

- 说明：当前 torch 不支持你的显卡架构（例如 sm_120）。
- 处理：升级到更高版本 CUDA 轮子（优先 cu128 或 nightly）。

3) OpenCV GStreamer Quicktime demuxer 警告

- 说明：Windows 下常见插件提示，通常不影响通过 ffmpeg 后端读 mp4。

4) 显存不足（CUDA out of memory）

- 降低 batch_size（如 32 -> 16 -> 8）。
- 降低输入分辨率（如 160 -> 128）。

## 12. 后续方向

当 PC 端效果稳定后，可继续做 ONNX 到 K230 的模型转换与板端推理验证。
