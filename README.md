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
├─ video/                              # 原始视频（每个类别一个文件夹，文件夹内可放多个视频）
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
│  ├─ live_camera_gesture_demo.py      # 实时摄像头分类 demo
│  └─ gesture_debounce.py              # 实时输出 debounce 状态机
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

把视频放到 `video/` 目录下的类别文件夹里。每个文件夹对应一个类别，文件夹中可放多个视频，抽帧时会合并到同一类别。

示例：

```text
video/
├─ ok_down/
│  ├─ ok_down_01.mp4
│  └─ ok_down_02.mp4
├─ ok_up/
│  ├─ ok_up_01.mp4
│  └─ ok_up_02.mp4
├─ thumb_down/
│  ├─ thumb_down_01.mp4
│  └─ thumb_down_02.mp4
└─ thumb_up/
   ├─ thumb_up_01.mp4
   └─ thumb_up_02.mp4
```

兼容说明：仍支持旧结构（`video/` 下直接放视频），但推荐使用文件夹结构。  
注意：请不要混用两种结构。当前逻辑是只要 `video/` 下存在非空类别子文件夹，就优先按“文件夹结构”读取，此时根目录平铺视频会被忽略。

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
python scripts/demo_classify_images.py --model_path training_runs/run_xxx/best_model.pth --demo_dir test_demo --device cuda:0 --hand_detector yolo --hand_det_model_path <hand_det_best.pt>
```

`<hand_det_best.pt>` 常见位置：

1. `training_runs_hand_det/run_xxx/weights/best.pt`
2. `runs/detect/training_runs_hand_det/run_xxx/weights/best.pt`

默认行为是复制原图到预测类别目录。若希望移动文件：

```bash
python scripts/demo_classify_images.py --model_path training_runs/run_xxx/best_model.pth --demo_dir test_demo --device cuda:0 --hand_detector yolo --hand_det_model_path <hand_det_best.pt> --move_files
```

如果要求必须检测到手框才分类：

```bash
python scripts/demo_classify_images.py --model_path training_runs/run_xxx/best_model.pth --demo_dir test_demo --device cuda:0 --hand_detector yolo --hand_det_model_path <hand_det_best.pt> --require_hand_box
```

输出说明：

1. test_demo/ok_down
2. test_demo/ok_up
3. test_demo/thumb_down
4. test_demo/thumb_up
5. test_demo/predictions.csv

## 7. 实时摄像头分类 demo（先 hand box 再分类）

```bash
python scripts/live_camera_gesture_demo.py --gesture_model_path training_runs/run_gesture/best_model.pth --hand_det_model_path <hand_det_best.pt> --device cuda:0
```

如果你已经训练好两个模型，也可以尝试省略模型路径自动选最新：

```bash
python scripts/live_camera_gesture_demo.py --device cuda:0
```

注意：实时脚本当前只会自动搜索 `training_runs_hand_det/*/weights/best.pt`。  
如果你的手检模型在 `runs/detect/training_runs_hand_det/...`，请显式传 `--hand_det_model_path`。

常用参数：

1. --camera_index 0（摄像头编号）
2. --smoothing_window 5（平滑窗口）
3. --min_confidence 0.6（进入新手势输出所需的最低置信度）
4. --hold_confidence 0.5（滞回保持阈值，默认比 min_confidence 低 0.10）
5. --min_top_margin 0.10（top1 和 top2 概率差至少为 0.10）
6. --stable_frames 3（同一手势连续稳定 N 帧后才输出）
7. --min_response_seconds 0.5（两次输出标签变化的最短间隔）
8. --hold_last_seconds 0.3（短暂低置信度时保持上一个有效手势）
9. --no_hand_timeout_seconds 0.2（短暂丢失手框时保持上一个输出）
10. --default_label default（低置信度且没有 uncertain 类时输出的默认类别）
11. --uncertain_label uncertain（如果该类别存在，低置信度时优先输出它）
12. --no_hand_label no_hand（长时间未检测到手框时输出的类别）
13. --no-mirror（关闭镜像）
14. --box_expand_ratio 1.25（手框扩张比例）
15. --gesture_model_path xxx.pth（手势分类模型路径）
16. --hand_det_model_path xxx.pt（手部检测模型路径）

实时脚本现在包含输出 debounce：先对概率做 `smoothing_window` 平滑，再检查置信度、top1/top2 margin、连续稳定帧数和最短响应时间。低置信度时，如果模型类别列表里包含 `uncertain`，输出 `uncertain`；否则输出 `--default_label`。短暂低置信度或短暂丢失手框时会保持上一个有效输出，超过 timeout 后才切到默认或 `no_hand`。

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
nncase compile --target k230 --input-format onnx --input-file <hand_detector_best.onnx> --output-file deploy/hand_detector.kmodel --dataset deploy/calib_hand_det --input-shape 1,3,320,320
```

`<hand_detector_best.onnx>` 建议用训练日志里打印的导出路径；常见在：

1. `training_runs_hand_det/run_xxx/weights/best.onnx`
2. `runs/detect/training_runs_hand_det/run_xxx/weights/best.onnx`

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

## 13. 云端最简验证流程

1) 将视频拆分成图片并自动标注手部框

```bash
python scripts/extract_frames_dataset.py --video_dir video --output_dir dataset/raw_frames --sample_every_n_frames 1 --resize_width 160 --resize_height 160 --overwrite
python scripts/auto_label_hand_boxes.py --input_dir dataset/raw_frames --output_images_dir dataset/hand_det/images_all --output_labels_dir dataset/hand_det/labels_all --summary_csv dataset/hand_det/autolabel_summary.csv --device cuda:0
python scripts/build_hand_det_yolo_dataset.py --images_dir dataset/hand_det/images_all --labels_dir dataset/hand_det/labels_all --output_dir dataset/hand_det/yolo --overwrite
```

2) 用标注数据训练手部识别模块

```bash
python scripts/train_hand_detector_yolo.py --data_yaml dataset/hand_det/yolo/dataset.yaml --imgsz 320 --epochs 80 --batch 32 --device cuda:0 --project training_runs_hand_det --run_name run_hand_det --export_onnx
```

3) 用手部识别模块切片后训练手势分类模块

```bash
python scripts/build_gesture_dataset_from_hand_detector.py --input_dir dataset/raw_frames --output_dir dataset/gesture_crops --hand_det_model_path <hand_det_best.pt> --detector yolo --require_hand_box --overwrite
python scripts/train_gesture_classifier.py --dataset_dir dataset/gesture_crops --output_dir training_runs --run_name run_gesture_crop --image_size 160 --epochs 25 --batch_size 32 --device cuda:0 --pretrained
```

4) 跑 camera 端 demo

```bash
python scripts/live_camera_gesture_demo.py --gesture_model_path training_runs/run_gesture_crop/best_model.pth --hand_det_model_path <hand_det_best.pt> --device cuda:0
```

5) 推荐 camera 端 demo 参数设置
```bash
python scripts/live_camera_gesture_demo.py --gesture_model_path training_runs/run_gesture_crop/best_model.pth --hand_det_model_path runs/detect/training_runs_hand_det/run_hand_det3/weights/best.pt --device auto --min_response_seconds 5.0 --hold_last_seconds 1.0 --no_hand_timeout_seconds 0.2 --smoothing_window 7 --min_confidence 0.9 --hold_confidence 0.7
```
