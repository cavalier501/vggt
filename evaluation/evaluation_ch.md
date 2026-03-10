# VGGT Evaluation 中文说明

本文档基于官方 [README](./README.md) 翻译，并补充了当前仓库在 A2/NPU 环境下的推荐用法。

当前建议目标是先跑通官方 evaluation 的默认前向评测流程，也就是不带 `--use_ba` 的 feed-forward evaluation。`--use_ba` 依赖 `pycolmap`、`pyceres` 等额外环境，本说明暂不展开。

## 1. 目录建议

建议把数据和权重统一放在当前仓库下，目录如下：

```text
C:\Users\OmniInfer\Desktop\zh00942897\code\code_v1\vggt\vggt_zh\
├── evaluation\
├── data\
│   ├── co3d_v2\
│   └── co3d_v2_annotations\
└── weights\
    └── model.pt
```

推荐路径：

- Co3D 原始数据：`C:\Users\OmniInfer\Desktop\zh00942897\code\code_v1\vggt\vggt_zh\data\co3d_v2`
- Co3D 预处理标注：`C:\Users\OmniInfer\Desktop\zh00942897\code\code_v1\vggt\vggt_zh\data\co3d_v2_annotations`
- 模型权重：`C:\Users\OmniInfer\Desktop\zh00942897\code\code_v1\vggt\vggt_zh\weights\model.pt`

## 2. 模型权重

官方 evaluation README 提到，TrackHead 的 `pos_embed` 配置在公开 checkpoint 中有一个小问题，并提供了一个修正过 tracker head 的 checkpoint：

```bash
wget https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
```

不过这件事主要影响带 tracking / BA 的评测路径。对于当前先跑通的默认 feed-forward evaluation，不强制要求使用这个修正版权重。你可以先使用当前已经在 A2 上验证过可加载的 `model.pt`。

## 3. 环境准备

在仓库根目录下执行：

```bash
cd C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh
pip install -e .
```

当前文档只覆盖默认 evaluation 路径，因此不要求先安装 `pycolmap`、`pyceres`、LightGlue。

## 4. 获取 Co3D 数据集

Co3D 需要从官方仓库获取。官方入口：

- Co3D 仓库：[facebookresearch/co3d](https://github.com/facebookresearch/co3d)

常见流程如下：

1. 打开 Co3D 官方仓库，按照它的说明申请或下载 Co3D v2 数据。
2. 下载并解压后，把各个 category 目录直接放到：
   `C:\Users\OmniInfer\Desktop\zh00942897\code\code_v1\vggt\vggt_zh\data\co3d_v2`
3. 保持 Co3D 原始目录结构，不要手工改文件名。

解压后的目录应该类似下面这样：

```text
data/co3d_v2/
├── apple/
│   ├── frame_annotations.jgz
│   ├── sequence_annotations.jgz
│   ├── set_lists/
│   │   └── set_lists_fewview_dev.json
│   ├── images/
│   └── ...
├── backpack/
├── parkingmeter/
└── ...
```

也就是说，每个类别目录下至少应该能看到这些关键内容：

- `frame_annotations.jgz`
- `sequence_annotations.jgz`
- `set_lists/set_lists_fewview_dev.json`
- 图像数据目录

## 5. 预处理 Co3D 标注

evaluation 使用的并不是原始 Co3D 标注格式，需要先做一次预处理，生成 `*_train.jgz` 和 `*_test.jgz`。

建议始终在仓库根目录运行命令。

### 5.1 先跑单类冒烟

```bash
python evaluation/preprocess_co3d.py \
  --category parkingmeter \
  --co3d_v2_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2 \
  --output_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2_annotations
```

如果成功，应该能看到类似文件：

```text
data/co3d_v2_annotations/parkingmeter_train.jgz
data/co3d_v2_annotations/parkingmeter_test.jgz
```

### 5.2 全量预处理

```bash
python evaluation/preprocess_co3d.py \
  --category all \
  --co3d_v2_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2 \
  --output_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2_annotations
```

官方 README 提到，这一步通常只需要几分钟。

## 6. 运行官方 evaluation

### 6.1 冒烟测试

先建议只跑一个类别，并限制评测序列数量：

```bash
python evaluation/test_co3d.py \
  --debug \
  --fast_eval \
  --model_path C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/weights/model.pt \
  --co3d_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2 \
  --co3d_anno_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2_annotations \
  --seed 0
```

含义：

- `--debug`：当前脚本里会只测 `parkingmeter`
- `--fast_eval`：每个类别最多抽 10 个序列
- `--model_path`：本地权重路径
- `--co3d_dir`：Co3D 原始数据根目录
- `--co3d_anno_dir`：预处理后标注目录

### 6.2 小规模全类测试

去掉 `--debug`，保留 `--fast_eval`：

```bash
python evaluation/test_co3d.py \
  --fast_eval \
  --model_path C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/weights/model.pt \
  --co3d_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2 \
  --co3d_anno_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2_annotations \
  --seed 0
```

这一步适合确认所有类别目录和预处理产物都齐全。

### 6.3 正式前向评测

```bash
python evaluation/test_co3d.py \
  --model_path C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/weights/model.pt \
  --co3d_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2 \
  --co3d_anno_dir C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh/data/co3d_v2_annotations \
  --seed 0
```

## 7. 官方 README 中的结果说明

官方 README 给出的 `--fast_eval` 参考结果如下：

- Feed-forward estimation:
  - AUC@30: 89.98
  - AUC@15: 83.89
  - AUC@5: 67.45
  - AUC@3: 56.65

如果跑完整全量评测，官方给出的说明是：

- Feed-forward evaluation 的 Mean AUC@30 大约是 89.5

需要注意，这些数字是官方环境下的参考值。A2/NPU 环境、权重版本、依赖版本不同，结果可能会有偏差。

## 8. 当前 A2 环境的建议

对于当前仓库，建议按下面顺序推进：

1. 先确认 `preprocess_co3d.py` 能生成 `parkingmeter_test.jgz`
2. 再跑 `test_co3d.py --debug --fast_eval`
3. 然后跑全类别 `--fast_eval`
4. 最后再决定是否跑 full eval

当前优先目标是先跑通官方 evaluation 的默认前向路径，不先扩展到 BA 路径。
