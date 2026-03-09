# VGGT 端到端推理运行报告

## 1. 结论

在当前仓库中，完整后处理链路已经封装在以下两个脚本里：

- `demo_viser.py`
- `demo_colmap.py`

它们不是只做模型前向，而是已经包含了完整后处理：

- 位姿解码：`pose_encoding_to_extri_intri`
- 深度逆投影：`unproject_depth_map_to_point_map`
- 坐标变换：`closed_form_inverse_se3`
- 可视化：viser 点云与相机位姿显示
- 导出：COLMAP `sparse/` 结果，且 `demo_colmap.py` 支持可选 BA

`demo_detailed_usage.py` 仍然只是中间张量级示例，不是端到端入口。

## 2. 运行前准备

工作目录：

```bash
cd C:/Users/OmniInfer/Desktop/zh00942897/code/code_v1/vggt/vggt_zh
```

安装 demo 依赖：

```bash
pip install -r requirements_demo.txt
```

说明：

- `demo_viser.py` 和 `demo_colmap.py` 默认会在线下载权重
- 现在这两个脚本都额外支持 `--pt_path`，可以强制使用本地 `model.pt`
- `demo_colmap.py --use_ba` 额外依赖 `pycolmap`、`pyceres`、`LightGlue`
- `demo_viser.py --mask_sky` 会额外下载 `skyseg.onnx`

## 3. 跑完整可视化链路

先用仓库自带样例：

```bash
python demo_viser.py --image_folder examples/kitchen/images --port 8080
```

如果你要显式指定本地权重：

```bash
python demo_viser.py --image_folder examples/kitchen/images --port 8080 --pt_path C:/path/to/model.pt
```

如果你要跑自己的图片目录：

```bash
python demo_viser.py --image_folder YOUR_IMAGE_FOLDER --port 8080 --pt_path C:/path/to/model.pt
```

这条链路会完成：

- 图像预处理
- 模型推理
- 相机内外参解码
- 基于深度或 point map 的 3D 点构建
- 相机 frustum 与点云可视化

## 4. 跑完整 COLMAP 导出链路

数据目录必须是：

```text
SCENE_DIR/
  images/
    *.png / *.jpg
```

先跑无 BA 版本：

```bash
python demo_colmap.py --scene_dir examples/kitchen
```

如果你要显式指定本地权重：

```bash
python demo_colmap.py --scene_dir examples/kitchen --pt_path C:/path/to/model.pt
```

对你自己的数据：

```bash
python demo_colmap.py --scene_dir YOUR_SCENE --pt_path C:/path/to/model.pt
```

输出目录：

```text
YOUR_SCENE/
  sparse/
    cameras.bin
    images.bin
    points3D.bin
    points.ply
```

## 5. 跑带 BA 的完整链路

```bash
python demo_colmap.py --scene_dir YOUR_SCENE --use_ba --pt_path C:/path/to/model.pt
```

如果你想先降低耗时和占用：

```bash
python demo_colmap.py --scene_dir YOUR_SCENE --use_ba --max_query_pts 2048 --query_frame_num 5 --pt_path C:/path/to/model.pt
```

说明：

- `demo_colmap.py --use_ba` 的轨迹部分走的是官方封装的跟踪逻辑
- 不需要再把 `demo_detailed_usage.py` 的 `track_head` 手工拼回主流程

## 6. 建议的验证顺序

建议按下面顺序检查：

1. `demo_viser.py` 在 `examples/kitchen/images` 上能正常启动并看到点云
2. `demo_colmap.py --scene_dir examples/kitchen` 能生成 `sparse/`
3. `demo_colmap.py --scene_dir examples/kitchen --use_ba` 能完整结束
4. 再切换到你自己的 `YOUR_SCENE/images` 数据集

## 7. 哪些代码不需要再补

以下内容不需要再围绕“后处理”单独补代码：

- 不需要把 `vggt/utils/pose_enc.py` 手工接回 `demo_detailed_usage.py`
- 不需要把 `vggt/utils/geometry.py` 单独写成自己的主程序
- 不需要自己再实现 COLMAP 导出
- 不需要自己再实现一份点云可视化主流程

因为这些能力已经在两个 demo 中串好了。