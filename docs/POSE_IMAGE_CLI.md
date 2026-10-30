# 图片姿态识别与 JSON 导出

入口为 `scripts/pose_image.py`。输入一张图片即可生成身体、双手和手指的完整姿态，使用 Body 模型的 77 个公共关节，每张图片处理一个主要人物。目录处理按图片依次执行，模型、检测器和几何运行时在进程内复用。

## 在现有 WSL 环境中运行

先在 Windows 终端进入 Ubuntu，再激活已部署的环境：

```bash
wsl -d aigc
conda activate gem-x
cd /home/hproof/git-proj/GEM-X
```

识别单张图片：

```bash
python scripts/pose_image.py i -i /path/person.jpg
```

识别整个目录，递归处理子目录，并指定输出根目录：

```bash
python scripts/pose_image.py i -i /path/images -r -o outputs/poses
```

从已有 `.pt` 独立导出 JSON，可输入文件或目录：

```bash
python scripts/pose_image.py e -i /path/body.pt
python scripts/pose_image.py e -i /path/predictions -r -o outputs/exported
```

独立导出仅加载 SOMA 几何运行时，不加载人体检测器、DINOv3 或识别权重。仅接受本项目 Body 单图、单人物的原生预测；不接受独立 Hand 或视频结果。`.pt` 使用 PyTorch 序列化，应使用本地可信的预测文件。

查看中文帮助：

```bash
python scripts/pose_image.py -h
python scripts/pose_image.py i -h
python scripts/pose_image.py e -h
```

从 Windows 直接调用时，可使用以下形式，输入路径须写成 WSL 可访问的路径，例如 Windows 的 `G:\images` 对应 `/mnt/g/images`：

```powershell
wsl -d aigc --cd /home/hproof/git-proj/GEM-X -- /home/hproof/miniconda3/envs/gem-x/bin/python scripts/pose_image.py i -i /mnt/g/images -r
```

## 参数

`infer` 可缩写为 `i`，`export` 可缩写为 `e`。支持明确的短选项，不支持任意截断长选项。

| 短选项 | 长选项 | 含义 |
| --- | --- | --- |
| `-i` | `--input` | 必填，输入图片、预测文件或目录 |
| `-o` | `--output-dir` | 输出根目录，默认仓库下的 `outputs/poses` |
| `-r` | `--recursive` | 递归处理子目录，默认仅处理当前层 |
| `-s` | `--soma-assets` | SOMA 资产目录，默认仓库下的 `assets/soma` |
| `-d` | `--device` | 推理或几何设备，默认 `cuda`，可选 `cpu` 或 `cuda:0` 等 |
| `-c` | `--checkpoint` | 仅识别，默认 `weights/body_s240000.ckpt` |
| `-b` | `--body-detector` | 仅识别，默认 `weights/yolox_humanart.onnx` |
| `-D` | `--detector-device` | 仅识别，人体检测器设备，默认 `cpu` |
| `-k` | `--intrinsics` | 仅识别，可选相机内参 JSON 或 YAML |
| `-h` | `--help` | 显示帮助 |

默认资源从脚本所在仓库解析，不依赖当前工作目录。自定义 `-k` 文件通过 `intrinsics` 字段提供 3×3 矩阵，焦距必须为正，最后一行为 `[0,0,1]`。未提供时沿用项目的相机内参估计。

图片后缀支持 JPG、JPEG、PNG、BMP、WEBP、TIF、TIFF，不区分大小写。目录中的文件按相对路径排序。输入与输出根目录不能完全相同；输出根目录位于输入内部时，会排除该输出子目录。

## 输出目录与错误处理

每次执行新建“命令加本地毫秒时间戳”目录；同毫秒冲突时添加 `_001` 等后缀。程序开始和结束时打印实际目录。

```text
outputs/poses/
└── infer_20261030_123513_298/
    ├── report.md
    ├── person.jpg.pt
    ├── person.jpg.pose.json
    └── nested/
        ├── other.png.pt
        └── other.png.pose.json
```

识别成功后同时保存原生 `.pt` 和 `.pose.json`。独立导出只保存新的 `.pose.json`；例如 `body.pt` 生成 `body.pt.pose.json`。保留原扩展名可避免 `person.jpg` 与 `person.png` 输出冲突。

坏图片、未检出人物或单文件协议错误会记录原因并继续。内存或显存不足时停止，报告保留已处理记录和未处理数量。退出码：全部成功为 `0`，存在文件失败或未检出为 `1`，配置或初始化失败为 `2`。输入或资源缺失在创建执行目录前检查；初始化失败会在已创建目录保存报告。

## 姿态 JSON 协议

协议版本为 `gemx_unity_pose_v1`，包含完整源骨架及其参考姿态。名称中的 Unity 表示预定交换用途；后续接入 Humanoid 仍需根据目标角色骨轴和参考姿态进行重定向。本次交付 Python CLI 和数据，不包含 Unity 脚本、动画片段或资源包。

| 字段 | 含义 |
| --- | --- |
| `source` | 来源图片、尺寸、模型和几何版本 |
| `coordinates` | 米；X 向右、Y 向上、Z 向前；左手坐标基；四元数顺序 `[x,y,z,w]` |
| `camera_translation` | 同一坐标基下的相机平移，供调试使用 |
| `validation.max_reconstruction_error_meters` | SOMA 重建关节与 `.pt` 保存关节的最大位置误差 |
| `joints` | 77 个公共关节，去除 SOMA 虚拟 Root；包含双手及手指 |
| `joints[].name` | 源关节名称 |
| `joints[].parent_index` | 父节点在数组中的索引，Hips 根节点为 `-1` |
| `joints[].reference_world_rotation` | 源参考姿态的全局四元数 |
| `joints[].posed_world_rotation` | 预测姿态的全局四元数 |
| `joints[].reference_root_relative_position` | 相同身份及骨长的参考位置，相对 Hips |
| `joints[].posed_root_relative_position` | 预测位置，相对 Hips |

全局旋转位于共同模型参考空间，不是游戏场景世界空间，也不是某个角色可直接赋值的局部旋转。参考姿态通过相同身份和骨长、单位关节旋转的 SOMA 前向运动学恢复。

几何沿用 SOMA-X `0.3.3`、参考姿态 `0.2.0` 和项目的 `legacy_pose021` 配置。相机坐标转为输出坐标时，位置使用 `C=diag(1,-1,1)`，旋转使用 `C @ R @ C`。导出校验层次无环、旋转合法、四元数归一化及数值有限；重建最大误差超过 0.1 毫米时拒绝导出。

## 性能报告

运行报告由 Python 按固定模板直接生成 `report.md`，使用 UTF-8 中文 Markdown，无需 AI 或网络服务。新执行不再生成 `report.json`；姿态数据仍保存为 `.pose.json`，历史执行文件保留。

报告包含输入文件表、分项耗时表、输出文件表、错误记录及汇总。每个输入记录格式后缀、宽和高、像素数、文件精确字节数及状态。尺寸取实际解码后用于识别的图片，像素数为宽乘高；未检出人物也保留尺寸，坏图无法读取尺寸时显示 `—`，仍记录能读取到的文件字节数。

独立导出时，尺寸从 `.pt` 的预测记录取得，字节数为输入 `.pt` 的大小，不冒充原图大小；此时像素吞吐率显示“不适用”。

| 指标 | 统计口径 |
| --- | --- |
| 资源初始化 | 依赖导入和资源初始化，独立导出也包含 SOMA 加载 |
| 首张耗时 | 第一项实际尝试的处理耗时，可能包含首次计算预热；终端标注“含首次运行开销” |
| 排除首张后的平均耗时 | 首张之后成功文件的平均耗时，终端也显示该值 |
| 后续耗时中位数及第 95 百分位 | 同一成功样本集的耗时分位数 |
| 端到端成功吞吐率（张/秒） | 成功数除以整体耗时，包含初始化和失败成本 |
| 持续成功吞吐率（张/秒） | 排除首张后，成功数除以后续所有尝试耗时，包含坏图和未检出成本 |
| 端到端成功像素吞吐率（百万像素/秒） | 成功图片的原图总像素数除以一百万，再除以整体耗时 |
| 持续成功像素吞吐率（百万像素/秒） | 排除首张后的成功原图总像素数除以一百万，再除以后续所有尝试耗时 |

吞吐率以“张/秒”为主，百万像素/秒用于辅助了解输入规模。检测器将输入缩放至固定尺寸，姿态模型使用固定尺寸的人体裁剪，所以原图像素数与网络计算量不成正比。文件字节数受 PNG、JPEG 等压缩方式影响，不用作主要性能指标。

整体计时截止报告写入之前；命令行解析及最终报告写盘不在该耗时内。GPU 阶段边界显式同步。没有足够样本时报告显示 `—`；只有一张图片时，不报告持续处理性能。

RTX 3060、当前 `gem-x` 环境的功能验证使用三张有效图片，以及一张黑图、一张坏图；三张有效图片来自仓库同一示例海报的不同格式，不能作为精度或长期性能基准。设置 `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`，检测器使用 CPU 时，测得：

- 初始化约 13.48 秒；首张约 6.98 秒；后续两张成功样本平均约 1.10 秒。
- 持续成功吞吐率约 0.64 张/秒，包含黑图与坏图的处理成本；含初始化的端到端约 0.13 张/秒。
- 三张有效图片均输出 77 关节和归一化四元数；重建最大误差约 0.0013 毫米。

上述历史实测报告位于 `outputs/pose_cli_validation/infer_20261030_123513_298/report.json`。实际吞吐率受输入尺寸、CPU 线程数和 GPU 当前负载影响；测试自己的目录时以本次报告为准。

部署到 `/home/hproof/git-proj/GEM-X` 后，已使用默认资源路径再次验证：单图输出位于 `outputs/pose_cli_deployed/infer_20261030_123956_607`，独立导出位于 `outputs/pose_cli_deployed/export_20261030_124114_513`。两种导出的关节名称、层次及旋转一致，位置最大分量差异约 0.001 毫米；独立导出验证期间禁止导入 `transformers`，仍正常完成。

新增测试和相关既有测试共 52 项通过，涵盖 Markdown、图片元数据和像素吞吐率；本次 Python 文件通过 Ruff 和 Black 检查。未运行训练或渲染测试，也未进行 Unity 接入验证。

Markdown 报告已在部署目录使用 `assets/images` 下四张图片实测，全部成功。报告为 `outputs/pose_report_deployed/infer_20261030_133150_809/report.md`；该次排除首张后平均 1.070 秒/张，持续吞吐率 0.934 张/秒、0.424 百万像素/秒。独立导出的报告为 `outputs/pose_report_deployed/export_20261030_133227_218/report.md`。

开发验证命令：

```bash
python -m pytest -q tests/inference/test_pose_json.py tests/inference/test_pose_batch.py tests/inference/test_pose_report.py tests/inference/test_pose_image_cli.py tests/inference/test_body_inputs.py tests/inference/test_entrypoints.py tests/common/test_geometry.py
```
