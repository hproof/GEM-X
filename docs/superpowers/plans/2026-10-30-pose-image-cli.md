# 图片姿态命令行工具实现计划

执行方式：在本会话中使用 `superpowers:executing-plans` 按任务执行；关键行为使用测试驱动开发，完成后独立审查代码。

目标：在已有 WSL 推理环境中提供单图、目录识别及已有 `.pt` 导出，产生带时间戳的执行目录、完整骨架 JSON 和性能报告。

架构：新增独立模块处理源骨架重建、JSON 坐标变换、批处理文件及统计；薄脚本只解析命令行。复用既有检测、主人物选择和预测组件，不改变原生推理入口。

技术：Python 3.11、PyTorch 2.10、SOMA-X 0.3.3、NumPy、SciPy、OpenCV、ONNXRuntime、pytest。

设计依据：[已确认设计](../specs/2026-10-30-unity-pose-design.md)。

## 全局约束

- 第一版仅 Body 单图静态姿态，77 个公共关节包含手指，不实现 Unity、服务、FBX 或 Hand 融合。
- 短参数及 `i`、`e` 子命令别名必须可用；禁止长参数前缀自动缩写。
- 每次实际处理创建新的毫秒时间戳目录，冲突使用递增后缀；所有产物写入该目录。
- JSON 协议固定为 `gemx_unity_pose_v1`；单位米，轴向右、上、前，四元数顺序 `[x,y,z,w]`。
- JSON 导出使用真实源参考变换和预测变换；重建关节与 `.pt` 最大位置误差不超过 0.1 毫米。
- 原 `.pt` 格式和现有 `gem-infer` 行为不变；不安装训练或渲染依赖。
- 注释、文档、错误提示和日志使用中文；标识符保持 Python 常规命名。

## 任务一：源骨架与 JSON 导出

文件：新增 `gem/inference/pose_json.py`、`tests/inference/test_pose_json.py`。

接口：

```python
def build_pose_document(prediction, names, parents, reference, posed):
    # 由公共变换构造可序列化的 JSON 字典，并验证输入。
    ...

class BodyPoseExporter:
    def __init__(self, soma_assets, device="cpu", geometry=None):
        ...

    def export(self, prediction):
        # 返回 JSON 字典；可复用现有 BodyGeometry，独立导出不加载识别网络。
        ...
```

- [ ] 先编写可手算的两关节旋转测试，检查坐标反射、父子关系、四元数及异常输入。
- [ ] 运行 `python -m pytest -q tests/inference/test_pose_json.py`，确认新模块缺失导致测试失败。
- [ ] 实现纯数据转换：矩阵使用 `C @ R @ C`，其中 `C = diag(1,-1,1)`；位置使用同一基变换。使用 SciPy 生成归一化四元数。
- [ ] 实现 SOMA 导出：使用已有 Body 几何兼容配置，调用相同的矩阵到轴角转换；准备身份后，对预测旋转和单位关节旋转分别运行公共骨架 FK，读取 `transforms`，去除虚拟根；验证预测关节与已保存关节的一致性。
- [ ] 对真实 `outputs/deployment_check/body.pt` 导出并记录重建误差。对 Hand、视频、多人物、非有限值和错误父子图给出中文错误。
- [ ] 测试通过后提交任务一。

## 任务二：批处理、目录及命令行

文件：新增 `gem/inference/pose_batch.py`、`gem/inference/pose_image_cli.py`、`scripts/pose_image.py`、`tests/inference/test_pose_batch.py`、`tests/inference/test_pose_image_cli.py`。

接口：

```python
def collect_inputs(input_path, command, recursive, output_root):
    # 返回输入基目录和稳定排序的文件快照。
    ...

def create_run_directory(output_root, command, now=None):
    # 原子创建独立执行目录；now 供确定性的目录冲突测试使用。
    ...

def run_batch(args):
    # 初始化一次依赖，串行处理，写报告并返回 0、1 或 2。
    ...

def main(argv=None):
    # 解析中文帮助，延迟导入批处理模块，返回退出码。
    ...
```

- [ ] 编写目录同毫秒冲突、后缀冲突、排除输出子目录、递归和确定性统计测试；编写 CLI 别名和轻量帮助测试。
- [ ] 运行三个新测试文件，确认新增功能缺失时失败。
- [ ] 使用 `mkdir(exist_ok=False)` 分配目录，用原子文件替换保存 JSON 和 `.pt`。不提供旧目录覆盖参数。
- [ ] 复用 `YOLOXONNX`，每张图片重新开始主人物跟踪，保持现有选择规则但复用 ONNX 会话；复用 `load_predictor` 和 `run_pose`。
- [ ] 按阶段计时，在 GPU 计算边界同步；分别统计初始化、首张、持续处理及端到端吞吐率。未检出和坏图片继续处理；显存不足立即保存报告并停止。
- [ ] 实现两个子命令及已确认短参数表，默认资源从仓库目录解析；启动和结束打印执行目录。
- [ ] 独立导出读取可信本地 `.pt`；只导出 JSON，不加载识别权重。
- [ ] 测试通过后提交任务二。

## 任务三：集成验证、文档与部署

文件：新增中文 `docs/POSE_IMAGE_CLI.md`，更新本计划的执行记录。

- [ ] 使用目录混合样本验证正常图片、坏图片、无人物图片、嵌套文件、同名不同后缀和返回码。
- [ ] 在 RTX 3060 上执行真实单图、至少三张有效图片的目录识别、独立导出；比较两种导出的关节及旋转，记录报告路径和吞吐率。
- [ ] 执行新增测试与相关已有推理测试；使用 Ruff 和 Black 检查本任务文件。
- [ ] 请求独立代码审查并修复重要问题；针对变更重新验证。
- [ ] 检查 Windows 和 Linux 目标工作区；将明确的新增文件同步至 `/home/hproof/git-proj/GEM-X`，保持资源和现有代码不变。
- [ ] 在实际部署目录使用默认参数再次执行，确认不依赖开发工作区；文档包含激活环境、单图、目录、导出、协议字段及指标定义。
- [ ] 提交并将完成的功能带回用户工作区，汇报真实测试范围、输出位置和示例命令。

## 执行记录

- 设计已确认；采用独立工作区 `G:\git_proj\GEM-X-pose-cli`，分支 `feat/pose-image-cli`。
- 已核对任务间接口：任务二调用任务一的导出器；任务三只新增文档和验证记录。现有检测辅助函数每次会重新加载检测器，因此新批处理在自己的模块中复用检测会话，再调用相同的跟踪选择函数。
