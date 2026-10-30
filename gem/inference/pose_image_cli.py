"""单图、目录识别及独立姿态导出的轻量命令行入口。"""

import argparse
import sys
from pathlib import Path


class ChineseParser(argparse.ArgumentParser):
    """为参数帮助和常见解析错误提供中文说明。"""

    def __init__(self, *args, **kwargs):
        kwargs.update(allow_abbrev=False, add_help=False)
        super().__init__(*args, **kwargs)
        self.add_argument("-h", "--help", action="help", help="显示帮助并退出")
        self._positionals.title = "位置参数"
        self._optionals.title = "选项"

    def format_help(self):
        return super().format_help().replace("usage: ", "用法：")

    def format_usage(self):
        return super().format_usage().replace("usage: ", "用法：")

    def error(self, message):
        for old, new in [
            ("the following arguments are required:", "缺少必需参数："),
            ("unrecognized arguments:", "无法识别的参数："),
            ("invalid choice:", "无效选项："),
            ("choose from", "可选值为"),
            ("expected one argument", "需要一个参数值"),
            ("argument ", "参数 "),
        ]:
            message = message.replace(old, new)
        self.print_usage(sys.stderr)
        self.exit(2, f"参数错误：{message}\n")


def build_parser():
    """建立完整参数和显式短选项，帮助阶段不加载推理依赖。"""
    root = Path(__file__).resolve().parents[2]
    parser = ChineseParser(description="图片或目录生成 Body 姿态，或将已有预测转换为骨架 JSON")
    commands = parser.add_subparsers(dest="command", required=True, title="子命令")
    for command, alias, desc in [
        ("infer", "i", "识别一张图片或一个图片目录"),
        ("export", "e", "从 Body 单图预测导出骨架 JSON"),
    ]:
        sub = commands.add_parser(command, aliases=[alias], help=desc, description=desc)
        sub.set_defaults(command=command)
        sub.add_argument("-i", "--input", required=True, type=Path, help="输入图片、预测文件或目录")
        sub.add_argument(
            "-o",
            "--output-dir",
            type=Path,
            default=root / "outputs/poses",
            help="输出根目录，默认仓库 outputs/poses；每次创建时间戳子目录",
        )
        sub.add_argument("-r", "--recursive", action="store_true", help="递归处理子目录")
        sub.add_argument(
            "-s",
            "--soma-assets",
            type=Path,
            default=root / "assets/soma",
            help="SOMA 0.3.3 资产目录，默认仓库 assets/soma",
        )
        sub.add_argument("-d", "--device", default="cuda", help="推理或几何计算设备，默认 cuda")
        if command == "infer":
            sub.add_argument(
                "-c",
                "--checkpoint",
                type=Path,
                default=root / "weights/body_s240000.ckpt",
                help="Body 模型权重",
            )
            sub.add_argument(
                "-b",
                "--body-detector",
                type=Path,
                default=root / "weights/yolox_humanart.onnx",
                help="人体 ONNX 检测器",
            )
            sub.add_argument("-D", "--detector-device", default="cpu", help="检测器设备，默认 cpu")
            sub.add_argument("-k", "--intrinsics", type=Path, help="可选的相机内参 JSON 或 YAML")
    return parser


def main(argv=None):
    """返回批处理退出码，参数错误由解析器以中文报告。"""
    args = build_parser().parse_args(argv)
    from .pose_batch import run_batch

    return run_batch(args)
