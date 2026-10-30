"""验证便捷命令行入口及无需加载模型的帮助。"""

import subprocess
import sys
from pathlib import Path

import pytest

from gem.inference.pose_image_cli import build_parser, main


def test_short_options_and_subcommand_aliases():
    p = build_parser()
    a = p.parse_args(
        [
            "i",
            "-i",
            "图片",
            "-o",
            "结果",
            "-r",
            "-c",
            "权重",
            "-s",
            "资产",
            "-b",
            "检测器",
            "-d",
            "cpu",
            "-D",
            "cpu",
            "-k",
            "内参",
        ]
    )
    b = p.parse_args(
        [
            "infer",
            "--input",
            "图片",
            "--output-dir",
            "结果",
            "--recursive",
            "--checkpoint",
            "权重",
            "--soma-assets",
            "资产",
            "--body-detector",
            "检测器",
            "--device",
            "cpu",
            "--detector-device",
            "cpu",
            "--intrinsics",
            "内参",
        ]
    )
    assert vars(a) == vars(b)
    assert p.parse_args(["e", "-i", "a.pt"]).command == "export"
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["i", "--inp", "图片"])
    assert exc.value.code == 2


def test_help_does_not_load_inference_libraries():
    code = """
import builtins
original=builtins.__import__
def blocked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','cv2','soma','transformers','scipy','numpy'}:
        raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=blocked
from gem.inference.pose_image_cli import main
main(['infer','--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout and "--body-detector" in result.stdout


def test_invalid_input_creates_no_execution_directory(tmp_path):
    out = tmp_path / "结果"
    assert main(["i", "-i", str(tmp_path / "不存在.jpg"), "-o", str(out), "-d", "cpu"]) == 2
    assert not out.exists()


def test_wrapper_runs_from_another_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/pose_image.py"
    result = subprocess.run(
        [sys.executable, str(script), "-h"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "infer" in result.stdout and "export" in result.stdout


def test_missing_option_value_is_reported_in_chinese(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["infer", "-i"])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "需要一个参数值" in error
    assert "expected one argument" not in error
