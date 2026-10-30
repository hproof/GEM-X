"""验证批处理目录隔离、输入发现和吞吐率计算。"""

from datetime import datetime, timezone

import pytest

from gem.inference.pose_batch import collect_inputs, create_run_directory, summarize_records


def test_runs_never_reuse_a_directory_even_at_identical_timestamps(tmp_path):
    now = datetime(2026, 10, 30, 15, 30, 12, 123000, tzinfo=timezone.utc)
    first = create_run_directory(tmp_path, "infer", now)
    (first / "保留.txt").write_text("旧结果", encoding="utf-8")
    second = create_run_directory(tmp_path, "infer", now)
    assert first.name == "infer_20261030_153012_123"
    assert second.name == "infer_20261030_153012_123_001"
    assert (first / "保留.txt").read_text(encoding="utf-8") == "旧结果"
    assert list(second.iterdir()) == []


def test_discovery_handles_extensions_recursion_and_nested_output(tmp_path):
    for name in ["person.jpg", "person.PNG", "忽略.txt", "nested/other.webp", "out/old.jpg"]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    root, flat = collect_inputs(tmp_path, "infer", False, tmp_path / "out")
    assert root == tmp_path
    assert [p.name for p in flat] == ["person.PNG", "person.jpg"]
    _, nested = collect_inputs(tmp_path, "infer", True, tmp_path / "out")
    assert [str(p.relative_to(root)).replace("\\", "/") for p in nested] == [
        "nested/other.webp",
        "person.PNG",
        "person.jpg",
    ]
    with pytest.raises(ValueError):
        collect_inputs(tmp_path / "忽略.txt", "infer", False, tmp_path / "out")


def test_metrics_exclude_first_attempt_but_include_failed_attempt_cost():
    rows = [
        {"status": "success", "total_seconds": 10.0},
        {"status": "success", "total_seconds": 2.0},
        {"status": "no_person", "total_seconds": 1.0},
        {"status": "failed", "total_seconds": 1.0},
        {"status": "success", "total_seconds": 4.0},
    ]
    report = summarize_records(rows, initialization_seconds=5.0, total_seconds=24.0)
    assert report["success_count"] == 3
    assert report["no_person_count"] == 1
    assert report["failed_count"] == 1
    assert report["steady_success_images_per_second"] == 0.25
    assert report["end_to_end_success_images_per_second"] == 0.125
    assert report["subsequent_success_mean_seconds"] == 3.0
    one = summarize_records(rows[:1], initialization_seconds=5, total_seconds=15)
    assert one["steady_success_images_per_second"] is None
    assert one["subsequent_success_mean_seconds"] is None


def test_single_input_and_empty_directory_are_distinguished(tmp_path):
    with pytest.raises(ValueError, match="没有"):
        collect_inputs(tmp_path, "infer", False, tmp_path / "out")
    p = tmp_path / "人物.PT"
    p.touch()
    root, files = collect_inputs(p, "export", False, tmp_path / "out")
    assert root == tmp_path and files == [p]


def test_output_ancestor_does_not_hide_nested_inputs(tmp_path):
    source = tmp_path / "inputs"
    p = source / "nested/a.jpg"
    p.parent.mkdir(parents=True)
    p.touch()
    _, files = collect_inputs(source, "infer", True, tmp_path)
    assert files == [p]


def test_directory_run_keeps_all_outputs_together_and_continues_after_errors(
    tmp_path, monkeypatch, capsys
):
    import argparse
    import json

    from gem.inference import pose_batch

    source = tmp_path / "inputs"
    source.mkdir()
    for name in ["a.jpg", "a.png", "broken.jpg", "empty.jpg"]:
        (source / name).write_bytes(b"12345")
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in [
        "SOMA_neutral.npz",
        "SOMA_template_rig.usda",
        "SOMA_procedural_transforms.json",
        "correctives_model.pt",
    ]:
        (assets / name).touch()
    weight = tmp_path / "weight"
    weight.touch()

    class Exporter:
        def export(self, prediction):
            return {"source": prediction["source"]}

    class Runner:
        loads = 0

        def __init__(self, args):
            Runner.loads += 1
            self.exporter = Exporter()

        def predict(self, path, timings, image):
            if path.stem == "broken":
                raise ValueError("图片损坏")
            image.update(width=640, height=480, pixel_count=307200)
            if path.stem == "empty":
                raise pose_batch.NoPersonError("未检出人物")
            return {"source": path.name}

    monkeypatch.setattr(pose_batch, "BodyImageRunner", Runner)
    args = argparse.Namespace(
        command="infer",
        input=source,
        output_dir=tmp_path / "out",
        recursive=False,
        soma_assets=assets,
        checkpoint=weight,
        body_detector=weight,
        intrinsics=None,
        device="cpu",
    )
    assert pose_batch.run_batch(args) == 1
    runs = list(args.output_dir.iterdir())
    assert len(runs) == 1
    assert {p.name for p in runs[0].iterdir()} == {
        "a.jpg.pt",
        "a.jpg.pose.json",
        "a.png.pt",
        "a.png.pose.json",
        "report.md",
    }
    report = (runs[0] / "report.md").read_text(encoding="utf-8")
    assert "| 成功 | 2 |" in report
    assert "| 失败 | 1 |" in report
    assert "| 未检出人物 | 1 |" in report
    assert "| 未处理 | 0 |" in report
    assert "640 × 480" in report and "307,200" in report
    assert "| 5 |" in report
    assert "图片损坏" in report and "未检出人物" in report
    assert "排除首张后的平均耗时" in capsys.readouterr().out
    assert Runner.loads == 1
    assert json.loads((runs[0] / "a.png.pose.json").read_text())["source"] == "a.png"


def test_report_survives_memory_failure_and_leaves_remaining_files_unprocessed(
    tmp_path, monkeypatch
):
    import argparse

    import torch

    from gem.inference import pose_batch

    source = tmp_path / "inputs"
    source.mkdir()
    for name in ["a.pt", "b.pt"]:
        torch.save({}, source / name)
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in [
        "SOMA_neutral.npz",
        "SOMA_template_rig.usda",
        "SOMA_procedural_transforms.json",
        "correctives_model.pt",
    ]:
        (assets / name).touch()

    class Exporter:
        def __init__(self, *args):
            pass

        def initialize(self):
            pass

        def export(self, prediction):
            raise torch.cuda.OutOfMemoryError("测试显存不足")

    monkeypatch.setattr("gem.inference.pose_json.BodyPoseExporter", Exporter)
    args = argparse.Namespace(
        command="export",
        input=source,
        output_dir=tmp_path / "out",
        recursive=False,
        soma_assets=assets,
        device="cpu",
    )
    assert pose_batch.run_batch(args) == 1
    report = next(args.output_dir.glob("*/report.md")).read_text(encoding="utf-8")
    assert "| 未处理 | 1 |" in report
    assert "显存不足" in report
    assert "a.pt" in report and "b.pt" not in report


def test_export_initialization_failure_does_not_attempt_each_file(tmp_path, monkeypatch):
    import argparse

    from gem.inference import pose_batch

    source = tmp_path / "inputs"
    source.mkdir()
    for name in ["a.pt", "b.pt"]:
        (source / name).touch()
    monkeypatch.setattr(pose_batch, "_validate_resources", lambda args: None)

    class Exporter:
        def __init__(self, *args):
            pass

        def initialize(self):
            raise ValueError("SOMA 资产损坏")

        def export(self, prediction):
            raise AssertionError("初始化失败时不应处理文件")

    monkeypatch.setattr("gem.inference.pose_json.BodyPoseExporter", Exporter)
    args = argparse.Namespace(
        command="export",
        input=source,
        output_dir=tmp_path / "out",
        recursive=False,
        soma_assets=tmp_path / "assets",
        device="cpu",
    )
    assert pose_batch.run_batch(args) == 2
    report = next(args.output_dir.glob("*/report.md")).read_text(encoding="utf-8")
    assert "初始化失败" in report and "资产损坏" in report
    assert "| 已尝试 | 0 |" in report
    assert "| 未处理 | 2 |" in report


def test_pixel_throughput_excludes_first_image_and_failed_pixels_but_includes_failure_time():
    rows = [
        {"status": "success", "total_seconds": 10, "image": {"pixel_count": 2000000}},
        {"status": "success", "total_seconds": 2, "image": {"pixel_count": 1000000}},
        {"status": "no_person", "total_seconds": 1, "image": {"pixel_count": 5000000}},
        {"status": "success", "total_seconds": 4, "image": {"pixel_count": 3000000}},
    ]
    summary = summarize_records(rows, initialization_seconds=3, total_seconds=20)
    assert summary["end_to_end_success_megapixels_per_second"] == pytest.approx(0.3)
    assert summary["steady_success_megapixels_per_second"] == pytest.approx(4 / 7)
    assert summary["steady_success_images_per_second"] == pytest.approx(2 / 7)
    assert summarize_records(rows[:1], 3, 13)["steady_success_megapixels_per_second"] is None
    rows[-1]["image"]["pixel_count"] = None
    assert summarize_records(rows, 3, 20)["steady_success_megapixels_per_second"] is None


def test_image_dimensions_are_kept_when_detection_finds_no_person(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from gem.inference.pose_batch import BodyImageRunner, NoPersonError

    path = tmp_path / "blank.png"
    assert cv2.imwrite(str(path), np.zeros((45, 80, 3), dtype=np.uint8))

    def no_person(*args):
        raise ValueError("No person track was detected")

    monkeypatch.setattr("gem.common.detectors.detect_track", no_person)
    runner = BodyImageRunner.__new__(BodyImageRunner)
    runner.device = "cpu"
    runner.detector = object()
    image = {}
    with pytest.raises(NoPersonError):
        runner.predict(path, {}, image)
    assert image == {"width": 80, "height": 45, "pixel_count": 3600}
