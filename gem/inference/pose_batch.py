"""复用模型的图片批处理、独立执行目录及可核对的性能统计。"""

import json
import math
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_inputs(input_path, command, recursive, output_root):
    """生成稳定排序的输入快照，排除位于输入内部的输出子目录。"""
    source, output = Path(input_path).resolve(), Path(output_root).resolve()
    suffixes = IMAGE_SUFFIXES if command == "infer" else {".pt"}
    if source.is_file():
        if source.suffix.lower() not in suffixes:
            raise ValueError(f"输入文件后缀不受支持：{source.name}")
        return source.parent, [source]
    if not source.is_dir():
        raise ValueError(f"输入路径不存在：{source}")
    files = []
    if output == source:
        raise ValueError("输出根目录不能与输入目录相同，请选择独立目录或子目录")
    for directory, dirs, names in os.walk(source):
        base = Path(directory)
        dirs[:] = sorted(
            d
            for d in dirs
            if not (output.is_relative_to(source) and (base / d).resolve().is_relative_to(output))
        )
        files.extend(base / name for name in names if Path(name).suffix.lower() in suffixes)
        if not recursive:
            break
    files = sorted(files, key=lambda p: p.relative_to(source).as_posix())
    if not files:
        raise ValueError("输入目录中没有符合条件的文件")
    return source, files


def create_run_directory(output_root, command, now=None):
    """原子分配毫秒时间戳目录，同一时刻执行也不覆盖旧结果。"""
    now = now or datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond//1000:03d}"
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        path = root / (f"{command}_{stamp}" + (f"_{index:03d}" if index else ""))
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def _atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, data):
    _atomic_write(
        path,
        lambda p: p.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        ),
    )


def _write_report(path, data):
    from .pose_report import format_report

    _atomic_write(path, lambda p: p.write_text(format_report(data), encoding="utf-8"))


def _pixel_rate(records, seconds):
    """成功像素作为分子；数据缺失时避免生成部分样本的误导性比率。"""
    counts = [r.get("image", {}).get("pixel_count") for r in records if r["status"] == "success"]
    if seconds <= 0 or not counts or any(count is None for count in counts):
        return None
    return sum(counts) / 1_000_000 / seconds


def _percentile(values, fraction):
    values = sorted(values)
    at = (len(values) - 1) * fraction
    low, high = math.floor(at), math.ceil(at)
    return values[low] + (values[high] - values[low]) * (at - low)


def summarize_records(records, initialization_seconds, total_seconds):
    """分开统计初始化、首张、后续成功耗时及包含失败成本的吞吐率。"""
    good = [r for r in records if r["status"] == "success"]
    later = records[1:]
    durations = [r["total_seconds"] for r in later if r["status"] == "success"]
    steady_seconds = sum(r["total_seconds"] for r in later)
    steady_count = sum(r["status"] == "success" for r in later)
    return {
        "attempted_count": len(records),
        "success_count": len(good),
        "failed_count": sum(r["status"] == "failed" for r in records),
        "no_person_count": sum(r["status"] == "no_person" for r in records),
        "initialization_seconds": initialization_seconds,
        "total_seconds": total_seconds,
        "first_item_seconds": records[0]["total_seconds"] if records else None,
        "subsequent_success_mean_seconds": mean(durations) if durations else None,
        "subsequent_success_p50_seconds": _percentile(durations, 0.5) if durations else None,
        "subsequent_success_p95_seconds": _percentile(durations, 0.95) if durations else None,
        "end_to_end_success_images_per_second": (
            len(good) / total_seconds if total_seconds > 0 else None
        ),
        "steady_success_images_per_second": (
            steady_count / steady_seconds if steady_seconds > 0 else None
        ),
        "end_to_end_success_megapixels_per_second": _pixel_rate(records, total_seconds),
        "steady_success_megapixels_per_second": _pixel_rate(later, steady_seconds),
    }


def _sync(device):
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@contextmanager
def _measure(timings, stage, device):
    _sync(device)
    start = perf_counter()
    try:
        yield
    finally:
        _sync(device)
        timings[stage] = perf_counter() - start


class NoPersonError(ValueError):
    """标记正常完成检测但没有可用人物的图片。"""


class BodyImageRunner:
    """持有一次加载的检测器、预测器和骨架导出器。"""

    def __init__(self, args):
        from gem.common.detectors import YOLOXONNX

        from .inputs import load_mapping
        from .pose import load_predictor
        from .pose_json import BodyPoseExporter

        self.device = args.device
        self.detector = YOLOXONNX(args.body_detector, device=args.detector_device)
        self.predictor, self.release = load_predictor(
            args.checkpoint,
            task="body",
            release="body",
            soma_assets=args.soma_assets,
            device=args.device,
        )
        self.exporter = BodyPoseExporter(args.soma_assets, args.device, self.predictor.geometry)
        self.intrinsics = None
        if args.intrinsics:
            import numpy as np

            self.intrinsics = load_mapping(args.intrinsics).get("intrinsics")
            a = np.asarray(self.intrinsics, dtype=float)
            if (
                a.shape != (3, 3)
                or not np.isfinite(a).all()
                or a[0, 0] <= 0
                or a[1, 1] <= 0
                or not np.allclose(a[2], [0, 0, 1])
            ):
                raise ValueError("相机内参必须是焦距为正且数值有限的 3×3 矩阵")

    def predict(self, path, timings, image):
        import cv2

        from gem.common.detectors import detect_track

        from .inputs import pose_intrinsics
        from .pose import run_pose

        with _measure(timings, "detection_seconds", self.device):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"无法读取图片：{path.name}")
            height, width = bgr.shape[:2]
            image.update(width=width, height=height, pixel_count=width * height)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                tracks = detect_track([rgb], self.detector)
            except ValueError as exc:
                if "No person track was detected" in str(exc):
                    raise NoPersonError("未检测到可用的人物") from exc
                raise
            camera = pose_intrinsics(rgb.shape[1], rgb.shape[0], self.intrinsics)
            annotations = {
                0: {
                    "intrinsics": camera,
                    "instances": [
                        {
                            "track_id": int(tracks.track_ids[0]),
                            "bbox": tracks.boxes_xys[0].tolist(),
                            "bbox_observed": bool(tracks.observed_primary[0]),
                        }
                    ],
                }
            }
        with _measure(timings, "inference_seconds", self.device):
            prediction = run_pose(
                self.predictor, path, annotations, task="body", release=self.release
            )
        return prediction


def _validate_resources(args):
    for name in [
        "SOMA_neutral.npz",
        "SOMA_template_rig.usda",
        "SOMA_procedural_transforms.json",
        "correctives_model.pt",
    ]:
        if not (args.soma_assets / name).is_file():
            raise ValueError(f"SOMA 资产缺失：{args.soma_assets/name}")
    if args.command == "infer":
        for name in ["checkpoint", "body_detector", "intrinsics"]:
            path = getattr(args, name)
            if path is not None and not path.is_file():
                raise ValueError(f"所需文件不存在：{path}")


def run_batch(args):
    """执行批处理并保存可追溯报告；文件级异常不影响后续图片。"""
    overall_start = perf_counter()
    started_at = datetime.now().astimezone()
    try:
        input_root, paths = collect_inputs(
            args.input, args.command, args.recursive, args.output_dir
        )
        _validate_resources(args)
        run_dir = create_run_directory(args.output_dir, args.command, started_at)
    except (OSError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    print(f"本次输出目录：{run_dir}", flush=True)
    records = []
    report = {
        "schema_version": "gemx_pose_run_v1",
        "command": args.command,
        "started_at": started_at.isoformat(),
        "output_directory": str(run_dir),
        "input_count": len(paths),
        "device": args.device,
        "records": records,
    }
    initialization_start = perf_counter()
    try:
        import torch

        from .pose_json import BodyPoseExporter

        device = torch.device(args.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("设备仅支持 cpu 或 cuda[:序号]")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请检查驱动或显式使用 -d cpu")
        if args.command == "infer":
            runner = BodyImageRunner(args)
            exporter = runner.exporter
        else:
            runner = None
            exporter = BodyPoseExporter(args.soma_assets, args.device)
            exporter.initialize()
        _sync(args.device)
    except Exception as exc:
        report["error"] = f"初始化失败：{exc}"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["summary"] = summarize_records(
            [], perf_counter() - initialization_start, perf_counter() - overall_start
        )
        try:
            _write_report(run_dir / "report.md", report)
        except OSError as write_error:
            print(f"无法保存执行报告：{write_error}", file=sys.stderr)
        print(report["error"], file=sys.stderr)
        print(f"本次输出目录：{run_dir}", flush=True)
        return 2
    initialization_seconds = perf_counter() - initialization_start
    exit_code = 0
    for index, path in enumerate(paths):
        relative = path.relative_to(input_root)
        output_base = run_dir / relative
        row = {
            "input": str(path),
            "relative_path": relative.as_posix(),
            "status": "success",
            "timings": {},
            "outputs": {},
            "input_size_bytes": None,
            "input_format": path.suffix.lstrip(".").upper(),
            "image": {"width": None, "height": None, "pixel_count": None},
        }
        start = perf_counter()
        try:
            row["input_size_bytes"] = path.stat().st_size
            if runner is not None:
                prediction = runner.predict(path, row["timings"], row["image"])
            else:
                with _measure(row["timings"], "load_seconds", args.device):
                    prediction = torch.load(path, map_location="cpu", weights_only=False)
            with _measure(row["timings"], "export_seconds", args.device):
                document = exporter.export(prediction)
                if runner is None:
                    width, height = document["source"]["image_size"]
                    row["image"].update(width=width, height=height, pixel_count=width * height)
            with _measure(row["timings"], "save_seconds", args.device):
                json_path = output_base.with_name(output_base.name + ".pose.json")
                _write_json(json_path, document)
                row["outputs"]["json"] = str(json_path.relative_to(run_dir))
                if runner is not None:
                    pt_path = output_base.with_name(output_base.name + ".pt")
                    _atomic_write(pt_path, lambda p, data=prediction: torch.save(data, p))
                    row["outputs"]["pt"] = str(pt_path.relative_to(run_dir))
        except Exception as exc:
            row["status"] = "no_person" if isinstance(exc, NoPersonError) else "failed"
            row["error"] = f"处理失败：{exc}"
            exit_code = 1
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                report["error"] = "内存或显存不足，已停止处理；不会自动切换设备"
                row["fatal"] = True
        row["total_seconds"] = perf_counter() - start
        records.append(row)
        labels = {"success": "成功", "no_person": "未检出人物", "failed": "失败"}
        print(
            f'[{index+1}/{len(paths)}] {relative}：{labels[row["status"]]}，'
            f'{row["total_seconds"]:.3f} 秒' + ("（含首次运行开销）" if index == 0 else ""),
            flush=True,
        )
        if row.get("error"):
            print(row["error"], file=sys.stderr)
        if row.get("fatal"):
            break
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["unprocessed_count"] = len(paths) - len(records)
    report["summary"] = summarize_records(
        records, initialization_seconds, perf_counter() - overall_start
    )
    try:
        _write_report(run_dir / "report.md", report)
    except OSError as exc:
        print(f"无法保存执行报告：{exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    average = summary["subsequent_success_mean_seconds"]
    steady_rate = summary["steady_success_images_per_second"]
    print(
        "排除首张后的平均耗时："
        + (f"{average:.3f} 秒/张" if average is not None else "样本不足")
        + "；持续成功吞吐率："
        + (f"{steady_rate:.3f} 张/秒" if steady_rate is not None else "样本不足"),
        flush=True,
    )
    print(
        f'完成：成功 {summary["success_count"]}，未检出 {summary["no_person_count"]}，'
        f'失败 {summary["failed_count"]}。本次输出目录：{run_dir}',
        flush=True,
    )
    return exit_code
