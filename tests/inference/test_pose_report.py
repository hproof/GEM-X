"""验证供人阅读的 Markdown 报告及异常信息。"""

from gem.inference.pose_batch import summarize_records
from gem.inference.pose_report import format_report


def test_markdown_report_has_metadata_stages_outputs_and_escaped_errors():
    records = [
        {
            "relative_path": "人物|图.png",
            "status": "failed",
            "input_size_bytes": 1024,
            "input_format": "PNG",
            "image": {"width": 320, "height": 240, "pixel_count": 76800},
            "timings": {"detection_seconds": 0.25},
            "total_seconds": 0.25,
            "outputs": {},
            "error": "读取|失败\n下一行",
        },
        {
            "relative_path": "ok.jpg",
            "status": "success",
            "input_size_bytes": 2048,
            "input_format": "JPG",
            "image": {"width": 640, "height": 480, "pixel_count": 307200},
            "timings": {"detection_seconds": 0.4, "inference_seconds": 0.5},
            "total_seconds": 1.0,
            "outputs": {"pt": "ok.jpg.pt", "json": "ok.jpg.pose.json"},
        },
    ]
    report = {
        "command": "infer",
        "started_at": "2026-10-30T14:00:00+08:00",
        "finished_at": "2026-10-30T14:00:04+08:00",
        "output_directory": "/输出/本次",
        "input_count": 2,
        "device": "cuda",
        "records": records,
        "summary": summarize_records(records, 2, 4),
        "unprocessed_count": 0,
    }
    text = format_report(report)
    assert text.startswith("# 图片姿态处理报告\n")
    assert "人物\\|图.png" in text and "读取\\|失败<br>下一行" in text
    assert "640 × 480" in text and "| 2048 |" in text
    assert "排除首张后的平均耗时" in text
    assert "张/秒" in text and "百万像素/秒" in text
    assert "0.250" in text and "0.500" in text
    assert "ok.jpg.pt" in text and "ok.jpg.pose.json" in text
    assert "首次运行开销" in text


def test_export_report_distinguishes_pt_size_from_original_image_size():
    records = [
        {
            "relative_path": "body.pt",
            "status": "success",
            "input_size_bytes": 4096,
            "input_format": "PT",
            "image": {"width": 800, "height": 600, "pixel_count": 480000},
            "timings": {"load_seconds": 0.1},
            "total_seconds": 1.0,
            "outputs": {},
        }
    ]
    text = format_report(
        {
            "command": "export",
            "started_at": "现在",
            "output_directory": "/输出",
            "device": "cpu",
            "input_count": 1,
            "records": records,
            "summary": summarize_records(records, 2, 3),
        }
    )
    assert "800 × 600" in text and "| 4096 |" in text
    assert "字节数为输入 .pt 文件大小" in text
    assert "不适用" in text
