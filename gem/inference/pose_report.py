"""将运行数据直接排版为供人阅读的中文 Markdown 报告。"""


def _cell(value):
    """避免文件名或异常消息破坏表格结构。"""
    if value is None:
        return "—"
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def format_report(report):
    """只依赖运行记录生成报告，不调用模型或外部服务。"""
    summary, records = report["summary"], report["records"]
    infer = report["command"] == "infer"
    text = [
        "# 图片姿态处理报告",
        _table(
            ["运行信息", "内容"],
            [
                ["任务", "图片识别" if infer else "已有姿态导出"],
                ["开始时间", report["started_at"]],
                ["结束时间", report.get("finished_at")],
                ["计算设备", report["device"]],
                ["输出目录", report["output_directory"]],
            ],
        ),
        "## 处理汇总",
        _table(
            ["项目", "数量"],
            [
                ["输入", report["input_count"]],
                ["已尝试", summary["attempted_count"]],
                ["成功", summary["success_count"]],
                ["未检出人物", summary["no_person_count"]],
                ["失败", summary["failed_count"]],
                ["未处理", report.get("unprocessed_count", report["input_count"] - len(records))],
            ],
        ),
        "## 耗时与吞吐率",
        _table(
            ["指标", "数值", "单位"],
            [
                ["资源初始化", _number(summary["initialization_seconds"]), "秒"],
                ["首张（含首次运行开销）", _number(summary["first_item_seconds"]), "秒"],
                [
                    "排除首张后的平均耗时",
                    _number(summary["subsequent_success_mean_seconds"]),
                    "秒/张",
                ],
                [
                    "排除首张后的耗时中位数",
                    _number(summary["subsequent_success_p50_seconds"]),
                    "秒/张",
                ],
                [
                    "排除首张后的耗时第 95 百分位",
                    _number(summary["subsequent_success_p95_seconds"]),
                    "秒/张",
                ],
                ["总耗时（含初始化）", _number(summary["total_seconds"]), "秒"],
                [
                    "持续成功吞吐率（排除首张）",
                    _number(summary["steady_success_images_per_second"]),
                    "张/秒",
                ],
                [
                    "端到端成功吞吐率",
                    _number(summary["end_to_end_success_images_per_second"]),
                    "张/秒",
                ],
                [
                    "持续成功像素吞吐率（排除首张）",
                    (
                        _number(summary.get("steady_success_megapixels_per_second"))
                        if infer
                        else "不适用"
                    ),
                    "百万像素/秒",
                ],
                [
                    "端到端成功像素吞吐率",
                    (
                        _number(summary.get("end_to_end_success_megapixels_per_second"))
                        if infer
                        else "不适用"
                    ),
                    "百万像素/秒",
                ],
            ],
        ),
        "首张指第一项实际尝试，资源初始化单独计时；首张可能包含延迟初始化和预热。"
        "后续平均耗时和分位数仅统计首张之后的成功项。持续吞吐率的分母包含后续失败和未检出的处理时间，"
        "分子仅计成功项；端到端指标还包含初始化和首张。— 表示无法读取、样本不足或该阶段未执行。",
        "张/秒是主要处理能力指标；百万像素/秒按原图像素数统计，仅供辅助比较。"
        "模型会将输入缩放或裁剪为固定尺寸，像素吞吐率不代表网络计算量。",
        "## 输入文件",
    ]
    inputs, timings, outputs, errors = [], [], [], []
    labels = {"success": "成功", "no_person": "未检出人物", "failed": "失败"}
    for record in records:
        image = record.get("image", {})
        width, height = image.get("width"), image.get("height")
        dimensions = f"{width} × {height}" if width is not None and height is not None else None
        name = record["relative_path"]
        inputs.append(
            [
                name,
                record.get("input_format"),
                dimensions,
                image.get("pixel_count"),
                record.get("input_size_bytes"),
                labels[record["status"]],
            ]
        )
        stages = record.get("timings", {})
        timings.append(
            [name]
            + [
                _number(stages.get(key))
                for key in [
                    "load_seconds",
                    "detection_seconds",
                    "inference_seconds",
                    "export_seconds",
                    "save_seconds",
                ]
            ]
            + [_number(record["total_seconds"])]
        )
        if record.get("outputs"):
            saved = record["outputs"]
            outputs.append([name, saved.get("pt"), saved.get("json")])
        if record.get("error"):
            errors.append([name, record["error"]])
    text.append(
        _table(["文件", "格式（后缀）", "宽 × 高（像素）", "像素数", "输入字节数", "状态"], inputs)
    )
    if not infer:
        text.append(
            "字节数为输入 .pt 文件大小；图片尺寸来自预测记录，原始图片文件字节数不在预测中。独立导出不读取原图，像素吞吐率不适用，张/秒表示姿态文件导出速度。"
        )
    text.extend(
        [
            "## 分项耗时",
            _table(
                [
                    "文件",
                    "读取预测（秒）",
                    "读图及检测（秒）",
                    "姿态推理（秒）",
                    "骨架导出（秒）",
                    "保存（秒）",
                    "总耗时（秒）",
                ],
                timings,
            ),
            "计时使用单调时钟，GPU 阶段边界显式同步。总耗时截止报告写入前，包含输入枚举和资源检查，不包含最终报告写盘；分项之外的少量调度开销计入每项总耗时。",
        ]
    )
    if outputs:
        text.extend(["## 输出文件", _table(["输入文件", "原生预测", "姿态 JSON"], outputs)])
    if report.get("error"):
        errors.insert(0, ["本次任务", report["error"]])
    if errors:
        text.extend(["## 错误记录", _table(["文件或任务", "原因"], errors)])
    return "\n\n".join(text) + "\n"
