#!/usr/bin/env python3
"""轻量级实验追踪工具——JSON 日志 + 产物可追溯，不依赖外网服务"""

import json, os, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

TRACKER_FILE = "experiments/tracker.json"


def init_tracker():
    """初始化追踪文件"""
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    if not os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "w") as f:
            json.dump({"runs": [], "created": str(datetime.now())}, f, indent=2)


def log_run(experiment: str, inputs: Dict[str, str],
            outputs: Dict[str, str], metrics: Optional[Dict] = None,
            notes: str = "", duration_sec: float = 0):
    """记录一次实验运行"""
    init_tracker()
    entry = {
        "experiment": experiment,
        "timestamp": str(datetime.now()),
        "duration_sec": round(duration_sec, 1),
        "inputs": inputs,
        "outputs": outputs,
        "metrics": metrics or {},
        "notes": notes,
    }
    with open(TRACKER_FILE, "r") as f:
        tracker = json.load(f)
    tracker["runs"].append(entry)
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2)


def list_runs(experiment: Optional[str] = None, limit: int = 10):
    """列出历史实验运行"""
    if not os.path.exists(TRACKER_FILE):
        return []
    with open(TRACKER_FILE, "r") as f:
        tracker = json.load(f)
    runs = tracker["runs"]
    if experiment:
        runs = [r for r in runs if r["experiment"] == experiment]
    for r in runs[-limit:]:
        metrics_str = ", ".join(f"{k}={v}" for k, v in r.get("metrics", {}).items())
        print(f"[{r['timestamp'][:19]}] {r['experiment']:<25} "
              f"{r['duration_sec']:>6.1f}s | {metrics_str}")


if __name__ == "__main__":
    list_runs()
