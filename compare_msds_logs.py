#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


EPOCH_PATTERN = re.compile(r"INFO\s+(\d+)/(\d+)\s+starting")
TEST_PATTERN = re.compile(
    r"\[test\]\s+pr:([0-9.]+)\s+rc:([0-9.]+)\s+auc:([0-9.]+)\s+ap:([0-9.]+)\s+f1:([0-9.]+)(?:\s+thr:([0-9.]+|argmax))?"
)
EARLY_STOP_PATTERN = re.compile(r"Early stop at epoch[: ]+(\d+)")


def _safe_mean(values):
    if not values:
        return float("nan")
    return float(statistics.mean(values))


def _safe_std(values):
    if len(values) < 2:
        return 0.0 if values else float("nan")
    return float(statistics.pstdev(values))


def parse_running_log(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    current_epoch = None
    max_started_epoch = -1
    early_stop_epoch = None
    test_records = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                max_started_epoch = max(max_started_epoch, current_epoch)

            early_match = EARLY_STOP_PATTERN.search(line)
            if early_match:
                early_stop_epoch = int(early_match.group(1))

            test_match = TEST_PATTERN.search(line)
            if test_match and current_epoch is not None:
                pr, rc, auc, ap, f1, threshold = test_match.groups()
                threshold_value = None
                if threshold and threshold != "argmax":
                    threshold_value = float(threshold)
                test_records.append(
                    {
                        "epoch": current_epoch,
                        "pr": float(pr),
                        "rc": float(rc),
                        "auc": float(auc),
                        "ap": float(ap),
                        "f1": float(f1),
                        "threshold": threshold_value,
                    }
                )

    if not test_records:
        raise RuntimeError(f"No [test] metrics found in {path}")

    return {
        "path": str(path.resolve()),
        "max_started_epoch": max_started_epoch,
        "early_stop_epoch": early_stop_epoch,
        "stop_epoch": early_stop_epoch if early_stop_epoch is not None else max_started_epoch,
        "test_records": test_records,
    }


def summarize(parsed, start_epoch):
    records = parsed["test_records"]
    f1_all = [x["f1"] for x in records]
    post = [x["f1"] for x in records if x["epoch"] >= start_epoch]
    thresholds = [x["threshold"] for x in records if x["threshold"] is not None]

    best = max(records, key=lambda x: x["f1"])
    summary = {
        "log_path": parsed["path"],
        "eval_count": len(records),
        "stop_epoch": parsed["stop_epoch"],
        "best_f1": float(best["f1"]),
        "best_epoch": int(best["epoch"]),
        "overall_mean_f1": _safe_mean(f1_all),
        "overall_std_f1": _safe_std(f1_all),
        "post_start_epoch": int(start_epoch),
        "post_count": len(post),
        "post_mean_f1": _safe_mean(post),
        "post_std_f1": _safe_std(post),
        "post_min_f1": float(min(post)) if post else float("nan"),
        "post_max_f1": float(max(post)) if post else float("nan"),
        "threshold_mean": _safe_mean(thresholds),
        "threshold_std": _safe_std(thresholds),
    }
    return summary


def compare(base, improved):
    return {
        "delta_best_f1": improved["best_f1"] - base["best_f1"],
        "delta_post_mean_f1": improved["post_mean_f1"] - base["post_mean_f1"],
        "delta_post_std_f1": improved["post_std_f1"] - base["post_std_f1"],
        "delta_stop_epoch": improved["stop_epoch"] - base["stop_epoch"],
    }


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def print_report(base_summary, improved_summary, delta):
    print("=== Baseline ===")
    for key in [
        "log_path",
        "eval_count",
        "stop_epoch",
        "best_f1",
        "best_epoch",
        "post_mean_f1",
        "post_std_f1",
        "post_min_f1",
        "post_max_f1",
        "threshold_mean",
        "threshold_std",
    ]:
        print(f"{key}: {_fmt(base_summary[key])}")
    print("")

    print("=== Improved ===")
    for key in [
        "log_path",
        "eval_count",
        "stop_epoch",
        "best_f1",
        "best_epoch",
        "post_mean_f1",
        "post_std_f1",
        "post_min_f1",
        "post_max_f1",
        "threshold_mean",
        "threshold_std",
    ]:
        print(f"{key}: {_fmt(improved_summary[key])}")
    print("")

    print("=== Delta (Improved - Baseline) ===")
    for key, value in delta.items():
        print(f"{key}: {_fmt(value)}")


def write_csv(path, base_summary, improved_summary, delta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "log_path",
        "eval_count",
        "stop_epoch",
        "best_f1",
        "best_epoch",
        "post_start_epoch",
        "post_count",
        "post_mean_f1",
        "post_std_f1",
        "post_min_f1",
        "post_max_f1",
        "threshold_mean",
        "threshold_std",
        "delta_best_f1",
        "delta_post_mean_f1",
        "delta_post_std_f1",
        "delta_stop_epoch",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"name": "baseline", **base_summary, **delta})
        writer.writerow({"name": "improved", **improved_summary, **delta})


def main():
    parser = argparse.ArgumentParser(description="Compare two MSDS running.log files")
    parser.add_argument("--baseline", required=True, help="path to baseline running.log")
    parser.add_argument("--improved", required=True, help="path to improved running.log")
    parser.add_argument("--start-epoch", type=int, default=20, help="start epoch for post statistics")
    parser.add_argument("--out-json", default=None, help="optional path to save json report")
    parser.add_argument("--out-csv", default=None, help="optional path to save csv report")
    args = parser.parse_args()

    base = summarize(parse_running_log(args.baseline), args.start_epoch)
    improved = summarize(parse_running_log(args.improved), args.start_epoch)
    delta = compare(base, improved)

    print_report(base, improved, delta)

    payload = {"baseline": base, "improved": improved, "delta": delta}
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_csv:
        write_csv(args.out_csv, base, improved, delta)


if __name__ == "__main__":
    main()
