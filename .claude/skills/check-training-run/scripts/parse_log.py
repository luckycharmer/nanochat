#!/usr/bin/env python3
"""Summarise a nanochat training log and flag likely problems.

Usage:
    python .claude/skills/check-training-run/scripts/parse_log.py LOG [--json] [--csv OUT.csv]

Understands the output of scripts/base_train.py and scripts/chat_sft.py.
Standard library only.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys

GPT2_CORE = 0.256525  # README: GPT-2 (1.6B) CORE score

NUM = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

STEP = re.compile(
    r"^step (\d+)(?:/(\d+))? \(([\d.]+)%\) \| loss: " + NUM +
    r" \| lrm: " + NUM +
    r" \| dt: " + NUM + r"ms \| tok/sec: ([\d,]+) \| (?:bf16_)?mfu: " + NUM +
    r".*?\| total time: " + NUM + r"m(?: \| eta: " + NUM + r"m)?",
    re.I,
)
VAL = re.compile(r"^Step (\d+) \| Validation bpb: " + NUM, re.I)
CORE = re.compile(r"^Step (\d+) \| CORE metric: " + NUM, re.I)
CHATCORE = re.compile(r"^Step (\d+) \| ChatCORE: " + NUM, re.I)
PEAK_MEM = re.compile(r"^Peak memory usage: " + NUM + "MiB")
TOTAL_TIME = re.compile(r"^Total training time: " + NUM + "m")
MIN_VAL = re.compile(r"^Minimum validation bpb: " + NUM)
GPU = re.compile(r"^GPU: (.+?) \| Peak FLOPS")
WARNING = re.compile(r"^WARNING:")
ERROR = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|"
    r"NCCL (?:error|WARN.*timeout)|RuntimeError:|Killed$|ChildFailedError|Segmentation fault",
    re.I,
)


def f(x):
    return float(x)


def new_run(index):
    return {
        "run": index, "kind": None, "gpu": None, "steps": [], "val_bpb": [],
        "core": [], "chatcore": [], "peak_mem_mib": None, "total_time_min": None,
        "min_val_bpb": None, "warnings": [], "errors": [],
    }


def parse(lines):
    runs = [new_run(1)]
    for lineno, raw in enumerate(lines, 1):
        line = ANSI.sub("", raw).strip()
        # torchrun may prefix lines with "[rank0]: " or similar
        line = re.sub(r"^\[(?:rank|default)\d*\]:\s*", "", line)
        run = runs[-1]

        m = STEP.search(line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2)) if m.group(2) else None
            kind = "base_train" if total is not None else "chat_sft"
            # a new run starts when the step counter resets or the log format changes
            if run["steps"] and (step < run["steps"][-1]["step"] or kind != run["kind"]):
                runs.append(new_run(len(runs) + 1))
                run = runs[-1]
            run["kind"] = run["kind"] or kind
            run["steps"].append({
                "step": step, "total": total, "pct": f(m.group(3)), "loss": f(m.group(4)),
                "lrm": f(m.group(5)), "dt_ms": f(m.group(6)),
                "tok_per_sec": int(m.group(7).replace(",", "")), "mfu": f(m.group(8)),
                "total_time_min": f(m.group(9)),
                "eta_min": f(m.group(10)) if m.group(10) else None,
            })
            continue

        for pat, key in ((VAL, "val_bpb"), (CORE, "core"), (CHATCORE, "chatcore")):
            m = pat.search(line)
            if m:
                run[key].append({"step": int(m.group(1)), "value": f(m.group(2))})
                break
        else:
            if (m := PEAK_MEM.search(line)):
                run["peak_mem_mib"] = f(m.group(1))
            elif (m := TOTAL_TIME.search(line)):
                run["total_time_min"] = f(m.group(1))
            elif (m := MIN_VAL.search(line)):
                run["min_val_bpb"] = f(m.group(1))
            elif (m := GPU.search(line)):
                run["gpu"] = m.group(1)
            elif WARNING.search(line):
                if line not in run["warnings"]:
                    run["warnings"].append(line)
            elif ERROR.search(line):
                run["errors"].append(f"line {lineno}: {line[:200]}")
    return [r for r in runs if r["steps"] or r["val_bpb"] or r["errors"]]


def analyse(run, spike_ratio=1.5, drop_ratio=0.5, low_mfu=20.0):
    flags = []
    steps = run["steps"]

    bad = [s["step"] for s in steps if not math.isfinite(s["loss"])]
    if bad:
        flags.append(f"NaN/Inf loss first at step {bad[0]} ({len(bad)} step(s) affected)")

    # spikes: smoothed loss jumps well above the running minimum after warm-up
    finite = [s for s in steps if math.isfinite(s["loss"])]
    running_min = math.inf
    spikes = []
    for s in finite:
        if s["step"] > 50 and running_min < math.inf and s["loss"] > spike_ratio * running_min:
            spikes.append(s["step"])
        running_min = min(running_min, s["loss"])
    if spikes:
        recovered = finite[-1]["loss"] <= 1.1 * min(x["loss"] for x in finite)
        flags.append(
            f"loss spike above {spike_ratio}x running minimum at step(s) {compress(spikes)}"
            + ("; loss recovered afterwards" if recovered else "; loss has NOT recovered")
        )

    vals = run["val_bpb"]
    rises = 0
    for a, b in zip(vals, vals[1:]):
        rises = rises + 1 if b["value"] > a["value"] else 0
        if rises >= 2:
            flags.append(f"val bpb rising over consecutive evals up to step {b['step']}")
            break

    timed = [s for s in steps if s["step"] > 10]
    stats = {}
    if timed:
        tps = [s["tok_per_sec"] for s in timed]
        mfu = [s["mfu"] for s in timed]
        stats = {"median_tok_per_sec": int(statistics.median(tps)),
                 "median_mfu": round(statistics.median(mfu), 2)}
        slow = [s["step"] for s in timed if s["tok_per_sec"] < drop_ratio * stats["median_tok_per_sec"]]
        if slow:
            flags.append(f"throughput drop below {int(drop_ratio * 100)}% of median at step(s) {compress(slow)}")
        if stats["median_mfu"] < low_mfu:
            flags.append(f"low MFU: median {stats['median_mfu']}% (threshold {low_mfu}%)")

    if run["warnings"]:
        flags.append(f"{len(run['warnings'])} warning line(s) in log")
    if run["errors"]:
        # the last matching line is usually the actual exception, not the "Traceback" header
        flags.append(f"error: {run['errors'][-1]}")

    finished = run["total_time_min"] is not None
    if not finished and not run["errors"]:
        flags.append("incomplete: no final summary yet (still running, or stopped without a traceback)")

    status = "failed" if run["errors"] else ("finished" if finished else "running or stopped")
    last = steps[-1] if steps else None
    summary = {
        "run": run["run"], "kind": run["kind"], "gpu": run["gpu"], "status": status,
        "last_step": last["step"] if last else None,
        "total_steps": last["total"] if last else None,
        "progress_pct": last["pct"] if last else None,
        "final_loss": last["loss"] if last else None,
        "min_loss": min((s["loss"] for s in finite), default=None),
        "eta_min": last["eta_min"] if last else None,
        "last_val_bpb": vals[-1]["value"] if vals else None,
        "min_val_bpb": run["min_val_bpb"] if run["min_val_bpb"] is not None
                       else min((v["value"] for v in vals), default=None),
        "last_core": run["core"][-1]["value"] if run["core"] else None,
        "beats_gpt2_core": (run["core"][-1]["value"] > GPT2_CORE) if run["core"] else None,
        "last_chatcore": run["chatcore"][-1]["value"] if run["chatcore"] else None,
        "total_time_min": run["total_time_min"],
        "peak_mem_mib": run["peak_mem_mib"],
        **stats,
        "warnings": run["warnings"],
        "flags": flags,
    }
    return summary


def compress(nums, limit=8):
    shown = ", ".join(str(n) for n in nums[:limit])
    return shown + (f" (+{len(nums) - limit} more)" if len(nums) > limit else "")


def fmt(v, spec=""):
    if v is None:
        return "-"
    return format(v, spec) if spec else str(v)


def print_text(summaries):
    for s in summaries:
        title = f"Run {s['run']} ({s['kind'] or 'unknown'})" + (f" on {s['gpu']}" if s["gpu"] else "")
        print(title)
        print("-" * len(title))
        prog = f"{fmt(s['last_step'])}" + (f"/{s['total_steps']}" if s["total_steps"] else "")
        print(f"Status            : {s['status']}  (step {prog}, {fmt(s['progress_pct'])}%"
              + (f", eta {s['eta_min']:.1f} min)" if s["eta_min"] is not None and s["status"] != "finished" else ")"))
        print(f"Smoothed loss     : final {fmt(s['final_loss'], '.4f')}, min {fmt(s['min_loss'], '.4f')}")
        print(f"Validation bpb    : last {fmt(s['last_val_bpb'], '.4f')}, min {fmt(s['min_val_bpb'], '.4f')}")
        if s["last_core"] is not None:
            verdict = "above" if s["beats_gpt2_core"] else "below"
            print(f"CORE              : {s['last_core']:.4f} ({verdict} GPT-2's {GPT2_CORE})")
        if s["last_chatcore"] is not None:
            print(f"ChatCORE          : {s['last_chatcore']:.4f}")
        print(f"Throughput        : median {fmt(s.get('median_tok_per_sec'), ',')} tok/sec, "
              f"median MFU {fmt(s.get('median_mfu'))}%")
        print(f"Training time     : {fmt(s['total_time_min'], '.2f')} min")
        print(f"Peak memory       : {fmt(s['peak_mem_mib'], ',.0f')} MiB")
        print("Flags             :" + ("" if s["flags"] else " none"))
        for fl in s["flags"]:
            print(f"  - {fl}")
        for w in s["warnings"][:5]:
            print(f"    {w}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--csv", metavar="OUT", help="write per-step rows of all runs to a CSV file")
    ap.add_argument("--spike-ratio", type=float, default=1.5)
    ap.add_argument("--low-mfu", type=float, default=20.0)
    args = ap.parse_args()

    try:
        with open(args.log, encoding="utf-8", errors="replace") as fh:
            runs = parse(fh)
    except OSError as exc:
        sys.exit(f"cannot read {args.log}: {exc}")

    if not runs:
        sys.exit("No nanochat training lines found. Is this a base_train or chat_sft log?")

    summaries = [analyse(r, spike_ratio=args.spike_ratio, low_mfu=args.low_mfu) for r in runs]

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "step", "loss", "lrm", "dt_ms", "tok_per_sec", "mfu", "total_time_min"])
            for r in runs:
                for s in r["steps"]:
                    w.writerow([r["run"], s["step"], s["loss"], s["lrm"], s["dt_ms"],
                                s["tok_per_sec"], s["mfu"], s["total_time_min"]])

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print_text(summaries)


if __name__ == "__main__":
    main()
