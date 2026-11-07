---
name: check-training-run
description: Read a nanochat training log (base_train or chat_sft output, e.g. runs/speedrun.log or a screen/torchrun log) and report how the run went - loss, validation bpb, CORE, throughput, MFU, ETA - plus anything that looks wrong, such as NaN loss, loss spikes, rising validation bpb, throughput drops, OOM or a crash. Use when asked to check, summarise, debug or compare a training run.
---

# Checking a nanochat training run

The goal is a short, accurate status report the user can act on. Numbers come from the log, never from memory or guesswork.

## 1. Find the log

Use the path the user gives. If none, look in this order and use the newest match:

1. `runs/*.log` (the speedrun script suggests `screen -L -Logfile runs/speedrun.log`)
2. `screenlog.*`, `*.log` or `nohup.out` in the repository root
3. Ask the user to paste the log or point to it

If the run is still going, say so: the report then describes the run so far.

## 2. Parse it with the helper script

```bash
python .claude/skills/check-training-run/scripts/parse_log.py <path-to-log>
```

Add `--json` for machine-readable output, or `--csv out.csv` to save the per-step table for plotting. The script uses only the standard library, so it works without the project's virtual environment.

It understands the exact lines printed by:

- `scripts/base_train.py`: `step 00100/05000 (2.00%) | loss: ... | lrm: ... | dt: ...ms | tok/sec: ... | bf16_mfu: ... | ... | total time: ...m | eta: ...m`, plus `Step NNNNN | Validation bpb: ...`, `Step NNNNN | CORE metric: ...` and the closing `Peak memory usage`, `Total training time`, `Minimum validation bpb` lines
- `scripts/chat_sft.py`: the same step line without the total step count, plus `Validation bpb` and `ChatCORE` lines

If a log contains several runs (for example base training followed by SFT), the script splits them and reports each one.

## 3. Read the flags with judgement

The script raises flags for:

| Flag | What it means | What to check next |
|---|---|---|
| `NaN/Inf loss` | Training diverged | Learning rate, `--fp8`, recent code changes; resume from an earlier checkpoint |
| `loss spike` | Smoothed loss jumped well above its recent minimum | Whether it recovers in later steps; one recovered spike is normal |
| `val bpb rising` | Validation got worse over consecutive evals | Overfitting, data repetition (`epoch` field), LR schedule |
| `throughput drop` | tok/sec fell well below the run's median | Other jobs on the GPU, thermal throttling, dataloader stalls |
| `low MFU` | Median MFU under the threshold | Batch size, `--device-batch-size`, attention fallback warnings |
| `warning lines` | e.g. the SDPA or Flash Attention fallback messages | These often explain low MFU |
| `error` | Traceback, CUDA out of memory, NCCL error, killed process | The lines just before the error |
| `incomplete` | The log stops before the final summary | Still running, or crashed without a traceback |

Every flag is a lead, not a conclusion. Look at the surrounding log lines before you blame anything, and say how sure you are.

Useful reference points:

- The GPT-2 CORE score to beat is **0.256525** (from the README). The script reports whether the run reached it.
- The leaderboard in `README.md` and `dev/LEADERBOARD.md` lists `val_bpb`, CORE and wall-clock time for past reference runs. Compare only with runs of the same depth and setup.
- The first ten steps are excluded from timing in `base_train.py`, so early `dt` values are always high. The script ignores them for throughput statistics.

## 4. Report

Keep it short and in plain language:

1. **Status**: finished, still running (with ETA), or failed (with the error).
2. **Key numbers**: final and minimum validation bpb, last CORE or ChatCORE, final smoothed loss, median tok/sec and MFU, total training time, peak memory.
3. **Problems**: each flag, with step numbers and your reading of it.
4. **Next step**: one or two concrete suggestions, only if something needs doing.

Don't paste the whole log. Quote at most a few lines when they show the problem.
