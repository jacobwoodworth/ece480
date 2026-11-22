#!/usr/bin/env python3
"""
How To Use:
  * Join the ESP32's WiFi network "MotorMonitor-T15" from the PC.
  * Run this script. It polls http://192.168.4.1/api/status, writes one CSV row
    per 1-second epoch, and tags every row with the fault label you give it.
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Which folder inside data/ each label is saved to
LABEL_FOLDERS = {
    "stable": "Stable",
    "overload": "Overload",
    "airflow_blocked": "Airflow Blocked",
    "misalignment": "Misalignment",
}
# Other spellings that are accepted for each label
ALIASES = {"normal": "stable", "healthy": "stable", "airflow": "airflow_blocked",
           "blocked": "airflow_blocked", "blocked_airflow": "airflow_blocked",
           "misaligned": "misalignment"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# Turn whatever the user typed into a standard label ("Airflow Blocked" becomes airflow_blocked)
def normalize_label(text):
    key = text.strip().lower().replace("-", " ").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return ALIASES.get(key, key)


# ESP32 state numbers mapped to names
STATE_NAMES = ["IDLE", "STARTING", "LEARNING", "NORMAL", "WARNING", "FAULT"]

# CSV column name paired with the matching key in the ESP32's JSON
FIELDS = [
    # counters from the ESP32
    ("esp_epoch", "ep"),
    ("esp_uptime_s", "up"),
    # measured features (these are what the random forest uses)
    ("irms_A", "i"),
    ("vrms_V", "v"),
    ("power_W", "p"),
    ("pf", "pf"),
    ("crest", "cr"),
    ("thd", "thd"),
    ("sideband", "sb"),
    ("ripple", "rp"),
    ("freq_Hz", "f"),
    # ESP32's own baseline results (kept for reference, not used for training)
    ("esp_dev_pct", "dev"),
    ("esp_score", "sc"),
    ("esp_health", "h"),
    ("esp_match", "lab"),
    ("baseline_valid", "bv"),
    ("calibrated", "cal"),
    ("supply_flag", "sup"),
    # sampling health checks
    ("fs_measured", "fs"),
    ("overruns", "ov"),
]
# Raw signal checks, stored under "raw" in the JSON
RAW_FIELDS = [("clip_I", "ci"), ("clip_V", "cv"), ("bias_I", "bi"), ("bias_V", "bv"),
              ("pp_I", "pi"), ("pp_V", "pv")]

# Full list of CSV column names in order
HEADER = (["pc_time", "run_id", "label", "state"] + [c for c, _ in FIELDS]
          + [c for c, _ in RAW_FIELDS] + ["esp_msg"])


# Request a page from the ESP32 and return it as a Python dict
def get_json(base, path, timeout=2.0):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


# Send a one-off command to the ESP32 like relearning the baseline
def send_cmd(base, cmd):
    try:
        res = get_json(base, "/api/cmd?c=" + cmd)
        print(f"Command '{cmd}': {'ok' if res.get('ok') else 'rejected'}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"Could not reach the ESP32 at {base}: {e}")
        sys.exit(1)


def main():
    # Command-line options
    ap = argparse.ArgumentParser(description="Log ESP32 motor-monitor features to CSV.")
    ap.add_argument("--label", help="fault label for this run, e.g. stable / overload / "
                                    "airflow_blocked / misalignment")
    ap.add_argument("--host", default="192.168.4.1", help="ESP32 address (default 192.168.4.1)")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help=f"data folder containing the label folders (default {DEFAULT_DATA_DIR})")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after this many seconds (default: run until Ctrl+C)")
    ap.add_argument("--notes", default="", help="free-text notes saved in runs_index.csv")
    ap.add_argument("--all-states", action="store_true",
                    help="also log IDLE/STARTING epochs (default: skip them, because "
                         "the motor is off or still in its start-up surge)")
    ap.add_argument("--cmd", choices=["relearn", "stop"], help="send a command to the ESP32 and exit")
    args = ap.parse_args()

    base = "http://" + args.host

    # Command mode: send the command and quit without logging
    if args.cmd:
        send_cmd(base, args.cmd)
        return
    if not args.label:
        ap.error("--label is required (e.g. --label stable)")

    # Check the label is one we know
    label = normalize_label(args.label)
    if label not in LABEL_FOLDERS:
        ap.error(f"unknown label '{args.label}'. Use one of: {', '.join(LABEL_FOLDERS)}")

    # Pick the label's folder and build a unique file name from the start time
    data_dir = Path(args.data_dir)
    folder = data_dir / LABEL_FOLDERS[label]
    if not folder.exists():
        print(f"Creating folder {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    start = datetime.now()
    run_id = f"{label}_{start.strftime('%Y%m%d_%H%M%S')}"
    path = folder / (run_id + ".csv")

    print(f"Logging to {path}")
    print(f"Polling {base}/api/status   (Ctrl+C to stop)")

    # Counters for the end-of-run summary
    rows = 0
    last_ep = None
    missed = 0
    reboots = 0
    warned_offline = False
    t0 = time.monotonic()

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        f.flush()
        try:
            # Main loop: keep asking the ESP32 for new data until time is up or Ctrl+C
            while True:
                if args.duration and time.monotonic() - t0 >= args.duration:
                    break

                # Get the latest readings, wait and retry if the ESP32 can't be reached
                try:
                    d = get_json(base, "/api/status")
                    if warned_offline:
                        print("Reconnected.")
                        warned_offline = False
                except (urllib.error.URLError, OSError, ValueError) as e:
                    if not warned_offline:
                        print(f"ESP32 not reachable ({e}). Is the PC on the MotorMonitor-T15 "
                              f"WiFi? Retrying...")
                        warned_offline = True
                    time.sleep(1.0)
                    continue

                # Only save when the ESP32 has finished a new 1-second epoch
                ep = d.get("ep")
                if ep is None or ep == last_ep:
                    time.sleep(0.25)
                    continue

                # Detect an ESP32 reboot or skipped seconds
                if last_ep is not None:
                    if ep < last_ep:
                        reboots += 1
                        print("ESP32 epoch counter went backwards - it probably rebooted.")
                    elif ep > last_ep + 1:
                        missed += ep - last_ep - 1
                        print(f"Missed {ep - last_ep - 1} epoch(s) (WiFi hiccup?).")
                last_ep = ep

                # Skip rows while the motor is off or still starting up
                st = int(d.get("st", 0))
                state = STATE_NAMES[st] if 0 <= st < len(STATE_NAMES) else str(st)
                if not args.all_states and state in ("IDLE", "STARTING"):
                    print(f"\r[{state:8s}] {d.get('msg', '')[:70]:70s}", end="", flush=True)
                    continue

                # Build one CSV row and save it straight away
                raw = d.get("raw", {})
                row = ([datetime.now().isoformat(timespec="milliseconds"), run_id, label, state]
                       + [d.get(k, "") for _, k in FIELDS]
                       + [raw.get(k, "") for _, k in RAW_FIELDS]
                       + [d.get("msg", "")])
                w.writerow(row)
                f.flush()
                rows += 1

                # Live status line in the terminal
                clip = " CLIPPING!" if raw.get("ci") or raw.get("cv") else ""
                print(f"\r[{state:8s}] rows={rows:5d}  I={d.get('i', 0):6.3f} A  "
                      f"P={d.get('p', 0):7.1f} W  PF={d.get('pf', 0):5.2f}  "
                      f"SB={d.get('sb', 0):.4f}{clip}      ", end="", flush=True)
        except KeyboardInterrupt:
            # Ctrl+C ends the run cleanly
            pass

    # End-of-run summary
    end = datetime.now()
    print(f"\nSaved {rows} rows to {path}  (missed epochs: {missed}, reboots: {reboots})")
    if rows == 0:
        print("No rows were logged. Check that the motor is running (state not IDLE/STARTING), "
              "or use --all-states.")

    # Add this run to data/runs_index.csv (one line per run)
    index = data_dir / "runs_index.csv"
    new = not index.exists()
    with open(index, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "label", "start", "end", "rows", "missed_epochs",
                        "reboots", "file", "notes"])
        w.writerow([run_id, label, start.isoformat(timespec="seconds"),
                    end.isoformat(timespec="seconds"), rows, missed, reboots,
                    str(path.relative_to(data_dir)), args.notes])


if __name__ == "__main__":
    main()