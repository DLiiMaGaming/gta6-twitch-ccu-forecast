#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Reads every case file (*.in) in the input directory. A case holds the
category's history up to the end of release-day hour 1, that hour's viewer
count, and a library of past big launches ("comparables") with their full
hourly curves. For each of hours 2..24 the program writes five quantiles
of that hour's viewer count to <id>.out.

Idea: a launch day is largely foretold by the shape of its first hours.
The hours before the checkpoint trace the unlock staircase -- which
regions unlocked when -- and launches whose staircase and scale resemble
this one tend to unfold alike. So the forecast is a similarity-weighted
distribution of the comparables' log-ratios ln(viewers_h / viewers_1):
each comparable gets a Gaussian-kernel weight from the root-mean-square
gap between its pre-checkpoint log-ratio profile and the target's, times
a kernel on the gap between the two hour-1 counts. For each forecast
hour, the weighted empirical quantiles of the log-ratios are blended
with a weighted normal fit of the same sample, which steadies the tails
where the effective sample is thin. Deterministic, offline, standard
library only.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores.
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Profile window: whole-hour offsets from the day's first hour, reaching
# back over the day before release, where the unlock staircase shows.
PROFILE_START = -31

# Kernel widths and the empirical/normal blend, fitted on the kit's
# practice launches (mean pinball loss about 0.19, house baseline 0.39).
BAND_PROFILE = 1.5
BAND_SCALE = 0.75
NORMAL_BLEND = 0.6
MIN_SHARED = 4


def instant(text):
    """Parse an ISO 8601 timestamp (trailing Z allowed)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hourly_series(points, day_start, stamp_key):
    """Viewer counts keyed by whole-hour offset from the day's first hour."""
    start = instant(day_start)
    series = {}
    for point in points or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        delta = (instant(point[stamp_key]) - start).total_seconds()
        series[int(delta // 3600)] = viewers
    return series


def target_series(case):
    """The target's hourly viewers: whole-hour means when the case carries
    them, else the raw history points averaged per hour."""
    hourly = case.get("history_hourly")
    if hourly:
        return hourly_series(hourly, case["day_start"], "from")
    start = instant(case["day_start"])
    totals, counts = {}, {}
    for point in case.get("history") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        off = int((instant(point["at"]) - start).total_seconds() // 3600)
        totals[off] = totals.get(off, 0.0) + viewers
        counts[off] = counts.get(off, 0) + 1
    return {off: totals[off] / counts[off] for off in totals}


def profile_ratios(series, base, offsets):
    """ln(viewers / base) at each offset, None where the hour is missing."""
    out = []
    for off in offsets:
        value = series.get(off)
        if value and value > 0 and base > 0:
            out.append(math.log(value / base))
        else:
            out.append(None)
    return out


def wquantile(pairs, tau):
    """Weighted empirical quantile of (value, weight) pairs, interpolated."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return None
    acc = 0.0
    for i, (value, weight) in enumerate(ordered):
        acc += weight
        if acc / total >= tau:
            if i == 0:
                return value
            prev_value = ordered[i - 1][0]
            frac = (tau * total - (acc - weight)) / weight if weight > 0 else 0.0
            return prev_value + (value - prev_value) * min(1.0, max(0.0, frac))
    return ordered[-1][0]


def round_half_up_2dp(x):
    """Round to 2 decimals, half up; whole numbers come back as ints, so
    the JSON writes 1240 rather than 1240.0."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def collect_comparables(case):
    """Usable comparables, sorted by name for determinism: each holds its
    release-day curve {hour 1..24: viewers} and its full offset series.

    A comparable is usable when its release day's first hour is present,
    so its log-ratios to that hour exist. The target category's own
    pre-release history is not a launch and is left out.
    """
    game_name = (case.get("game") or {}).get("name", "")
    pool = []
    for comp in case.get("comparables") or []:
        name = comp.get("name", "")
        if game_name and game_name in name:
            continue
        series = hourly_series(comp.get("hourly"), comp["day_start"], "from")
        day = {h: series.get(h - 1) for h in range(1, 25)}
        if day.get(1):
            pool.append({"name": name, "day": day, "series": series})
    pool.sort(key=lambda item: item["name"])
    return pool


def kernel_weights(pool, target, p1):
    """One kernel weight per pool entry, from profile and scale distance.

    Falls back to equal weights when the target's history is too sparse
    to place it among the launches.
    """
    offsets = list(range(PROFILE_START, 1))
    mine = profile_ratios(target, p1, offsets)
    target_log = math.log(max(1.0, p1))
    weights = []
    for entry in pool:
        theirs = profile_ratios(entry["series"], entry["day"][1], offsets)
        sq, shared = 0.0, 0
        for a, b in zip(mine, theirs):
            if a is not None and b is not None:
                sq += (a - b) ** 2
                shared += 1
        if shared < MIN_SHARED:
            weights.append(0.0)
            continue
        rms = math.sqrt(sq / shared)
        scale_gap = abs(math.log(entry["day"][1]) - target_log)
        weights.append(math.exp(-0.5 * (rms / BAND_PROFILE) ** 2)
                       * math.exp(-0.5 * (scale_gap / BAND_SCALE) ** 2))
    if pool and not any(w > 0 for w in weights):
        weights = [1.0] * len(pool)
    return weights


def hour1_print(case):
    """Hour 1's viewer count from the case's known rows (1.0 if absent)."""
    for known in case.get("known") or []:
        if known.get("hour") == 1 and known.get("viewers"):
            return max(1.0, float(known["viewers"]))
    return 1.0


def forecast_case(case):
    """Five quantiles per forecast hour, as log-ratios to hour 1's print
    converted back to viewer counts."""
    p1 = hour1_print(case)
    levels = case.get("levels") or list(TAUS)
    zetas = [NormalDist().inv_cdf(t) for t in levels]
    rows = sorted(case["forecast"], key=lambda r: r["hour"])

    series = target_series(case)
    pool = collect_comparables(case)
    weights = kernel_weights(pool, series, p1) if pool else []

    hours = []
    for row in rows:
        h = row["hour"]
        pairs = []
        for entry, w in zip(pool, weights):
            if w > 0 and entry["day"].get(h):
                pairs.append((math.log(entry["day"][h] / entry["day"][1]), w))
        if pairs:
            total_w = sum(w for _, w in pairs)
            mean = sum(v * w for v, w in pairs) / total_w
            var = sum(w * (v - mean) ** 2 for v, w in pairs) / total_w
            spread = math.sqrt(max(var, 1e-6))
            ratios = []
            for tau, z in zip(levels, zetas):
                emp = wquantile(pairs, tau)
                ratios.append(emp * (1 - NORMAL_BLEND)
                              + (mean + z * spread) * NORMAL_BLEND)
        else:
            # No comparable speaks for this hour: stay at hour 1's count
            # with a lognormal spread that widens over the day.
            sigma = 0.25 + 0.05 * math.sqrt(max(0, h - 1))
            ratios = [z * sigma for z in zetas]
        ratios.sort()
        values = []
        prev = 0.0
        for r in ratios:
            value = p1 * math.exp(r)
            value = min(1e9, max(value, prev, 1e-6))
            values.append(round_half_up_2dp(value))
            prev = value
        hours.append({"hour": h, "q": values})
    return {"hours": hours}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
