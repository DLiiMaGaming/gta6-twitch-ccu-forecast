#!/usr/bin/env python3
"""Forecast a game launch day's Twitch viewers, hour by hour.

Reads every case file (*.in) in the input directory. Each case carries the
category's history up to the end of release-day hour 1, that hour's viewer
count, and a library of past launches ("comparables") with their full hourly
curves. For each of hours 2..24 the program writes five quantiles of the
hour's viewer count.

Method: a launch's day is largely foretold by the shape of its first hours.
The hours before the checkpoint show the unlock staircase (which regions
unlocked when), and launches whose staircases and scale resemble this one
tend to unfold alike. So the forecast is a similarity-weighted distribution
of the comparables' log-ratios ln(viewers_h / viewers_1): each comparable is
weighted by a Gaussian kernel on the root-mean-square distance between its
pre-checkpoint log-ratio profile and the target's, times a kernel on the
log distance of the two hour-1 counts. Per forecast hour, the weighted
empirical quantiles of the log-ratios are blended with a weighted normal
fit (mean and spread of the same weighted sample), which smooths the tails
where the effective sample is thin. Everything is a deterministic function
of the case file: no randomness, no network, standard library only.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Profile offsets: whole hours relative to the day's first hour (offset 0 is
# hour 1 itself). The day before release is where the unlock staircase shows.
PROFILE_FROM = -31

# Kernel bandwidths and the empirical/normal blend, fitted on the kit's
# practice launches and cross-checked by leaving each comparable out in turn.
BW_PROFILE = 1.5
BW_SCALE = 0.75
SHRINK_TO_NORMAL = 0.6
MIN_PROFILE_OVERLAP = 4


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_offsets(points, day_start, stamp_key):
    """Map whole-hour offsets from day_start to viewer counts."""
    start = parse_time(day_start)
    series = {}
    for point in points or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        delta = (parse_time(point[stamp_key]) - start).total_seconds()
        series[int(delta // 3600)] = viewers
    return series


def hourly_history(case):
    """The target's hourly viewers by offset, from whole-hour means when the
    case carries them, else by averaging the raw history points per hour."""
    hourly = case.get("history_hourly")
    if hourly:
        return hour_offsets(hourly, case["day_start"], "from")
    totals, counts = {}, {}
    start = parse_time(case["day_start"])
    for point in case.get("history") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        offset = int((parse_time(point["at"]) - start).total_seconds() // 3600)
        totals[offset] = totals.get(offset, 0.0) + viewers
        counts[offset] = counts.get(offset, 0) + 1
    return {off: totals[off] / counts[off] for off in totals}


def log_profile(series, base, offsets):
    """ln(viewers / base) at each offset, None where the hour is missing."""
    out = []
    for off in offsets:
        value = series.get(off)
        if value and value > 0 and base > 0:
            out.append(math.log(value / base))
        else:
            out.append(None)
    return out


def normsinv(tau):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if tau < plow:
        q = math.sqrt(-2 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if tau > phigh:
        q = math.sqrt(-2 * math.log(1 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def weighted_quantile(pairs, tau):
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


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int so the
    JSON writes 1240 rather than 1240.0."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def comparable_pool(case):
    """Usable comparables: (day curve {hour: viewers}, hourly series, name).

    A comparable is usable when its release day's first hour is present, so
    its log-ratios to that hour exist.
    """
    game_name = (case.get("game") or {}).get("name", "")
    pool = []
    for comp in case.get("comparables") or []:
        name = comp.get("name", "")
        if game_name and game_name in name:
            continue  # the target category's own pre-release history, not a launch
        series = hour_offsets(comp.get("hourly"), comp["day_start"], "from")
        day = {h: series.get(h - 1) for h in range(1, 25)}
        if day.get(1):
            pool.append((name, day, series))
    pool.sort(key=lambda item: item[0])
    return pool


def similarity_weights(case, pool, target_series, p1):
    """One kernel weight per pool entry, from profile and scale distance."""
    offsets = list(range(PROFILE_FROM, 1))
    target_profile = log_profile(target_series, p1, offsets)
    target_log = math.log(max(1.0, p1))
    weights = []
    for _name, day, series in pool:
        theirs = log_profile(series, day[1], offsets)
        sq, overlap = 0.0, 0
        for mine, other in zip(target_profile, theirs):
            if mine is not None and other is not None:
                sq += (mine - other) ** 2
                overlap += 1
        if overlap < MIN_PROFILE_OVERLAP:
            weights.append(0.0)
            continue
        rms = math.sqrt(sq / overlap)
        scale_gap = abs(math.log(day[1]) - target_log)
        weights.append(math.exp(-0.5 * (rms / BW_PROFILE) ** 2)
                       * math.exp(-0.5 * (scale_gap / BW_SCALE) ** 2))
    if pool and not any(w > 0 for w in weights):
        # The target's history is too sparse to place it among the launches:
        # fall back to the pooled distribution rather than no distribution.
        weights = [1.0] * len(pool)
    return weights


def forecast_case(case):
    rows = sorted(case["forecast"], key=lambda r: r["hour"])
    p1 = 1.0
    for known in case.get("known") or []:
        if known.get("hour") == 1 and known.get("viewers"):
            p1 = max(1.0, float(known["viewers"]))
            break
    levels = case.get("levels") or LEVELS
    zetas = [normsinv(tau) for tau in levels]

    target_series = hourly_history(case)
    pool = comparable_pool(case)
    weights = similarity_weights(case, pool, target_series, p1) if pool else []

    hours = []
    for row in rows:
        h = row["hour"]
        pairs = []
        for (_name, day, _series), w in zip(pool, weights):
            if w > 0 and day.get(h):
                pairs.append((math.log(day[h] / day[1]), w))
        quantiles = []
        if pairs:
            total_w = sum(w for _, w in pairs)
            mean = sum(v * w for v, w in pairs) / total_w
            var = sum(w * (v - mean) ** 2 for v, w in pairs) / total_w
            spread = math.sqrt(max(var, 1e-6))
            for tau, z in zip(levels, zetas):
                emp = weighted_quantile(pairs, tau)
                blended = emp * (1 - SHRINK_TO_NORMAL) + (mean + z * spread) * SHRINK_TO_NORMAL
                quantiles.append(blended)
        else:
            # No comparable speaks for this hour: fall back to hour 1's count
            # with a lognormal spread that widens over the day.
            sigma = 0.25 + 0.05 * math.sqrt(max(0, h - 1))
            quantiles = [z * sigma for z in zetas]
        quantiles.sort()
        values = []
        prev = 0.0
        for q in quantiles:
            value = p1 * math.exp(q)
            value = min(1e9, max(value, prev, 1e-6))
            values.append(round2(value))
            prev = value
        hours.append({"hour": h, "q": values})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
