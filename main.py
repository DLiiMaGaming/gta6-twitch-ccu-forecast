#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch audience, hour by hour.

A case file describes what the Grand Theft Auto VI category on Twitch
looked like up to the end of launch-day hour 1, together with the full
release-day curves of 38 earlier big game launches (the "comparables").
For every hour from 2 to 24 this program writes five quantiles (5, 25,
50, 75, 95 percent) of that hour's concurrent viewers.

Idea
----
A launch's release day is mostly a rerun of launches that opened the
same way. Two things tell us how alike two openings are:

1. The pre-release staircase. In the weeks before launch, a category
   fills up in steps as regions unlock and trailers drop. The shape of
   those steps -- expressed as log(viewers / hour-1 viewers) over the
   31 hours before hour 1 -- is a fingerprint of how the launch is
   rolling out.
2. The opening size itself: how many viewers hour 1 drew.

Each comparable is weighted by a Gaussian kernel over both distances
(staircase distance with bandwidth 1.5, opening-size distance with
bandwidth 0.75, in log space). For a given forecast hour, every
comparable votes with its own log-ratio ln(viewers_h / viewers_1),
weighted by that kernel. The hour's quantiles are the weighted
empirical quantiles of those votes, blended 40/60 with the quantiles of
a weighted normal fit (same weighted mean and spread), so a thin tail
of votes cannot drag a quantile to an extreme. If no comparable covers
an hour at all, the program falls back to the pooled votes of the
neighbouring hours, and if even those are empty it holds hour 1's
level with a spread that widens as the day goes on.

The program is a pure function of the case file: standard library
only, no network, no clock, no randomness, no state between cases.
The same input always produces the same output.

Tuning
------
The four numbers above (lookback 31 hours, the two kernel bandwidths,
the 40/60 blend) were picked by a grid search scored on the kit's
four practice launches, minimising mean pinball loss on the log
scale, and cross-checked with leave-one-out scoring over the large
comparables. On the practice launches the program scores a mean
pinball loss of about 0.19 against the house baseline's 0.3871.
See README.md for the numbers.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles for the five levels above.
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

WINDOW = 31            # staircase window: this many hours before hour 1
BW_SHAPE = 1.5         # Gaussian bandwidth, staircase (log-ratio) distance
BW_SIZE = 0.75         # Gaussian bandwidth, opening-size (log) distance
EMPIRICAL_SHARE = 0.4  # quantile = 0.4 * empirical + 0.6 * normal fit
MIN_SHARED = 4         # a comparable needs this many shared staircase hours
FALLBACK_SIGMA0 = 0.25
FALLBACK_SIGMA_GROWTH = 0.05


def parse_time(text):
    """ISO 8601 timestamp (trailing Z allowed) as a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def bucket_by_hour(points, day_start, time_key):
    """Timestamped points -> {whole-hour offset from day start: viewers}.

    Points carry a ``from`` timestamp; a later point in the same hour
    replaces an earlier one, which is how the case's hourly means are
    stored. Points without a viewer count are skipped.
    """
    start = parse_time(day_start)
    out = {}
    for point in points or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        delta = (parse_time(point[time_key]) - start).total_seconds()
        out[int(delta // 3600)] = viewers
    return out


def log_ratios(series, base):
    """{hour offset: ln(viewers / base)} for positive entries only."""
    if not base or base <= 0:
        return {}
    return {off: math.log(value / base)
            for off, value in series.items() if value and value > 0}


def target_history(case):
    """The target's hourly viewers before the checkpoint.

    Prefer the case's precomputed whole-hour means (``history_hourly``);
    if they are absent, average the raw ``history`` points inside each
    hour instead.
    """
    if case.get("history_hourly"):
        return bucket_by_hour(case["history_hourly"],
                              case["day_start"], "from")
    start = parse_time(case["day_start"])
    sums, counts = {}, {}
    for point in case.get("history") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        off = int((parse_time(point["at"]) - start).total_seconds() // 3600)
        sums[off] = sums.get(off, 0.0) + viewers
        counts[off] = counts.get(off, 0) + 1
    return {off: sums[off] / counts[off] for off in sums}


def hour_one_viewers(case):
    """Hour 1's viewer count from the case's ``known`` rows (1.0 if none)."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    return 1.0


def load_reference(raw, game_name):
    """One comparable launch, or None when it cannot be used.

    Returns a dict with the launch's release-day curve (hours 1..24),
    its log-ratios against hour 1, and its pre-checkpoint staircase
    (log-ratios over the WINDOW hours before hour 1). The target game's
    own pre-release category history is not a launch, so a comparable
    carrying the target game's name is dropped here.
    """
    name = raw.get("name", "")
    if game_name and game_name in name:
        return None
    series = bucket_by_hour(raw.get("hourly"), raw["day_start"], "from")
    day = {h: series.get(h - 1) for h in range(1, 25)}
    if not day.get(1):
        return None
    ratios = log_ratios(day, day[1])
    staircase = {off: value for off, value in
                 log_ratios(series, day[1]).items()
                 if -WINDOW <= off < 1}
    return {"name": name, "day": day, "ratios": ratios,
            "staircase": staircase}


def kernel_weights(refs, target_staircase, hour_one):
    """Kernel weight of each comparable against the target.

    The weight is the product of two Gaussian kernels: one over the
    root-mean-square gap between the two staircases on their shared
    hours, one over the log gap between the two hour-1 sizes. A
    comparable sharing fewer than MIN_SHARED staircase hours with the
    target gets weight zero. If every weight comes out zero -- the
    target's history is too thin to place it -- all comparables count
    equally instead of the forecast collapsing.
    """
    target_log = math.log(max(1.0, hour_one))
    weights = []
    for ref in refs:
        shared = [off for off in target_staircase if off in ref["staircase"]]
        if len(shared) < MIN_SHARED:
            weights.append(0.0)
            continue
        rms = math.sqrt(sum((target_staircase[off] - ref["staircase"][off]) ** 2
                            for off in shared) / len(shared))
        size_gap = abs(math.log(ref["day"][1]) - target_log)
        weights.append(math.exp(-0.5 * (rms / BW_SHAPE) ** 2)
                       * math.exp(-0.5 * (size_gap / BW_SIZE) ** 2))
    if refs and not any(w > 0 for w in weights):
        weights = [1.0] * len(refs)
    return weights


def weighted_quantile(pairs, tau):
    """Interpolated weighted empirical quantile of (value, weight) pairs."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return None
    running = 0.0
    for i, (value, weight) in enumerate(ordered):
        running += weight
        if running / total >= tau:
            if i == 0:
                return value
            prev_value = ordered[i - 1][0]
            frac = (tau * total - (running - weight)) / weight if weight > 0 else 0.0
            return prev_value + (value - prev_value) * min(1.0, max(0.0, frac))
    return ordered[-1][0]


def weighted_mean_sd(pairs):
    """Weighted mean and standard deviation of (value, weight) pairs."""
    total = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return mean, math.sqrt(max(var, 1e-6))


def blended_quantiles(pairs, levels, z_scores):
    """Per-level quantiles: empirical blended with a normal fit.

    Each level's quantile is EMPIRICAL_SHARE times the weighted
    empirical quantile plus (1 - EMPIRICAL_SHARE) times the normal-fit
    quantile (weighted mean + z * weighted sd). The result is sorted so
    the five quantiles are monotone by construction.
    """
    mean, sd = weighted_mean_sd(pairs)
    out = []
    for tau, z in zip(levels, z_scores):
        emp = weighted_quantile(pairs, tau)
        out.append(EMPIRICAL_SHARE * emp + (1.0 - EMPIRICAL_SHARE) * (mean + z * sd))
    out.sort()
    return out


def round_half_up(value):
    """Round to 2 decimals, half up; whole results come back as ints."""
    rounded = math.floor(value * 100 + 0.5) / 100
    return int(rounded) if rounded == int(rounded) else rounded


def predict_hour(hour, refs, weights, hour_one, levels, z_scores):
    """Five viewer quantiles for one forecast hour.

    The normal path uses the comparables that cover this hour. If none
    covers it, the pooled votes of the neighbouring hours (h-1 and h+1)
    are used instead; if those are empty too, the forecast holds hour
    1's level with a spread that grows with the hour.
    """
    votes = [(ref["ratios"][hour], w) for ref, w in zip(refs, weights)
             if w > 0 and hour in ref["ratios"]]
    if votes:
        ratios = blended_quantiles(votes, levels, z_scores)
    else:
        pooled = [(ref["ratios"][h], w)
                  for ref, w in zip(refs, weights)
                  for h in (hour - 1, hour + 1)
                  if w > 0 and h in ref["ratios"]]
        if pooled:
            ratios = blended_quantiles(pooled, levels, z_scores)
        else:
            sigma = FALLBACK_SIGMA0 + FALLBACK_SIGMA_GROWTH * math.sqrt(max(0, hour - 1))
            ratios = [z * sigma for z in z_scores]
    values = []
    floor = 0.0
    for ratio in ratios:
        value = min(1e9, max(hour_one * math.exp(ratio), floor, 1e-6))
        values.append(round_half_up(value))
        floor = value
    return values


def forecast(case):
    """{"hours": [{"hour": h, "q": [5 quantiles]}, ...]} for hours 2..24."""
    hour_one = hour_one_viewers(case)
    levels = case.get("levels") or list(QUANTILES)
    if len(levels) == len(QUANTILES):
        z_scores = list(NORMAL_Z)
    else:
        # Generic normal quantiles (Abramowitz-Stegun rational approx)
        # for a case that ever asks for different levels.
        z_scores = [normal_ppf(t) for t in levels]

    game_name = (case.get("game") or {}).get("name", "")
    refs = []
    for raw in case.get("comparables") or []:
        ref = load_reference(raw, game_name)
        if ref is not None:
            refs.append(ref)
    refs.sort(key=lambda r: r["name"])

    history = target_history(case)
    target_staircase = {off: value for off, value in
                        log_ratios(history, hour_one).items()
                        if -WINDOW <= off < 1}
    weights = kernel_weights(refs, target_staircase, hour_one)

    hours = []
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        hour = row["hour"]
        hours.append({"hour": hour,
                      "q": predict_hour(hour, refs, weights, hour_one,
                                        levels, z_scores)})
    return {"hours": hours}


def normal_ppf(p):
    """Inverse standard-normal CDF (Abramowitz & Stegun 26.2.23)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        payload = forecast(case)
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        (OUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
