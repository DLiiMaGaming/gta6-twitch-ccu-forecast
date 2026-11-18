#!/usr/bin/env python3
"""Hour-by-hour Twitch forecast for GTA VI's launch day.

The program takes one case file at a time from the input directory. A
case says what the Grand Theft Auto VI category on Twitch did up to the
end of hour 1 of release day, and it bundles the hourly curves of dozens
of earlier big launches. For every hour from 2 to 24 the program writes
five quantiles of that hour's viewer count.

How it reasons: launches that opened alike tend to play out alike. The
hours before the checkpoint show each launch's unlock staircase -- the
regions coming online one after another -- so the program measures how
closely a past launch's staircase and opening size match this one's, and
lets the closest matches speak loudest. Concretely, each past launch
contributes the log-ratio of its hour-h viewers to its hour-1 viewers,
weighted by a Gaussian kernel over staircase distance and over opening
size. Each hour's quantiles come from the weighted empirical quantiles of
those ratios, pulled partway toward a weighted normal fit so thin tails
do not wag the forecast. When no past launch covers an hour, the forecast
holds hour 1's level with a spread that grows as the day goes on.

Everything is a deterministic function of the case file: no network, no
clock, no randomness, standard library only.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

CASE_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = tuple(NormalDist().inv_cdf(t) for t in LEVELS)

LOOKBACK = 31        # profile reaches this many hours before hour 1
WIDTH_STAIRCASE = 1.5
WIDTH_OPENING = 0.75
PULL_TO_NORMAL = 0.6
MIN_OVERLAP = 4


def _moment(text):
    """An ISO 8601 timestamp, trailing Z included, as a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hourly(points, day_start, key):
    """Collapse timestamped points into {whole-hour offset: viewers}.

    Offsets count from the day's first hour. A later point in the same
    hour replaces an earlier one, which is how whole-hour means arrive.
    """
    start = _moment(day_start)
    series = {}
    for point in points or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        delta = (_moment(point[key]) - start).total_seconds()
        series[int(delta // 3600)] = viewers
    return series


def _target_hours(case):
    """The target's hourly viewers, preferring the case's whole-hour
    means and otherwise averaging its raw points within each hour."""
    if case.get("history_hourly"):
        return _hourly(case["history_hourly"], case["day_start"], "from")
    start = _moment(case["day_start"])
    buckets = {}
    for point in case.get("history") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        off = int((_moment(point["at"]) - start).total_seconds() // 3600)
        buckets.setdefault(off, []).append(viewers)
    return {off: sum(vals) / len(vals) for off, vals in buckets.items()}


def _ratios(series, base):
    """ln(viewers / base) per hour offset, skipping missing or dead hours."""
    out = {}
    for off, value in series.items():
        if value and value > 0 and base > 0:
            out[off] = math.log(value / base)
    return out


class Comparable:
    """One past launch, with its release-day curve and its log-ratios.

    ``day`` maps hours 1..24 to viewer counts; ``ratios`` maps the same
    hours to ln(viewers / hour-1 viewers). A launch only counts when its
    hour 1 is present, so the ratios exist at all.
    """

    def __init__(self, raw):
        self.name = raw.get("name", "")
        series = _hourly(raw.get("hourly"), raw["day_start"], "from")
        self.day = {h: series.get(h - 1) for h in range(1, 25)}
        self.series = series
        self.ratios = _ratios(self.day, self.day.get(1) or 0) if self.day.get(1) else {}

    @property
    def usable(self):
        return bool(self.day.get(1))

    def staircase(self):
        """Log-ratios over the pre-checkpoint window, as {offset: value}."""
        prof = _ratios(self.series, self.day[1])
        return {o: v for o, v in prof.items() if -LOOKBACK <= o < 1}


def _opening_viewers(case):
    """Hour 1's print from the known rows; 1.0 when the case has none."""
    for known in case.get("known") or []:
        if known.get("hour") == 1 and known.get("viewers"):
            return max(1.0, float(known["viewers"]))
    return 1.0


def _weights(comps, target_prof, p1):
    """Kernel weight per comparable from staircase and opening-size gaps.

    Comparables sharing too little staircase with the target get zero.
    When nothing scores above zero -- the target's history is too thin
    to place it -- every comparable counts equally instead.
    """
    target_log = math.log(max(1.0, p1))
    weights = []
    for comp in comps:
        theirs = comp.staircase()
        common = [o for o in target_prof if o in theirs]
        if len(common) < MIN_OVERLAP:
            weights.append(0.0)
            continue
        rms = math.sqrt(sum((target_prof[o] - theirs[o]) ** 2
                            for o in common) / len(common))
        gap = abs(math.log(comp.day[1]) - target_log)
        weights.append(math.exp(-0.5 * (rms / WIDTH_STAIRCASE) ** 2)
                       * math.exp(-0.5 * (gap / WIDTH_OPENING) ** 2))
    if comps and not any(w > 0 for w in weights):
        weights = [1.0] * len(comps)
    return weights


def _wquantile(pairs, tau):
    """Weighted empirical quantile of (value, weight) pairs, interpolated
    across the step where the cumulative weight crosses tau."""
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
            prev = ordered[i - 1][0]
            frac = (tau * total - (acc - weight)) / weight if weight > 0 else 0.0
            return prev + (value - prev) * min(1.0, max(0.0, frac))
    return ordered[-1][0]


def _round2(value):
    """Round half up to 2 decimals; whole results come back as ints so
    the JSON writes 1240 instead of 1240.0."""
    rounded = math.floor(value * 100 + 0.5) / 100
    return int(rounded) if rounded == int(rounded) else rounded


def forecast(case):
    """The forecast object for one case: five quantiles per hour, 2..24."""
    p1 = _opening_viewers(case)
    levels = case.get("levels") or list(LEVELS)
    zetas = [NormalDist().inv_cdf(t) for t in levels]
    plan = sorted(case["forecast"], key=lambda r: r["hour"])

    target = _target_hours(case)
    target_prof = {o: v for o, v in _ratios(target, p1).items()
                   if -LOOKBACK <= o < 1}

    game = (case.get("game") or {}).get("name", "")
    comps = []
    for raw in case.get("comparables") or []:
        comp = Comparable(raw)
        if not comp.usable:
            continue
        if game and game in comp.name:
            continue  # the target's own pre-release history, not a launch
        comps.append(comp)
    comps.sort(key=lambda c: c.name)
    weights = _weights(comps, target_prof, p1)

    hours = []
    for row in plan:
        h = row["hour"]
        pairs = [(c.ratios[h], w) for c, w in zip(comps, weights)
                 if w > 0 and h in c.ratios]
        if pairs:
            total = sum(w for _, w in pairs)
            mean = sum(v * w for v, w in pairs) / total
            var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
            spread = math.sqrt(max(var, 1e-6))
            ratios = []
            for tau, z in zip(levels, zetas):
                emp = _wquantile(pairs, tau)
                ratios.append(emp * (1 - PULL_TO_NORMAL)
                              + (mean + z * spread) * PULL_TO_NORMAL)
        else:
            # Nothing in the library covers this hour: hold hour 1's
            # level and let the spread widen with the hour.
            sigma = 0.25 + 0.05 * math.sqrt(max(0, h - 1))
            ratios = [z * sigma for z in zetas]
        ratios.sort()
        values = []
        prev = 0.0
        for r in ratios:
            value = p1 * math.exp(r)
            value = min(1e9, max(value, prev, 1e-6))
            values.append(_round2(value))
            prev = value
        hours.append({"hour": h, "q": values})
    return {"hours": hours}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(CASE_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        payload = forecast(case)
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        (OUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
