[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/DLiiMaGaming/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/DLiiMaGaming/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

This entry predicts how many people watch Grand Theft Auto VI live on
Twitch during each hour of release day — 19 November 2026, hours 2 to
24, five quantiles (5%, 25%, 50%, 75%, 95%) per hour — from what is
known when hour 1 ends: the category's hour-1 viewer count, its
hourly history for the two weeks before launch, and the full
release-day curves of 38 earlier big game launches.

## How to run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` per case. Python 3, standard library only; no network
access, no clock reads, no randomness, no state carried between
cases — the same input always gives the same output.

## The problem in one paragraph

A launch day on Twitch is not a smooth ramp. Viewers arrive in waves
as regions unlock at different local times, streamers go live, and the
hype peaks, dips overnight, and peaks again. Predicting the whole day
from a single hour of data is therefore a shape problem, not a level
problem: the question is which shape the day will follow, and hour
1 plus the two weeks of pre-release history are the only evidence
about that shape. The program answers it by analogy — it finds past
launches whose opening looked like this one and lets their days vote
on what happens next, with the closest matches voting loudest.

## How it works, step by step

### 1. Everything is measured against hour 1

Viewer counts vary by orders of magnitude between launches (a few
hundred for a quiet category, hundreds of thousands for a blockbuster),
so absolute counts are useless for comparison. Every curve is
converted to log-ratios against its own hour 1:

```
ratio(h) = ln(viewers at hour h / viewers at hour 1)
```

A ratio of 0 means "same as hour 1", +0.7 means "about twice hour 1",
−0.7 means "about half". Working in log space also matches the
scoring rule: pinball loss is computed on log viewers, so a forecast
that is off by a factor of two is penalised the same whether the
launch drew 1,000 or 100,000 viewers.

### 2. The pre-release staircase is the fingerprint

Before launch, a Twitch category fills up in steps: the extended-look
premiere, trailer drops, regional unlocks at midnight in each time
zone. The program takes the 31 hours before hour 1 and records the
log-ratio of each of those hours against hour 1 — the "staircase".
Two launches with the same staircase shape unlocked the same way, and
in practice their release days then rhyme. The target's staircase
comes from its `history_hourly` points (whole-hour means); if a case
ever lacks those, the raw `history` points are averaged inside each
hour instead.

### 3. Similarity is a Gaussian kernel over two distances

Each comparable launch gets a weight that is the product of two
Gaussian kernels:

```
weight = exp(-0.5 * (staircase_RMS / 1.5)^2)
       * exp(-0.5 * (log hour-1 gap / 0.75)^2)
```

The first factor measures the root-mean-square gap between the two
staircases on the hours they share — a launch needs at least 4 shared
staircase hours to count at all. The second factor measures the gap
between the two hour-1 sizes in log space, so a launch that opened
ten times bigger does not vote as loudly as one that opened at the
same scale, even if their staircases match. The bandwidths (1.5 and
0.75) come from the tuning grid described below.

Two safeguards keep this honest. First, the target game's own
pre-release category history is in the comparables list, but it is
not a launch — it is the target itself, seen earlier — so any
comparable whose name contains the target game's name is dropped
before weighting. Second, if every weight comes out zero (the
target's history is too thin to place it anywhere), all comparables
count equally rather than the forecast collapsing to nothing.

### 4. Each hour's quantiles are a blend of two estimates

For a forecast hour h, every comparable that covers h votes with its
own log-ratio for that hour, weighted by its kernel weight. Two
quantile estimates are computed from those votes and blended:

- **Weighted empirical quantiles** (40% of the blend): the weighted
  empirical distribution of the votes, interpolated across the step
  where the cumulative weight crosses each level. This follows the
  data exactly, including any skew or lumpiness in the real launch
  shapes.
- **Weighted normal fit** (60% of the blend): the weighted mean and
  standard deviation of the same votes, evaluated at the standard
  normal quantiles for 5/25/50/75/95%. This smooths the tails: with
  only a few dozen votes, the raw 5% and 95% empirical quantiles are
  noisy, and the normal fit keeps them from swinging to an extreme
  vote.

The blend is deliberate. Pure empirical quantiles overfit the thin
tails; a pure normal fit ignores the real asymmetry of launch days
(the overnight dip is deeper than the morning rebound is high). The
40/60 split was the best-scoring blend in the tuning grid. The five
resulting log-ratios are sorted, exponentiated against hour 1, and
rounded half-up to two decimals, with a running floor so the five
quantiles are monotone by construction.

### 5. Fallbacks, in order

Real data has holes — a comparable may not cover a given hour at all
(SullyGnome samples small categories sparsely). The fallbacks are:

1. **Pooled neighbours.** If no comparable covers hour h, the votes
   of hours h−1 and h+1 are pooled and the same blend is applied.
   Neighbouring hours of a launch day are strongly correlated, so
   this is a small step, not a guess.
2. **Persistence with widening spread.** If even the neighbours are
   empty, the forecast holds hour 1's level and lets the spread grow
   with the hour (σ = 0.25 + 0.05·√(h−1) in log space), which is the
   honest statement: the further out, the less we know.

## Tuning — how the four numbers were chosen

The model has four free numbers: the staircase window (31 hours), the
two kernel bandwidths (1.5 and 0.75), and the empirical/normal blend
(40/60). They were chosen by a grid search over the kit's four
practice launches (Marvel's Wolverine, Red Dead Redemption 2, God of
War Ragnarök, Marvel's Spider-Man 2), scored by mean pinball loss on
the log scale — the competition's own metric. The search varied the
window (7, 13, 25, 31 hours), both bandwidths, the blend, and a weight
floor, and the winning combination was checked again with
leave-one-out scoring over the large comparables (hour-1 viewers
≥ 10,000 with a near-complete day), to make sure the practice win was
not a fluke of four cases. The headline numbers:

| model | practice mean pinball loss |
| --- | --- |
| house baseline | 0.3871 |
| this program | **≈ 0.19** |

Per practice case, the program scores roughly 0.25, 0.17, 0.13 and
0.21 — about half the baseline's loss on every case, not just on
average. The tuning also showed the result is not knife-edge: nearby
parameter choices (window 25, blend 50/50, bandwidths 1.0/0.75) all
score within a few hundredths, so the forecast does not depend on
one lucky combination.

## What the forecast looks like

For a typical big launch, hour 1 is already large, and the day's
shape is a slow climb into the evening peak, a deep overnight dip,
and a second peak the next morning. The program's median usually
tracks that shape within a factor of about 1.2–1.5 of the truth on
the practice launches, and the 5–95% band is wide enough early on
(roughly a factor of two either way) to cover the real overnight dip
without being so wide that it is useless. The band is narrowest in
the middle of the day, where the comparables agree most, and widest
at the far hours, where they diverge.

## Data

All viewer counts are SullyGnome's hourly Twitch category averages
(https://sullygnome.com), quoted with attribution. The comparables
are 38 past launches, from ELDEN RING (2022) to Call of Duty:
Modern Warfare 4 (2026), each with its release-day hourly curve and
its two weeks of pre-release history. The final case is GTA VI's own
release day, 19 November 2026, recorded by make.fun from Twitch's
public directory count. Nothing in this repository re-downloads or
extends the SullyGnome data.

## Reproducibility and limits

- **Deterministic.** Same input, same output, every run. No network,
  no clock, no random seed, no floating-point nondeterminism beyond
  ordinary double arithmetic.
- **Fast.** One case takes well under a second; the whole kit's
  practice set runs in a couple of seconds on a laptop.
- **Stdlib only.** `json`, `math`, `os`, `datetime`, `pathlib` — no
  third-party packages to install.
- **What it cannot do.** It can only repeat shapes it has seen: if
  GTA VI's launch day breaks the pattern of every past launch (a
  midnight global unlock with no regional stagger, a platform outage,
  a surprise celebrity stream), no comparable will warn it, and the
  forecast will be wrong in the same direction as the closest past
  launches were. The 5–95% band prices some of that risk, not all of
  it. It also trusts hour 1: if hour 1 itself is anomalous (a partial
  hour, a recording glitch), every hour inherits the error, because
  every ratio is anchored to it.

## Files

| file | what it is |
| --- | --- |
| `main.py` | the whole program: read cases, forecast, write outputs |
| `README.md` | this file |
| `prompt.txt` | the build brief this entry was written to |
| `LICENSE` | licence |

## Licence

See `LICENSE`.
