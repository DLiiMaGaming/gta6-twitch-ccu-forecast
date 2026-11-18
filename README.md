[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/DLiiMaGaming/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/DLiiMaGaming/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

This entry predicts how many people watch Grand Theft Auto VI live on
Twitch during each hour of release day, 19 November 2026 — hours 2 to 24,
five quantiles (5%, 25%, 50%, 75%, 95%) per hour — from what is known when
hour 1 ends.

## How to run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` per case. Python 3, standard library only; no network, and the
same input always gives the same output.

## How it works

Launches that opened alike tend to play out alike. The hours before the
checkpoint show each launch's unlock staircase, so the program measures
how closely a past launch's staircase and opening size match this one's,
and lets the closest matches speak loudest. Each past launch contributes
the log-ratio of its hour-h viewers to its hour-1 viewers, weighted by a
Gaussian kernel over staircase distance and over opening size. Each
hour's quantiles come from the weighted empirical quantiles of those
ratios, pulled partway toward a weighted normal fit so thin tails do not
wag the forecast. When no past launch covers an hour, the forecast holds
hour 1's level with a spread that grows as the day goes on.

## Practice result

On the kit's four practice launches the program scores a mean pinball
loss of **0.1916** on the log scale, against the house baseline's
**0.3871** — roughly half the baseline's loss.
