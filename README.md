# GTA VI Twitch CCU forecast — make.fun entry

Entry for `gta6-twitch-ccu-forecast-01`: forecast the Twitch concurrent
viewers of the Grand Theft Auto VI category for hours 2–24 of release day,
as five quantiles per hour, from everything recorded up to the end of hour 1.

## Run

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out`
per case. Python 3 standard library only, offline, deterministic.

## Method

Each case carries past launches ("comparables") with full hourly curves.
A launch's pre-checkpoint hours trace its unlock staircase; launches whose
staircase shape and hour-1 scale resemble the target's tend to unfold
alike. The forecast is a similarity-weighted distribution of the
comparables' log-ratios ln(viewers_h / viewers_1): Gaussian-kernel weights
on profile distance and log-scale distance, per-hour weighted empirical
quantiles blended with a weighted normal fit. Fitted on the kit's practice
launches; on them it scores a mean pinball loss of 0.1916 against the house
baseline's 0.3871.
