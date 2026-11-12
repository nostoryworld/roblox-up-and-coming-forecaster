# Roblox Up-and-Coming top-10 forecaster

Entry for the make.fun bounty **Predict Roblox's Up-and-Coming top 10**
(`roblox-up-and-coming-forecast-01`, $500). A program that reads the
history of Roblox's Up-and-Coming chart and forecasts its top 10, with a
probability for each game at each place, for every hour of the 72 hours
after entries close.

## How it works

`main.py` (Python 3.13, standard library only) reads
`/input/history.ndjson` and `/input/task.json` and writes
`/output/forecast.json`.

Each game's rank is treated as a random walk with a small drift: the drift
is the game's recent rank velocity over the last 24 hours (linear fit,
shrunk by 0.21, because rank velocity barely persists), and the noise grows
as 6 ranks per sqrt(day). The probability of a game holding a place is its
time average over the 72 forecast hours, from the normal CDF of that walk.
The result is blended 80/20 with the plain persistence forecast (the chart
barely moves in the first hours). Games that left the chart in the last 3
days are kept as candidates with a penalty, in case they climb back. Where
the history carries a game's publish date, its mass stops at its 28th day,
when it ages off the chart. No network, no clock, no files outside the
history. Deterministic: the same history always writes the same forecast.

## Backtest

19 rolling 72-hour windows on the kit's archive (Oct 30 – Nov 7 2026),
scored with the exact log-score rule: mean **-36.84** against the
persistence baseline's **-46.60**, better in every single window. The house
baseline scores -43.51 on the practice window.

## Run it locally

```
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Referral

- Bounty: https://make.fun/bounty/roblox-up-and-coming-forecast-01
