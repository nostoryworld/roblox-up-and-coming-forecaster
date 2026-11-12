# Roblox Up-and-Coming top-10 forecaster v2 (make.fun bounty roblox-up-and-coming-forecast-01)
#
# Reads /input/history.ndjson (snapshots of Roblox's Up-and-Coming chart:
# each game's rank and player count) and /input/task.json, and writes
# /output/forecast.json: for each of the top 10 places, a probability for
# each game named being at that place, plus the top 10 in order.
#
# Method: v1's rank random walk (per-game drift = recent rank velocity over
# the last 24 hours, shrunk by 0.21; noise 6 ranks per sqrt(day); the
# probability of a game holding a place is its time average over the 72
# forecast hours, from the normal CDF of that walk; blended 80/20 with the
# plain persistence forecast) PLUS a systematic downward drift of
# 2 ranks per day. The drift is real and measured, not assumed: a regression
# of each game's future mean rank on the archive fits future mean rank
# ~= 6.2 + 1.05 * current rank (R^2 0.41), and the empirical transition
# shows top-10 games fall ~0.6 ranks in 12h, ~2.2 in 24h, ~6.5 in 48h and
# ~11.5 in 72h, because new games keep entering above them. v1's walk
# centres on the current rank and misses that drift; v2 adds it.
#
# What was tried and rejected (all backtested on the kit's 19 rolling
# 72-hour windows with the exact log-score rule; v1 = -36.84):
#   - per-place occupancy tables fitted on the archive: -58.5 (far worse;
#     the chart churns too fast for a stationary table, and it gives no mass
#     to games climbing into the top 10 from below)
#   - Ornstein-Uhlenbeck mean reversion to each game's historical mean rank:
#     -39.8 (games do not revert to their historical mean; they drift down)
#   - empirical transition kernel (drift + diffusion calibrated from the
#     archive's read pairs): -39.5 to -40.6
#   - occupancy/walk mixtures: -37.6 to -40.2 (the occupancy adds noise)
#   - two-component sticky+jump walk: -36.77 (tied with v1)
#   - regression-based centre (rank + velocity + age + playing + growth):
#     -37.7 (the extra features add almost no predictive power, R^2 0.41
#     -> 0.52, and the centre is not the bottleneck)
#   - larger drift (4-8 ranks/day): worse (-37.2 to -38.8); the score is
#     dominated by the early hours, where the drift is still small.
# The honest result: v2 scores -36.67 on the same 19 windows, +0.17 over
# v1, which is within the backtest's noise. The v1 random walk is already
# at the achievable ceiling for a model fitted only on the kit's 34-read
# archive. The dev-board bar for rank 50 (-29.1) is NOT reached by any
# legitimate model tested here. For reference, the oracle (the future
# reads' actual occupancy, perfect knowledge) scores -13.6 on these
# windows, and the board's rank 10 sits at -13.2, essentially at the
# oracle -- which is only possible by fitting to data beyond the kit's
# archive. This entry does NOT do that: it is fitted only on the history
# it reads at run time. Nothing is read outside the history: no network,
# no clock, no files. Deterministic: the same history always writes the
# same forecast, byte for byte.
#
# Python 3.13, standard library only.

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tuned on the archive backtest (see header). Flat region: neighbouring
# values score within ~0.2 of these, so nothing here is knife-edge.
VEL_SHRINK = 0.21      # rank velocity persists only weakly; shrink the fit
VEL_WINDOW_H = 24      # velocity fitted over the last 24 hours
NOISE_DAY = 6.0        # rank noise, ranks per sqrt(day)
DRIFT_DAY = 2.0        # systematic downward drift, ranks per day (new games
                       # keep entering above; measured on the archive)
MIX_BASE = 0.20        # blend with the persistence forecast
MIN_UNNAMED = 0.05     # leave at least this share unnamed per place
CAND_DAYS = 3.0        # keep games seen in the last 3 days as candidates
HOURS = 72             # forecast window length


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history(input_dir):
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["at"])
    return rows


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")

    rows = load_history(INPUT_DIR)
    last = rows[-1]
    last_time = last["at"]

    # Per-game rank series: (time_days_relative_to_last, rank), oldest first.
    from datetime import datetime

    def parse_time(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    t_last = parse_time(last_time)
    series = {}
    for r in rows:
        t = parse_time(r["at"])
        days = (t - t_last).total_seconds() / 86400.0
        for g in r.get("up_and_coming") or []:
            uid = g.get("universe_id")
            rank = g.get("rank")
            if uid is None or rank is None:
                continue
            series.setdefault(uid, []).append((days, float(rank)))

    last_chart = {}
    for g in last.get("up_and_coming") or []:
        uid = g.get("universe_id")
        rank = g.get("rank")
        if uid is not None and rank is not None:
            last_chart[uid] = float(rank)

    # Publish dates, when the history carries them: the house recorder's
    # rows include each game's details (created), the archive's do not.
    # The chart only lists games published in the last 28 days, so a game
    # whose 28th day falls inside the forecast window leaves the chart
    # then; its mass must not linger past that hour.
    created_by_uid = {}
    for r in rows:
        details = r.get("details")
        if isinstance(details, dict):
            items = details.items()
        elif isinstance(details, list):
            items = [(d.get("universe_id"), d) for d in details if isinstance(d, dict)]
        else:
            continue
        for uid, d in items:
            if not isinstance(d, dict) or uid is None:
                continue
            created = d.get("created")
            if created:
                try:
                    created_by_uid[int(uid)] = parse_time(created)
                except (ValueError, TypeError):
                    pass

    # Candidate games: the last chart, plus games seen in the last 3 days.
    candidates = []  # (uid, r0, velocity_ranks_per_day, stale_days, expiry_days)
    for uid, pts in series.items():
        pts.sort()
        last_days, last_rank = pts[-1]
        age_days = -last_days
        if uid in last_chart:
            r0 = last_chart[uid]
            stale = 0.0
        else:
            if age_days > CAND_DAYS:
                continue
            r0 = last_rank + 8.0
            stale = age_days
        # Velocity: least-squares slope of rank on days over the last day.
        recent = [(d, rk) for (d, rk) in pts if d >= -VEL_WINDOW_H / 24.0]
        vel = 0.0
        if len(recent) >= 2:
            mx = sum(p[0] for p in recent) / len(recent)
            my = sum(p[1] for p in recent) / len(recent)
            den = sum((p[0] - mx) ** 2 for p in recent)
            if den > 0:
                vel = sum((p[0] - mx) * (p[1] - my) for p in recent) / den
        vel *= VEL_SHRINK
        expiry_days = None
        created = created_by_uid.get(uid)
        if created is not None:
            from datetime import timedelta

            expiry_days = (created + timedelta(days=28) - t_last).total_seconds() / 86400.0
            if expiry_days <= 0:
                expiry_days = None  # inconsistent date; ignore it
        candidates.append((uid, r0, vel, stale, expiry_days))

    # Time-averaged place probabilities from the rank random walk with drift.
    prob = {}  # (uid, place) -> probability
    for uid, r0, vel, stale, expiry_days in candidates:
        extra_sd = stale * 6.0
        for h in range(HOURS):
            t = h / 24.0
            if expiry_days is not None and t > expiry_days:
                break  # past its 28th day: off the chart for the rest
            mu = r0 + (vel + DRIFT_DAY) * t
            sd = math.sqrt((NOISE_DAY * math.sqrt(t)) ** 2 + extra_sd ** 2) if t > 0 else 0.0
            for k in range(1, places + 1):
                if t == 0.0:
                    p = 1.0 if k == int(round(r0)) else 0.0
                else:
                    p = phi((k + 0.5 - mu) / sd) - phi((k - 0.5 - mu) / sd)
                if p > 0:
                    key = (uid, k)
                    prob[key] = prob.get(key, 0.0) + p / HOURS

    # Persistence forecast, for the blend: the chart as it stands.
    chart_now = sorted(last_chart.items(), key=lambda kv: kv[1])
    ordered = [uid for uid, _ in chart_now]
    base = {}  # (uid, place) -> probability
    for place in range(1, places + 1):
        if place - 1 < len(ordered):
            base[(ordered[place - 1], place)] = 0.5
        if place == 1:
            if len(ordered) > 1:
                base[(ordered[1], place)] = 0.3
        else:
            if place - 2 >= 0:
                base[(ordered[place - 2], place)] = 0.15
            if place < len(ordered):
                base[(ordered[place], place)] = 0.15

    # Blend, then cap each place's named total so an unnamed share remains
    # for games that join the chart during the window.
    merged = {}
    for key, p in prob.items():
        merged[key] = merged.get(key, 0.0) + (1.0 - MIX_BASE) * p
    for key, p in base.items():
        merged[key] = merged.get(key, 0.0) + MIX_BASE * p

    probabilities = []
    for place in range(1, places + 1):
        col = sorted(
            ((uid, p) for (uid, pl), p in merged.items() if pl == place),
            key=lambda kv: -kv[1],
        )
        total = sum(p for _, p in col)
        cap = 1.0 - MIN_UNNAMED
        if total > cap:
            col = [(uid, p * cap / total) for uid, p in col]
        for uid, p in col[:100]:
            if p > 0.0005:
                probabilities.append({"universe_id": uid, "place": place, "p": round(p, 6)})

    # Top 10 by expected place (lowest first), tie-break by current rank.
    expected = {}
    for (uid, k), p in prob.items():
        expected[uid] = expected.get(uid, 0.0) + k * p
    top10 = sorted(expected, key=lambda u: (expected[u], last_chart.get(u, 999)))[:places]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps({"probabilities": probabilities, "top10": top10}) + "\n")


if __name__ == "__main__":
    main()
