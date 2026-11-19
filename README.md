# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kazamisao/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kazamisao/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

GTA VI trophy rarity forecaster — count-space cohort model: forecast every GTA VI trophy's earn rate on Exophase at 2026-11-26T00:00Z, as five quantiles per trophy, scored by mean pinball loss on the logit scale.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` reads every `*.in` in the input directory and writes `<id>.out`
for each. License: MIT.

## The model (count space, not rate space)

A trophy's earn rate is `holders / players-tracked`, and the page is a
snapshot as of Exophase's last refresh, not a live count: on launch night the
page sat frozen for hours (638 players, every rate unchanged) and then jumped
in one refresh (3,868 players, every rate but the one at 100% lower). So the
forecast is built from two quantities, `P(t)` (players tracked) and `E_i(t)`
(holders of trophy i), with `rate_i = E_i / P`, each leg log-normal.

- **Flat history** (the practice: all reads identical): no trend is
  estimable, so launch-night priors carry the medians. The page's snapshot is
  stale since the first read, so growth is counted from the first read to the
  target: players `exp(0.38/h)`, holders `exp(0.05/h)`, saturating on long
  horizons. Median rates fall steeply (dilution), with an honest spread.
- **Rich history** (the final: ~2 days of reads): a cohort model. Players
  arrive over time (`P(t)` forecast from its recent growth rate, decaying);
  a player of age `a` holds trophy `i` with probability
  `R_i * (1 - exp(-a / tau_i))`. Holders are the convolution of arrivals
  with that curve: `E_i(t) = E0_i + R_i * H_tau_i(t)`, linear in `R_i` given
  `tau_i`, fitted by least squares over a tau grid. This respects
  saturation: as cohorts age and arrivals slow, rates climb toward `R_i`
  instead of exploding like a log-log trend would.
- **Hedge**: in the rich regime the median is shrunk 65% of the way from the
  last rate toward the model rate, because the post-deadline regime can flip
  (a player-base surge dilutes rates) and a far-moved median is the exposed
  leg. Spread comes from fit residuals plus staleness and horizon terms.
- A trophy at 100% in the last read stays ~100%.

## Local verification

- Scorer reproduces the RULES worked example exactly (loss 0.403404).
- Practice (reconstructed answers from the documented 04:30Z refresh:
  638 -> 3,868 players, 943 -> 4,276 trophies earned; three allocations of
  the 103 new holders + two stress variants): mean loss **~0.13** vs the
  house baseline's ~1.51, worst reconstruction 0.18, coverage ~0.98.
- Synthetic final cases (cohort ground truth, logistic and power-law player
  growth, refresh lumpiness, a weekend second wave): beats both the
  persistence baseline and the house curve baseline on every scenario
  (typical loss 0.17-0.34 vs 0.34-0.49 and 0.23-5.70); the harsh
  +50%-second-wave stress case lands at 0.34, level with persistence.
- Deterministic, stdlib only, ~0.1 s per case (cap 8 s); a bad case can
  never abort the run (per-case isolation, valid no-information output).

---

<sub>Entered on [make.fun](https://make.fun/?ref=p5dwa49p&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
