"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Model, in count space. A trophy's earn rate is holders / players-tracked, and the
page Exophase prints is a snapshot as of its last refresh, not a live count: the
practice shows the page frozen for hours (638 players, every rate unchanged) and
then jumping in one refresh (3,868 players, every rate but the one at 100%
lower). So the forecast is built from two quantities, not from the rates:

  P(t)  -- players tracked,   E_i(t) -- holders of trophy i.

rate_i(target) = E_i(target) / P(target), modelled as log-normal in each leg:
log rate ~ Normal(log E_med - log P_med, sqrt(sE^2 + sP^2)), quantiles via the
standard normal z_tau. Medians move (new players keep arriving, holders keep
earning); the spread is honest about refresh staleness and extrapolation.

Two regimes, picked from the case's own history (consecutive identical reads
are one snapshot -- the page was just not refreshed):
  * flat -- a single distinct snapshot (the practice): no trend is
    estimable, so launch-night priors carry the medians. The page's snapshot
    is stale since the first read, so growth is counted from the first read
    to the target: P grows exp(kP * hours), holders exp(kE * hours), both
    saturating on long horizons (exponential growth cannot persist for days).
  * rich -- two or more distinct snapshots (the final, ~2 days of reads):
    a cohort model. Players arrive over time (P(t) forecast from its recent
    growth rate, decaying); a player of age a holds trophy i with probability
    R_i * (1 - exp(-a / tau_i)). Holders are the convolution of arrivals
    with that curve, plus the maturation of the opening cohort:
    E_i(t) = R_i * K_tau_i(t), linear in R_i given tau_i, fitted by least
    squares over a tau grid, then read at the target with the fitted P(t).
    This respects saturation: as cohorts age and arrivals slow, rates climb
    toward R_i instead of exploding like a log-log trend would. With few
    snapshots the rich median is shrunk toward persistence and the spread
    widened (a smooth blend, no cliff at any snapshot count).
  * hedge -- in the rich regime the median is shrunk part-way from the last
    rate toward the model rate, because the post-deadline regime can flip
    (a player-base surge dilutes rates) and a far-moved median is the exposed
    leg. Spread comes from fit residuals plus staleness and horizon terms.

A trophy at 100% in the latest non-null read stays ~100% (every tracked
player holds it, through a 6x player influx on launch night).

Standard library only. Deterministic: no randomness, no network, no time
dependence beyond the case file, so two runs write identical bytes. A case
that cannot be forecast (no history, malformed fields) still gets a valid
wide no-information output instead of aborting the run.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch-night priors, from the one refresh the competition's own rules
# document (638 -> 3,868 players, 943 -> 4,276 trophies earned, over ~4.7 h):
#   kP = ln(3868/638)/4.67 ~= 0.386/h, kE = ln(408/305)/4.67 ~= 0.062/h
# (holders ex-100%). Slightly conservative values; the spread below carries
# the uncertainty.
DEFAULTS = {
    "k_p_flat": 0.38,      # per-hour exponential growth of players tracked
    "k_e_flat": 0.05,      # per-hour exponential growth of holders
    "p_mult_cap": 50.0,    # flat regime: P never forecast above 50x the snapshot
    "sigma_flat": 0.40,    # total log-spread of the rate, flat regime
    "sigma_e_flat": 0.25,  # holder-leg log-spread, flat regime
    "sigma_stale": 0.30,   # answer snapshot may lag the target by hours
    "sigma_horizon": 0.08,  # extra log-spread per sqrt(day) of extrapolation
    "sigma_cap": 1.20,
    "sigma_floor": 0.35,
    "a0_hours": 3.0,       # assumed age (h) of the opening cohort at first read
    "small_n_spread": 0.60,  # extra log-spread when the rich fit has few snapshots
    "median_shrink": 0.65,  # rich-regime median hedge toward persistence
}


def parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_release(game: dict) -> datetime:
    raw = str(game.get("release_date", "2026-11-19"))
    if "T" in raw:
        return parse_time(raw)
    return parse_time(raw + "T00:00:00Z")


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, |err| < 1.2e-9).
    Only used when a case asks for levels other than the scored five."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, like JavaScript's Math.round."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def growth_mult(k: float, hours: float, tail_exp: float) -> float:
    """Growth multiplier over `hours`: exponential while the window is short
    (launch-night behaviour), then a slow power-law tail -- exponential
    growth cannot persist for days, so the flat regime must saturate instead
    of exploding on a long horizon."""
    if hours <= 12.0:
        return math.exp(k * hours)
    return math.exp(k * 12.0) * (1.0 + (hours - 12.0) / 36.0) ** tail_exp


def distinct_snapshots(case: dict):
    """Collapse consecutive identical reads (same players + same rates) into
    one snapshot each: the page simply was not refreshed between them."""
    snaps = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for _key, read in snaps]


def ols(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (a, b, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2) if n > 2 else 0.0
    return a, b, math.sqrt(var)


def growth_rate_fit(snaps, target, release, tau_decay=24.0):
    """Forecast players tracked from the recent growth rate, decaying.

    g = the log-growth rate of P over the recent distinct snapshots (OLS on
    log P vs time, last 24 h of snapshots, clamped to [0, 0.15]/h; with
    fewer than 2 recent points, the last two snapshots). Future growth is
    assumed to decay exponentially with timescale tau_decay, so
    P(target) = P_last * exp(g * tau_decay * (1 - exp(-H / tau_decay))),
    H = hours from the last snapshot to the target. A saturated page (g~0)
    stays flat; an exploding one is bounded instead of extrapolated to
    infinity. Returns (P_med_target, sigma_logP), or (None, None)."""
    pts = []
    for read in snaps:
        h = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        pts.append((h, math.log(max(1, read["players_tracked"]))))
    if len(pts) < 2:
        return None, None
    t_last, logp_last = pts[-1]
    recent = [(h, y) for h, y in pts if h >= t_last - 24.0]
    if len(recent) < 2:
        recent = pts[-2:]
    _a, g, sd = ols([h for h, _y in recent], [y for _h, y in recent])
    g = min(0.15, max(0.0, g))
    h_t = max(0.0, (target - release).total_seconds() / 3600.0)
    H = max(0.0, h_t - t_last)
    log_growth = g * tau_decay * (1.0 - math.exp(-H / tau_decay)) if g > 0 else 0.0
    p_med = math.exp(logp_last + log_growth)
    sig = math.sqrt(max(0.05, sd) ** 2 + (0.35 * log_growth) ** 2 + 0.15 ** 2)
    return p_med, sig


TAU_GRID = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def cohort_kernels(times_h, p_hist, t_target, p_target, a0=3.0):
    """Precompute, per tau, K_tau(t) for the cohort model, once per case.

    K_tau(t) = P_first * (1 - exp(-(a0 + t - t_first)/tau))   [opening cohort
    maturing: the players present at the first snapshot keep earning]
    + H_tau(t)                                                [later arrivals]
    where H_tau(t) = sum of arrivals dP weighted by (1 - exp(-age/tau)).
    Holders are E(t) = R * K_tau(t): linear in R given tau, so each trophy
    fits R by least squares. H is built incrementally per tau in one pass:
    G_j = G_{j-1} * exp(-dt/tau) + dP_j, H_j = (P_j - P_first) - G_j.
    Returns {tau: ([K at snapshot times], K at target)}."""
    t_first = times_h[0]
    p_first = p_hist[0]
    # Timeline: snapshot times, then 2-hourly steps along a log-linear P path
    # from the last snapshot to the target (the fitted P curve's arrivals).
    ext_t, ext_p = [t_first], [p_first]
    for j in range(1, len(times_h)):
        ext_t.append(times_h[j])
        ext_p.append(p_hist[j])
    if t_target > times_h[-1] and p_target > p_hist[-1]:
        n_step = max(1, int((t_target - times_h[-1]) / 2.0))
        log0, log1 = math.log(p_hist[-1]), math.log(p_target)
        for s in range(1, n_step + 1):
            tt = times_h[-1] + (t_target - times_h[-1]) * s / n_step
            ext_t.append(tt)
            ext_p.append(math.exp(log0 + (log1 - log0) * s / n_step))
    n_snap = len(times_h)
    out = {}
    for tau in TAU_GRID:
        g = 0.0
        prev_t, prev_p = ext_t[0], ext_p[0]
        ks = []
        for j in range(len(ext_t)):
            if j > 0:
                dt = ext_t[j] - prev_t
                dp = max(0.0, ext_p[j] - prev_p)
                g = g * math.exp(-dt / tau) + dp
                prev_t, prev_p = ext_t[j], ext_p[j]
            age = a0 + (ext_t[j] - t_first)
            h_t = (ext_p[j] - p_first) - g
            ks.append(p_first * (1.0 - math.exp(-age / tau)) + max(0.0, h_t))
        out[tau] = (ks[:n_snap], ks[-1])
    return out


def cohort_e_fit(e_obs, kernels):
    """Fit one trophy's holders by the cohort model: E(t) = R * K_tau(t).

    For each tau on the grid, R is the exact least-squares fit (clamped to
    [0, 1]: a probability), and the tau with the smallest SSE wins.
    Returns (E_med_target, sigma_logE)."""
    best = None
    for tau, (ks, k_target) in kernels.items():
        num = sum(k * e for k, e in zip(ks, e_obs))
        den = sum(k * k for k in ks)
        if den <= 0:
            continue
        r = min(1.0, max(0.0, num / den))
        sse = sum((e - r * k) ** 2 for k, e in zip(ks, e_obs))
        if best is None or sse < best[0]:
            best = (sse, tau, r)
    if best is None:
        return max(0.05, e_obs[-1]), 0.8
    _sse, tau, r = best
    ks, k_target = kernels[tau]
    e_med = max(0.05, r * k_target)
    errs = []
    for j, e in enumerate(e_obs):
        fit = r * ks[j]
        if e > 0 and fit > 0:
            errs.append(math.log(e) - math.log(fit))
    sig = math.sqrt(sum(x * x for x in errs) / len(errs)) if errs else 0.5
    return e_med, max(0.10, sig)


def no_info_output(case: dict) -> dict:
    """Valid output when a case cannot be forecast (no usable history):
    every trophy present, wide honest quantiles, no crash, no missing file."""
    return {"trophies": [{"id": t["id"], "q": [0.01, 1.0, 10.0, 50.0, 99.99]}
                         for t in case["trophies"]]}


def forecast_case(case: dict, params: dict | None = None) -> dict:
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    levels = case["levels"]
    z = Z if levels == LEVELS else [norm_ppf(l) for l in levels]
    if not case["history"]:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    last = case["history"][-1]
    target = parse_time(case["target"])
    release = parse_release(case["game"])
    horizon_days = case["horizon_minutes"] / 1440.0

    snaps = distinct_snapshots(case)
    n_snaps = len(snaps)
    first = snaps[0]
    t_first = parse_time(first["at"])
    hours_grow = (target - t_first).total_seconds() / 3600.0
    t_target_h = max(0.0, (target - release).total_seconds() / 3600.0)

    # Latest non-null rate per trophy (for the 100% rule and last-E).
    latest_rate = []
    last_E = []
    for i in range(n_tr):
        rate = None
        for read in reversed(case["history"]):
            if read["rates"][i] is not None:
                rate = read["rates"][i]
                last_E.append(round(rate * read["players_tracked"] / 100.0))
                break
        if rate is None:
            last_E.append(0)
        latest_rate.append(rate)

    if n_snaps == 1:
        # ---- flat regime: launch-night priors from the (stale) snapshot ----
        p_mult = min(p["p_mult_cap"], growth_mult(p["k_p_flat"], hours_grow, 0.8))
        p_med = max(1.0, first["players_tracked"]) * p_mult
        g_e = growth_mult(p["k_e_flat"], hours_grow, 0.4)
        e_med = [max(0.05, e * g_e) for e in last_E]
        sig = [math.sqrt(p["sigma_flat"] ** 2
                         + (p["sigma_horizon"] * math.sqrt(horizon_days)) ** 2)] * n_tr
        shrink = 0.0
    else:
        # ---- rich regime: cohort model, smoothly blended by snapshot count ----
        w = min(1.0, (n_snaps - 1) / 4.0)
        p_fit, sig_p_fit = growth_rate_fit(snaps, target, release)
        if p_fit is None:
            # Unreachable with >= 2 snapshots (growth_rate_fit needs 2
            # points), kept as a guard: never let a degenerate P fit crash
            # the case; fall back to flat priors from the snapshot.
            p_med = max(1.0, first["players_tracked"]) * min(
                p["p_mult_cap"], growth_mult(p["k_p_flat"], hours_grow, 0.8))
            sig_p = math.sqrt(max(0.0, p["sigma_flat"] ** 2 - p["sigma_e_flat"] ** 2))
            e_med = [max(0.05, e * growth_mult(p["k_e_flat"], hours_grow, 0.4)) for e in last_E]
            sig_e = [p["sigma_e_flat"]] * n_tr
            sig = [p["sigma_flat"]] * n_tr
            shrink = 0.0
        else:
            p_med = p_fit
            sig_p = sig_p_fit
            times_h = [max(0.0, (parse_time(r["at"]) - release).total_seconds() / 3600.0)
                       for r in snaps]
            p_hist = [float(r["players_tracked"]) for r in snaps]
            kernels = cohort_kernels(times_h, p_hist, t_target_h, p_med, p["a0_hours"])
            e_med, sig_e = [], []
            for i in range(n_tr):
                e_obs = []
                for read in snaps:
                    r = read["rates"][i]
                    e_obs.append(round(r * read["players_tracked"] / 100.0)
                                 if r is not None else None)
                # Drop leading-null snapshots from this trophy's fit and
                # start its holder count at the first observed read.
                first_obs = next((j for j, v in enumerate(e_obs) if v is not None), None)
                if first_obs is None:
                    e_med.append(0.05)
                    sig_e.append(0.8)
                    continue
                sub_k = {tau: (ks[first_obs:], kt) for tau, (ks, kt) in kernels.items()}
                em, se = cohort_e_fit(e_obs[first_obs:], sub_k)
                e_med.append(em)
                sig_e.append(se)
            shrink = p["median_shrink"] * w
            extra = (1.0 - w) * p["small_n_spread"]
            sig = [min(p["sigma_cap"], math.sqrt(
                sig_p ** 2 + sig_e[i] ** 2 + p["sigma_stale"] ** 2
                + (p["sigma_horizon"] * math.sqrt(horizon_days)) ** 2 + extra ** 2))
                for i in range(n_tr)]
            sig = [max(p["sigma_floor"], s) for s in sig]

    rows = []
    for i, trophy in enumerate(trophies):
        if latest_rate[i] is not None and latest_rate[i] == 100.0:
            rows.append({"id": trophy["id"], "q": [99.0, 99.8, 100, 100, 100]})
            continue
        e = min(e_med[i], 0.995 * p_med)
        med_rate = 100.0 * e / p_med
        lg = logit(med_rate)
        if shrink and latest_rate[i] is not None:
            # Hedge the median toward persistence: the post-deadline regime
            # can flip (a player-base surge dilutes rates), and a median
            # that has moved a long way from the last rate is the exposed leg.
            lg = logit(latest_rate[i]) + shrink * (lg - logit(latest_rate[i]))
        s = sig[i]
        qs = [round4(clamp(inv_logit(lg + zz * s))) for zz in z]
        qs = [max(qs[k], qs[k - 1]) if k else qs[0] for k in range(len(qs))]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def dump_forecast(forecast: dict) -> str:
    """Serialize, shrinking precision if the 65,536-byte cap binds."""
    text = json.dumps(forecast, separators=(",", ":")) + "\n"
    if len(text.encode("utf-8")) <= 65536:
        return text
    slim = {"trophies": [{"id": r["id"], "q": [round(q, 2) for q in r["q"]]}
                         for r in forecast["trophies"]]}
    return json.dumps(slim, separators=(",", ":")) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue  # unparseable file: no trophy list to write rows for
        try:
            forecast = forecast_case(case)
        except Exception:
            # One bad case must never abort the run or poison its siblings.
            try:
                forecast = no_info_output(case)
            except Exception:
                continue
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            dump_forecast(forecast), encoding="utf-8")


if __name__ == "__main__":
    main()
