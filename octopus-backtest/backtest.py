"""Rolling-origin backtest: TimesFM 3.0 (zero-shot, MLX) vs simple baselines on
Octopus Energy JP 30-min household consumption.

Usage:
  python octopus-backtest/backtest.py --db <octopus.db> --out <results_dir>

The DB is opened read-only. Raw consumption is never written to --out; only
aggregated metrics tables (and per-origin error rows) are.
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SLOTS_PER_DAY = {"30min": 48, "1h": 24}
MIN_CONTEXT_DAYS = 7
ORIGIN_EVERY_H = 6


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def load(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = pd.read_sql("select start_at, kwh from half_hourly order by start_at", con)
    con.close()
    df["t"] = pd.to_datetime(df.start_at, utc=True).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    s = df.set_index("t").kwh
    full = pd.date_range(s.index.min(), s.index.max(), freq="30min")
    s = s.reindex(full)
    return s


def to_gran(s, gran):
    """Returns (values_filled, observed_mask, index). Gap slots are linearly
    interpolated for use as context but masked out of evaluation targets."""
    if gran == "1h":
        # hour is observed only if both half-hours are observed
        obs = s.notna().resample("1h").sum() == 2
        v = s.resample("1h").sum(min_count=2)
    else:
        obs = s.notna()
        v = s
    filled = v.interpolate(limit_direction="both")
    return filled.values.astype(np.float64), obs.values, v.index


def tod_dow_feats(idx, spd):
    """Calendar features usable as known-future covariates."""
    frac = (idx.hour * 60 + idx.minute) / 1440.0
    wk = (idx.day_of_week >= 5).astype(float)
    return np.column_stack([
        np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac),
        np.sin(4 * np.pi * frac), np.cos(4 * np.pi * frac),
        wk,
    ])


# ---------------- baselines ----------------
def snaive(hist, h, season):
    reps = int(np.ceil(h / season))
    return np.tile(hist[-season:], reps)[:h]


def ma_flat(hist, h, spd, days=7):
    return np.full(h, hist[-days * spd:].mean())


def tod_profile(hist, h, spd, days=7):
    """Mean of the same time-of-day over the last `days` days (seasonal MA)."""
    x = hist[-days * spd:]
    n = (len(x) // spd) * spd
    prof = x[len(x) - n:].reshape(-1, spd).mean(0)
    return np.tile(prof, int(np.ceil(h / spd)))[:h]


def med_flat(hist, h, spd, days=7):
    return np.full(h, np.median(hist[-days * spd:]))


def tod_median(hist, h, spd, days=7):
    """Median of the same time-of-day over the last `days` days (MAE-friendly)."""
    x = hist[-days * spd:]
    n = (len(x) // spd) * spd
    prof = np.median(x[len(x) - n:].reshape(-1, spd), axis=0)
    return np.tile(prof, int(np.ceil(h / spd)))[:h]


def ridge_fc(hist, idx_hist, idx_fut, h, spd):
    """Ridge on calendar harmonics + weekend flag + intercept, fit on the whole history so far."""
    Xh = tod_dow_feats(idx_hist, spd)
    Xf = tod_dow_feats(idx_fut, spd)
    m = Ridge(alpha=1.0).fit(Xh, hist)
    return m.predict(Xf)


# ---------------- metrics ----------------
def mase_scale(hist, season):
    return np.mean(np.abs(hist[season:] - hist[:-season]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-timesfm", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    s = load(a.db)
    log(f"loaded {s.notna().sum()} observed half-hours, span {s.index.min()} .. {s.index.max()}, "
        f"missing slots {int(s.isna().sum())}")

    fc_model = None
    if not a.no_timesfm:
        from timesfm3.mlx import TimesFM3Forecaster
        fc_model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
        log("TimesFM 3.0 (MLX) loaded")

    rows = []
    for gran in ["30min", "1h"]:
        spd = SLOTS_PER_DAY[gran]
        v, obs, idx = to_gran(s, gran)
        step_h = 24 / spd
        for hname, hdays in [("24h", 1), ("7d", 7)]:
            h = hdays * spd
            first = MIN_CONTEXT_DAYS * spd
            origins = [o for o in range(first, len(v) - h + 1, int(ORIGIN_EVERY_H / step_h))]
            # keep only origins whose target window has at least 80% observed
            origins = [o for o in origins if obs[o:o + h].mean() >= 0.8]
            log(f"{gran} {hname}: {len(origins)} origins")
            if not origins:
                continue
            preds = {m: [] for m in ["snaive_day", "snaive_week", "ma7d_flat", "ma7d_tod", "med7d_flat", "med7d_tod", "ridge_cal"]}
            ctxs = [v[:o] for o in origins]
            for o in origins:
                hist = v[:o]
                preds["snaive_day"].append(snaive(hist, h, spd))
                preds["snaive_week"].append(snaive(hist, h, 7 * spd))
                preds["ma7d_flat"].append(ma_flat(hist, h, spd))
                preds["ma7d_tod"].append(tod_profile(hist, h, spd))
                preds["med7d_flat"].append(med_flat(hist, h, spd))
                preds["med7d_tod"].append(tod_median(hist, h, spd))
                preds["ridge_cal"].append(ridge_fc(hist, idx[:o], idx[o:o + h], h, spd))
            if fc_model is not None:
                t0 = time.time()
                outs = list(fc_model.predict_batch([c.astype(np.float32) for c in ctxs], horizon=h))
                preds["timesfm3_zeroshot"] = [np.asarray(o.forecast, dtype=np.float64) for o in outs]
                log(f"  TimesFM zero-shot {time.time()-t0:.1f}s")
                # with known-future calendar covariates (past_future_covariates)
                t0 = time.time()
                pf = []
                for o, c in zip(origins, ctxs):
                    cov = tod_dow_feats(idx[:o + h], spd).T.astype(np.float32)  # (5, o+h)
                    r = fc_model.predict(c.astype(np.float32), horizon=h, past_future_covariates=cov)
                    pf.append(np.asarray(r.forecast, dtype=np.float64).reshape(-1)[:h])
                preds["timesfm3_calcov"] = pf
                log(f"  TimesFM +calendar covariates {time.time()-t0:.1f}s")
            for m, plist in preds.items():
                for o, p, c in zip(origins, plist, ctxs):
                    y = v[o:o + h]
                    mk = obs[o:o + h]
                    e = (p - y)[mk]
                    sc_d = mase_scale(c, spd)
                    rows.append(dict(gran=gran, horizon=hname, model=m, origin=str(idx[o]), n=int(mk.sum()),
                                     mae=np.abs(e).mean(), mse=(e ** 2).mean(),
                                     scale_snaive_day=sc_d, mean_y=y[mk].mean()))
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_origin_errors.csv", index=False)

    # aggregate: MAE/RMSE pooled over origins (weight by n), MASE = pooled MAE / mean in-sample daily-seasonal naive MAE
    def agg(g):
        w = g.n.values
        mae = np.average(g.mae, weights=w)
        rmse = np.sqrt(np.average(g.mse, weights=w))
        return pd.Series(dict(origins=len(g), MAE_kWh=mae, RMSE_kWh=rmse,
                              MASE=mae / g.scale_snaive_day.mean(),
                              MAE_W=mae * (2000 if g.name[0] == "30min" else 1000)))
    tab = per.groupby(["gran", "horizon", "model"]).apply(agg, include_groups=False).reset_index()
    tab.to_csv(out / "metrics.csv", index=False)
    with open(out / "metrics.md", "w") as f:
        for (g, hh), t in tab.groupby(["gran", "horizon"]):
            f.write(f"\n### {g} / {hh}  (origins={int(t.origins.iloc[0])})\n\n")
            f.write(t.drop(columns=["gran", "horizon", "origins"]).sort_values("MAE_kWh")
                    .to_markdown(index=False, floatfmt=".4f") + "\n")

    # paired bootstrap over origins: TimesFM vs each other model, MAE difference
    rng = np.random.default_rng(0)
    bs = []
    if fc_model is not None:
        for (g, hh), d in per.groupby(["gran", "horizon"]):
            pv = d.pivot(index="origin", columns="model", values="mae")
            for m in pv.columns:
                if m.startswith("timesfm3"):
                    continue
                for tm in ["timesfm3_zeroshot", "timesfm3_calcov"]:
                    diff = (pv[tm] - pv[m]).values  # negative => TimesFM better
                    # origins are 6h apart with overlapping windows: resample whole days (4 origins/day at 30min grid)
                    per_day = int(24 / ORIGIN_EVERY_H)
                    blocks = [diff[i:i + per_day] for i in range(0, len(diff), per_day)]
                    bm = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]).mean()
                          for _ in range(2000)]
                    bs.append(dict(gran=g, horizon=hh, timesfm=tm, vs=m, mean_mae_diff=diff.mean(),
                                   ci_lo=np.percentile(bm, 2.5), ci_hi=np.percentile(bm, 97.5),
                                   frac_origins_timesfm_better=(diff < 0).mean(), origins=len(diff)))
        pd.DataFrame(bs).to_csv(out / "bootstrap_vs_baselines.csv", index=False)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
