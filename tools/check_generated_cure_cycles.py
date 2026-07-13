import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_exp_params(path="exp_params.json"):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def physical_two_hold_curve(t_min, p, T_ini=20.0):
    """
    Physical two-hold cure cycle in minutes and Celsius.

    p = {
        ramp1, hold1_T, hold1_d,
        ramp2, hold2_T, hold2_d
    }
    """

    ramp1 = p["ramp1"]
    hold1_T = p["hold1_T"]
    hold1_d = p["hold1_d"]
    ramp2 = p["ramp2"]
    hold2_T = p["hold2_T"]
    hold2_d = p["hold2_d"]

    t_ramp1_end = (hold1_T - T_ini) / ramp1
    t_hold1_end = t_ramp1_end + hold1_d

    t_ramp2_end = t_hold1_end + (hold2_T - hold1_T) / ramp2
    t_hold2_end = t_ramp2_end + hold2_d

    t_end = t_min[-1]
    cooldown_rate = (T_ini - hold2_T) / max(t_end - t_hold2_end, 1e-9)

    T = np.empty_like(t_min, dtype=float)

    mask1 = t_min < t_ramp1_end
    mask2 = (t_min >= t_ramp1_end) & (t_min < t_hold1_end)
    mask3 = (t_min >= t_hold1_end) & (t_min < t_ramp2_end)
    mask4 = (t_min >= t_ramp2_end) & (t_min < t_hold2_end)
    mask5 = t_min >= t_hold2_end

    T[mask1] = T_ini + ramp1 * t_min[mask1]
    T[mask2] = hold1_T
    T[mask3] = hold1_T + ramp2 * (t_min[mask3] - t_hold1_end)
    T[mask4] = hold2_T
    T[mask5] = hold2_T + cooldown_rate * (t_min[mask5] - t_hold2_end)

    return T, {
        "t_ramp1_end": t_ramp1_end,
        "t_hold1_end": t_hold1_end,
        "t_ramp2_end": t_ramp2_end,
        "t_hold2_end": t_hold2_end,
    }


def sample_case_from_exp_params(cfg, rng):
    return {
        "ramp1": rng.uniform(cfg["r1_min"], cfg["r1_max"]),
        "hold1_T": rng.uniform(cfg["ht1_min"], cfg["ht1_max"]),
        "hold1_d": rng.uniform(cfg["hd1_min"], cfg["hd1_max"]),
        "ramp2": rng.uniform(cfg["r2_min"], cfg["r2_max"]),
        "hold2_T": rng.uniform(cfg["ht2_min"], cfg["ht2_max"]),
        "hold2_d": rng.uniform(cfg["hd2_min"], cfg["hd2_max"]),
        "htc_bot": rng.uniform(cfg["hcb_min"], cfg["hcb_max"]),
        "htc_top": rng.uniform(cfg["hct_min"], cfg["hct_max"]),
        "tool_len": rng.uniform(cfg["lt_min"], cfg["lt_max"]),
    }


def check_curve_features(t_min, T, p, stages, T_ini=20.0):
    """
    Return basic PASS/FAIL checks.
    """

    eps_T = 1.5
    eps_slope = 0.15

    # Numerical slope
    dTdt = np.gradient(T, t_min)

    # Regions
    r1_region = (t_min > 5) & (t_min < max(stages["t_ramp1_end"] - 5, 6))
    h1_region = (t_min > stages["t_ramp1_end"] + 2) & (t_min < stages["t_hold1_end"] - 2)

    r2_region = (t_min > stages["t_hold1_end"] + 2) & (t_min < stages["t_ramp2_end"] - 2)
    h2_region = (t_min > stages["t_ramp2_end"] + 2) & (t_min < stages["t_hold2_end"] - 2)

    checks = {}

    checks["start_near_T_ini"] = abs(T[0] - T_ini) < eps_T
    checks["max_not_too_high"] = T.max() <= p["hold2_T"] + eps_T
    checks["min_not_too_low"] = T.min() >= T_ini - eps_T

    if r1_region.sum() > 5:
        checks["ramp1_slope_ok"] = abs(np.median(dTdt[r1_region]) - p["ramp1"]) < eps_slope
    else:
        checks["ramp1_slope_ok"] = True

    if h1_region.sum() > 5:
        checks["hold1_temp_ok"] = abs(np.median(T[h1_region]) - p["hold1_T"]) < eps_T
    else:
        checks["hold1_temp_ok"] = True

    if r2_region.sum() > 5:
        checks["ramp2_slope_ok"] = abs(np.median(dTdt[r2_region]) - p["ramp2"]) < eps_slope
    else:
        checks["ramp2_slope_ok"] = True

    if h2_region.sum() > 5:
        checks["hold2_temp_ok"] = abs(np.median(T[h2_region]) - p["hold2_T"]) < eps_T
    else:
        checks["hold2_temp_ok"] = True

    checks["overall"] = all(checks.values())
    return checks


def main():
    cfg = load_exp_params("exp_params.json")

    out_dir = Path("verification_cure_cycles")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(65203)

    t_max_min = cfg["t_max"] / 60.0
    t_min = np.linspace(0, t_max_min, 1000)

    all_rows = []

    plt.figure(figsize=(8, 5))

    for i in range(50):
        p = sample_case_from_exp_params(cfg, rng)
        T, stages = physical_two_hold_curve(t_min, p, T_ini=cfg["T_ini"])
        checks = check_curve_features(t_min, T, p, stages, T_ini=cfg["T_ini"])

        row = {
            "case_id": i,
            **p,
            **stages,
            "T_min": float(T.min()),
            "T_max": float(T.max()),
            **checks,
        }
        all_rows.append(row)

        if i < 10:
            plt.plot(t_min, T, alpha=0.8)

    plt.xlabel("Time (min)")
    plt.ylabel("Air temperature, T_air (°C)")
    plt.title("Randomly generated two-hold cure cycles")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_generated_cure_cycles.png", dpi=200)
    plt.close()

    # Save report
    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "cure_cycle_feature_check.csv", index=False)

    print("\nSaved:")
    print(out_dir / "sample_generated_cure_cycles.png")
    print(out_dir / "cure_cycle_feature_check.csv")

    print("\nFeature check summary:")
    print(df[[
        "case_id",
        "ramp1",
        "hold1_T",
        "hold1_d",
        "ramp2",
        "hold2_T",
        "hold2_d",
        "T_min",
        "T_max",
        "overall",
    ]].head(10))

    print("\nNumber of failed cases:")
    print((~df["overall"]).sum())


if __name__ == "__main__":
    main()