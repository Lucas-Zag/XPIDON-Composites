import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
from utils.exp_params import ExpParams

DATA_DIR = Path("debug_generated_data")

exp = ExpParams("exp_params.json")
u = np.load(DATA_DIR / "u_res_train.npy")

# 同一个 case 被重复很多次，先去重
u_unique = np.unique(np.round(u, 8), axis=0)

names = [
    "ramp1_C_per_min",
    "hold1_T_C",
    "hold1_d_min",
    "ramp2_C_per_min",
    "hold2_T_C",
    "hold2_d_min",
    "htc_bot_W_m2K",
    "htc_top_W_m2K",
    "tool_len_m",
]

phys = np.zeros_like(u_unique)

phys[:, 0] = u_unique[:, 0] * 60 / (exp.T_scaler * exp.t_max)
phys[:, 1] = u_unique[:, 1] / exp.T_scaler
phys[:, 2] = u_unique[:, 2] * exp.t_max / 60

phys[:, 3] = u_unique[:, 3] * 60 / (exp.T_scaler * exp.t_max)
phys[:, 4] = u_unique[:, 4] / exp.T_scaler
phys[:, 5] = u_unique[:, 5] * exp.t_max / 60

phys[:, 6] = u_unique[:, 6] / exp.h_scaler
phys[:, 7] = u_unique[:, 7] / exp.h_scaler
phys[:, 8] = u_unique[:, 8] / exp.len_scaler

print("\nUnique process cases:", phys.shape[0])

for i, name in enumerate(names):
    print(f"{name:18s} min={phys[:, i].min():.6f}, max={phys[:, i].max():.6f}, mean={phys[:, i].mean():.6f}")

print("\nFirst 5 physical process cases:")
for row in phys[:5]:
    print(dict(zip(names, row)))