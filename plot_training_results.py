import os
import glob
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from utils.phy_params import PhyParams
from utils.exp_params import ExpParams
from models.pidon import XPIDON


OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOSS_DIR = os.path.join(OUT_DIR, "loss_data")
MODEL_DIR = os.path.join(OUT_DIR, "models")
PRED_DIR = os.path.join(OUT_DIR, "predictions")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


def extract_tag(path):
    name = os.path.basename(path)
    m = re.search(r"xpidon_class_(\d+)_(T|a)\.pkl", name)
    if not m:
        raise ValueError(f"Cannot extract tag from {name}")
    return m.group(1)


def tag_to_fraction(tag):
    return int(tag) / 10000.0


def safe_semilogy(history, key, label):
    values = history.get(key, [])
    if len(values) > 0:
        plt.semilogy(values, label=label)


# ============================================================
# 1. Merge all loss data
# ============================================================
loss_files = sorted(glob.glob(os.path.join(LOSS_DIR, "loss_history_*.pkl")))

if not loss_files:
    raise FileNotFoundError("No loss_history_*.pkl files found in outputs/loss_data")

loss_history = {}

print("Using loss files:")
for lf in loss_files:
    print(" ", lf)
    with open(lf, "rb") as f:
        h = pickle.load(f)

    for k, v in h.items():
        loss_history.setdefault(k, [])
        loss_history[k].extend(v)

print("\nMerged loss length:")
for k, v in loss_history.items():
    print(k, len(v))

merged_loss_file = os.path.join(LOSS_DIR, "loss_history_merged.pkl")
with open(merged_loss_file, "wb") as f:
    pickle.dump(loss_history, f)

print("Saved:", merged_loss_file)


# ============================================================
# 2. Temperature network loss figure
# ============================================================
plt.figure(figsize=(10, 6))

safe_semilogy(loss_history, "loss_T", "Total temperature loss")
safe_semilogy(loss_history, "loss_res", "Part PDE residual")
safe_semilogy(loss_history, "loss_res_tool", "Tool PDE residual")
safe_semilogy(loss_history, "loss_bct", "Top BC")
safe_semilogy(loss_history, "loss_bcb", "Bottom BC")
safe_semilogy(loss_history, "loss_ics_T", "Part IC")
safe_semilogy(loss_history, "loss_ics_tool", "Tool IC")
safe_semilogy(loss_history, "loss_inf", "Interface temperature")
safe_semilogy(loss_history, "loss_flux", "Interface flux")

if not plt.gca().has_data():
    plt.text(0.5, 0.5, "No temperature loss data saved.\nRun more iterations or record T loss in alpha block.",
             ha="center", va="center", transform=plt.gca().transAxes)

plt.xlabel("Logged training step")
plt.ylabel("Loss")
plt.title("Temperature Network Loss Components")
plt.legend(fontsize=8)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()

temperature_loss_path = os.path.join(FIG_DIR, "temperature_network_loss.png")
plt.savefig(temperature_loss_path, dpi=300)
plt.close()

print("Saved:", temperature_loss_path)


# ============================================================
# 3. Degree of cure network loss figure
# ============================================================
plt.figure(figsize=(8, 5))

safe_semilogy(loss_history, "loss_a", "Total degree of cure loss")
safe_semilogy(loss_history, "loss_ode", "Cure ODE residual")
safe_semilogy(loss_history, "loss_ics_a", "Degree of cure IC")

if not plt.gca().has_data():
    plt.text(0.5, 0.5, "No degree of cure loss data saved.",
             ha="center", va="center", transform=plt.gca().transAxes)

plt.xlabel("Logged training step")
plt.ylabel("Loss")
plt.title("Degree of Cure Network Loss Components")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()

doc_loss_path = os.path.join(FIG_DIR, "doc_network_loss.png")
plt.savefig(doc_loss_path, dpi=300)
plt.close()

print("Saved:", doc_loss_path)


# ============================================================
# 4. Prediction by subdomain
# ============================================================
phy_params = PhyParams("phy_params.json")
exp_params = ExpParams("exp_params.json")

m_inp = 9
branch_layers = [m_inp, 50, 50, 50, 100]
trunk_layers = [2, 50, 50, 50, 50, 50, 100]
nomad_layers_T = [100, 50, 50, 50, 50, 1]

T_files = sorted(glob.glob(os.path.join(MODEL_DIR, "xpidon_class_*_T.pkl")))
a_files = sorted(glob.glob(os.path.join(MODEL_DIR, "xpidon_class_*_a.pkl")))

if not T_files or not a_files:
    raise FileNotFoundError("No model files found in outputs/models")

T_by_tag = {extract_tag(f): f for f in T_files}
a_by_tag = {extract_tag(f): f for f in a_files}

tags = sorted(set(T_by_tag.keys()) & set(a_by_tag.keys()), key=lambda x: int(x))

print("\nUsing model tags:")
for tag in tags:
    print(tag, T_by_tag[tag], a_by_tag[tag])

u_mid = (np.array(exp_params.minvals) + np.array(exp_params.maxvals)) / 2.0

all_time_physical = []
all_T = []
all_a = []

previous_fraction = 0.0

for tag in tags:
    end_fraction = tag_to_fraction(tag)

    if end_fraction <= previous_fraction:
        print("Skipping invalid interval:", tag)
        continue

    t_physical_min = previous_fraction * exp_params.t_max
    t_physical_max = end_fraction * exp_params.t_max

    model = XPIDON(
        phy_params,
        exp_params,
        branch_layers,
        trunk_layers,
        nomad_layers_T,
        t_physical_min,
        t_physical_max
    )

    with open(T_by_tag[tag], "rb") as f:
        T_params = pickle.load(f)

    with open(a_by_tag[tag], "rb") as f:
        a_params = pickle.load(f)

    nt = 120

    # Physical time only for plotting
    t_physical = np.linspace(t_physical_min, t_physical_max, nt)

    # Important:
    # Network input time is local normalized subdomain time, not full physical time.
    t_local = np.linspace(0.0, 1.0, nt)

    x_fixed = 0.5 * np.ones_like(t_local)

    Y_star = np.column_stack([t_local, x_fixed])
    U_star = np.tile(u_mid, (nt, 1))

    T_pred = model.pred_T(T_params, jnp.array(U_star), jnp.array(Y_star))
    a_pred = model.pred_a(a_params, jnp.array(U_star), jnp.array(Y_star))

    all_time_physical.append(t_physical)
    all_T.append(np.array(T_pred))
    all_a.append(np.array(a_pred))

    previous_fraction = end_fraction

if not all_time_physical:
    raise RuntimeError("No prediction data generated.")

all_time_physical = np.concatenate(all_time_physical)
all_T = np.concatenate(all_T)
all_a = np.concatenate(all_a)

order = np.argsort(all_time_physical)
all_time_physical = all_time_physical[order]
all_T = all_T[order]
all_a = all_a[order]

prediction_data = {
    "time_seconds": all_time_physical,
    "time_minutes": all_time_physical / 60.0,
    "temperature_prediction": all_T,
    "degree_of_cure_prediction": all_a,
}

prediction_pkl = os.path.join(PRED_DIR, "final_temperature_doc_prediction.pkl")
with open(prediction_pkl, "wb") as f:
    pickle.dump(prediction_data, f)

prediction_csv = os.path.join(PRED_DIR, "final_temperature_doc_prediction.csv")
np.savetxt(
    prediction_csv,
    np.column_stack([
        all_time_physical,
        all_time_physical / 60.0,
        all_T,
        all_a
    ]),
    delimiter=",",
    header="time_seconds,time_minutes,temperature_prediction,degree_of_cure_prediction",
    comments=""
)

print("Saved:", prediction_pkl)
print("Saved:", prediction_csv)


# ============================================================
# 5. Final prediction figure
# ============================================================
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(all_time_physical / 60.0, all_T, label="Predicted temperature")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("Temperature")
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(all_time_physical / 60.0, all_a, linestyle="--", label="Predicted degree of cure")
ax2.set_ylabel("Degree of cure")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Predicted Temperature and Degree of Cure")
plt.tight_layout()

prediction_fig = os.path.join(FIG_DIR, "final_temperature_doc_prediction.png")
plt.savefig(prediction_fig, dpi=300)
plt.close()

print("Saved:", prediction_fig)
print("\nDone. All generated files are inside outputs/.")