import os
import glob
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from utils.air_temp import Temp_air
from utils.phy_params import PhyParams
from utils.exp_params import ExpParams
from models.pidon import XPIDON


OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOSS_DIR = os.path.join(OUT_DIR, "loss_data")
MODEL_DIR = os.path.join(OUT_DIR, "models")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
# Set this to manually select the current run's subdomain models.
# Use None only when outputs/ is clean.
RUN_TAGS = None
# Example for 3 subdomains:
#RUN_TAGS = ["03333", "06667", "10000"]
# Example for 5 subdomains:
# RUN_TAGS = ["02000", "04000", "06000", "08000", "10000"]

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
#loss_files = sorted(glob.glob(os.path.join(LOSS_DIR, "loss_history_*.pkl")))
loss_files = sorted(
    f for f in glob.glob(os.path.join(LOSS_DIR, "loss_history_*.pkl"))
    if os.path.basename(f) != "loss_history_merged.pkl"
)


if not loss_files:
    raise FileNotFoundError("No loss_history_*.pkl files found in outputs/loss_data")

loss_history = {}
loss_file_histories = []

print("Using loss files:")
for lf in loss_files:
    print(" ", lf)
    with open(lf, "rb") as f:
        h = pickle.load(f)
    loss_file_histories.append((os.path.basename(lf), h))
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

# Draw subdomain boundaries for temperature loss plot
temp_lengths = [len(h.get("loss_T", [])) for _, h in loss_file_histories]
temp_boundaries = np.cumsum(temp_lengths)[:-1]

for b in temp_boundaries:
    plt.axvline(b, color="gray", linestyle=":", linewidth=1, alpha=0.6)

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

# Draw subdomain boundaries for degree-of-cure loss plot
doc_lengths = [len(h.get("loss_a", [])) for _, h in loss_file_histories]
doc_boundaries = np.cumsum(doc_lengths)[:-1]

for b in doc_boundaries:
    plt.axvline(b, color="gray", linestyle=":", linewidth=1, alpha=0.6)


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

if RUN_TAGS is not None:
    tags = [t for t in tags if t in RUN_TAGS]

print("\nUsing model tags:")
for tag in tags:
    print(tag, T_by_tag[tag], a_by_tag[tag])

if not tags:
    raise RuntimeError("No valid model tags selected. Check RUN_TAGS or clean outputs/models.")



#u_mid = (np.array(exp_params.minvals) + np.array(exp_params.maxvals)) / 2.0
# ============================================================
# Fixed physical validation case
# ============================================================

ramp1_C_min = 2.2
hold1_T_C = 110.0
hold1_d_min = 58.0

ramp2_C_min = 2.2
hold2_T_C = 180.0
hold2_d_min = 105.0

htc_bot_W_m2K = 90.0
htc_top_W_m2K = 75.0
tool_len_m = 0.025


# Convert physical parameters to the exact scaled format
# used during XPIDON training.
u_case = np.array([
    # ramp 1
    (ramp1_C_min / 60.0)
    * (exp_params.T_scaler * exp_params.t_max),

    # first hold temperature
    hold1_T_C * exp_params.T_scaler,

    # first hold duration
    hold1_d_min * 60.0 / exp_params.t_max,

    # ramp 2
    (ramp2_C_min / 60.0)
    * (exp_params.T_scaler * exp_params.t_max),

    # second hold temperature
    hold2_T_C * exp_params.T_scaler,

    # second hold duration
    hold2_d_min * 60.0 / exp_params.t_max,

    # bottom HTC
    htc_bot_W_m2K * exp_params.h_scaler,

    # top HTC
    htc_top_W_m2K * exp_params.h_scaler,

    # tool thickness
    tool_len_m * exp_params.len_scaler,
], dtype=np.float32)


print("\nValidation case in physical units:")
print("ramp1 =", ramp1_C_min, "C/min")
print("hold1 =", hold1_T_C, "C,", hold1_d_min, "min")
print("ramp2 =", ramp2_C_min, "C/min")
print("hold2 =", hold2_T_C, "C,", hold2_d_min, "min")
print("hbot   =", htc_bot_W_m2K, "W/m2K")
print("htop   =", htc_top_W_m2K, "W/m2K")
print("Lt     =", tool_len_m, "m")

print("\nScaled XPIDON input:")
print(u_case)





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
    #U_star = np.tile(u_mid, (nt, 1))
    U_star = np.tile(u_case, (nt, 1))
    T_pred_scaled = model.pred_T(
        T_params,
        jnp.array(U_star),
        jnp.array(Y_star)
    )

    a_pred = model.pred_a(
        a_params,
        jnp.array(U_star),
        jnp.array(Y_star)
    )

    T_pred_C = (
        np.asarray(T_pred_scaled)
        / exp_params.T_scaler
    )

    all_time_physical.append(t_physical)
    all_T.append(T_pred_C)
    all_a.append(np.asarray(a_pred))

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

# ============================================================
# Generate the same autoclave air-temperature curve
# ============================================================

Ta_scaled = Temp_air(
    exp_params.T_ini,   # already scaled inside ExpParams
    u_case[0],          # scaled ramp1
    u_case[1],          # scaled hold1 temperature
    u_case[2],          # scaled hold1 duration
    u_case[3],          # scaled ramp2
    u_case[4],          # scaled hold2 temperature
    u_case[5],          # scaled hold2 duration
    1.0,                # normalized total process time
).two_hold(
    all_time_physical / exp_params.t_max
)

Ta_C = (
    np.asarray(Ta_scaled)
    / exp_params.T_scaler
)

exotherm_C = all_T - Ta_C

peak_index = int(np.argmax(exotherm_C))

print("\nExotherm diagnostic:")
print(
    "Maximum T_part - T_air =",
    float(exotherm_C[peak_index]),
    "C"
)
print(
    "Peak time =",
    float(all_time_physical[peak_index] / 60.0),
    "min"
)
print(
    "Part temperature at peak =",
    float(all_T[peak_index]),
    "C"
)
print(
    "Air temperature at peak =",
    float(Ta_C[peak_index]),
    "C"
)





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
ax1.plot(
    all_time_physical / 60.0,
    Ta_C,
    linestyle=":",
    linewidth=2,
    label="Autoclave air temperature"
)
# Mark temporal subdomain boundaries
for tag in tags[:-1]:
    boundary_min = tag_to_fraction(tag) * exp_params.t_max / 60.0
    ax1.axvline(boundary_min, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax1.text(
        boundary_min,
        ax1.get_ylim()[1] * 0.95,
        tag,
        rotation=90,
        va="top",
        ha="right",
        fontsize=7,
        color="gray"
    )


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