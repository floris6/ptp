import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA  (Table 1)
# ─────────────────────────────────────────────────────────────────────────────
raw = {
    "Water":          {"200": [22.91, 22.80, 22.60], "150": [15.96, 16.39, 16.26]},
    "Oil":            {"200": [82.19, 81.81, 80.98], "150": [62.25, 62.27, 62.46]},
    "Soap 50:50":     {"200": [98.93,100.01, 98.32], "150": [71.74, 72.48, 71.35]},
    "Soap 75:25":     {"200": [263.55,262.72,262.11], "150": [197.66,198.37,198.82]},
}

# Densities (kg/L) — Table 2
densities = {
    "Water":      0.976,
    "Oil":        0.894,
    "Soap 50:50": 1.001,
    "Soap 75:25": 1.035,
}

# Literature / given viscosities (mPa·s) — Table 2
viscosity_given = {
    "Water":      1.00,
    "Oil":        3.28,
    "Soap 50:50": 4.46,
    "Soap 75:25": 12.23,
}

compounds   = list(raw.keys())
heights     = [200, 150]          # mm
n_trials    = 3                   # replicates
sigma_h_mm  = 0.5                 # ruler reading uncertainty (mm)
sigma_mass  = 0.05                # balance uncertainty (g)
sigma_vol   = 0.5                 # graduated-cylinder uncertainty (mL)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Derived statistics for drainage time
# ─────────────────────────────────────────────────────────────────────────────
t_mean, t_std, t_sem = {}, {}, {}   # keys: (compound, height_str)

for cmp in compounds:
    for h_str in ["200", "150"]:
        vals = np.array(raw[cmp][h_str])
        t_mean[(cmp, h_str)] = vals.mean()
        # sample std dev
        t_std[(cmp, h_str)]  = vals.std(ddof=1)
        # SEM (used as uncertainty on the mean)
        t_sem[(cmp, h_str)]  = t_std[(cmp, h_str)] / np.sqrt(n_trials)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Relative viscosity & its propagated uncertainty
#
#   η_rel = (ρ_f  × t_f) / (ρ_w × t_w)          (from Poiseuille + falling head)
#
#   σ_η / η = √[ (σ_ρ_f/ρ_f)² + (σ_t_f/t_f)² + (σ_ρ_w/ρ_w)² + (σ_t_w/t_w)² ]
#
#   σ_ρ / ρ = √[ (σ_m/m)² + (σ_V/V)² ]   (density from balance + cylinder)
# ─────────────────────────────────────────────────────────────────────────────
# Liquid masses used in Table 3
mass_liq = {"Water": 71.64, "Oil": 78.17, "Soap 50:50": 80.10, "Soap 75:25": 82.79}
vol_liq  = {"Water": 80.1,  "Oil": 80.1,  "Soap 50:50": 80.0,  "Soap 75:25": 80.0}

sigma_rho_rel = {}   # relative density uncertainty
for cmp in compounds:
    m, V = mass_liq[cmp], vol_liq[cmp]
    sigma_rho_rel[cmp] = np.sqrt((sigma_mass/m)**2 + (sigma_vol/V)**2)

# Combine the two heights for averaged viscosity uncertainty
visc_calc   = {}   # calculated relative viscosity (per height, then averaged)
visc_err    = {}   # propagated absolute uncertainty

rho_w = densities["Water"]
sig_rho_w_rel = sigma_rho_rel["Water"]

visc_per_height = {cmp: [] for cmp in compounds}
visc_err_per_height = {cmp: [] for cmp in compounds}

for cmp in compounds:
    rho_f = densities[cmp]
    sig_rho_f_rel = sigma_rho_rel[cmp]

    for h_str in ["200", "150"]:
        t_f = t_mean[(cmp, h_str)]
        t_w = t_mean[("Water", h_str)]
        sig_t_f_rel = t_sem[(cmp, h_str)]  / t_f
        sig_t_w_rel = t_sem[("Water", h_str)] / t_w

        eta_rel = (rho_f * t_f) / (rho_w * t_w)

        sig_eta_rel = eta_rel * np.sqrt(
            sig_rho_f_rel**2 + sig_t_f_rel**2 +
            sig_rho_w_rel**2 + sig_t_w_rel**2
        )

        visc_per_height[cmp].append(eta_rel)
        visc_err_per_height[cmp].append(sig_eta_rel)

    # Weighted mean of the two height results
    w = 1.0 / np.array(visc_err_per_height[cmp])**2
    visc_calc[cmp] = np.sum(w * visc_per_height[cmp]) / np.sum(w)
    visc_err[cmp]  = 1.0 / np.sqrt(np.sum(w))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Flow rate proxy  Q = h / t   (mm/s, proportional to true Q)
#
#   σ_Q / Q = √[ (σ_h/h)² + (σ_t/t)² ]
# ─────────────────────────────────────────────────────────────────────────────
Q_vals, Q_errs = [], []   # one point per (compound, height)
eta_for_Q, eta_err_for_Q = [], []

for cmp in compounds:
    for i, h_str in enumerate(["200", "150"]):
        h = float(h_str)
        t = t_mean[(cmp, h_str)]
        sig_t = t_sem[(cmp, h_str)]

        Q = h / t
        sig_Q = Q * np.sqrt((sigma_h_mm/h)**2 + (sig_t/t)**2)

        Q_vals.append(Q)
        Q_errs.append(sig_Q)
        eta_for_Q.append(visc_calc[cmp])
        eta_err_for_Q.append(visc_err[cmp])

Q_vals    = np.array(Q_vals)
Q_errs    = np.array(Q_errs)
eta_for_Q = np.array(eta_for_Q)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Fit Poiseuille model:  Q = C / η
# ─────────────────────────────────────────────────────────────────────────────
def poiseuille(eta, C):
    return C / eta

popt, pcov = curve_fit(poiseuille, eta_for_Q, Q_vals, sigma=Q_errs,
                       absolute_sigma=True, p0=[10.0])
C_fit   = popt[0]
C_err   = np.sqrt(pcov[0, 0])

eta_fit = np.linspace(0.8, 14, 400)
Q_fit   = poiseuille(eta_fit, C_fit)

# R² on fitted data
Q_pred  = poiseuille(eta_for_Q, C_fit)
SS_res  = np.sum((Q_vals - Q_pred)**2)
SS_tot  = np.sum((Q_vals - Q_vals.mean())**2)
R2      = 1 - SS_res / SS_tot

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS & LABELS
# ─────────────────────────────────────────────────────────────────────────────
colors  = {"Water": "#2196F3", "Oil": "#FF9800",
           "Soap 50:50": "#4CAF50", "Soap 75:25": "#9C27B0"}
markers = {"Water": "o", "Oil": "s", "Soap 50:50": "^", "Soap 75:25": "D"}
labels_short = {"Water": "Water", "Oil": "Oil",
                "Soap 50:50": "Soap 50:50", "Soap 75:25": "Soap 75:25"}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Average drainage time with SEM error bars (grouped bar chart)
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

x      = np.arange(len(compounds))
width  = 0.35
bar_colors_200 = [colors[c] for c in compounds]
bar_colors_150 = [colors[c] for c in compounds]
visc_given = viscosity_given   # alias

means_200 = [t_mean[(c, "200")] for c in compounds]
errs_200  = [t_sem [(c, "200")] for c in compounds]
means_150 = [t_mean[(c, "150")] for c in compounds]
errs_150  = [t_sem [(c, "150")] for c in compounds]

bars1 = ax1.bar(x - width/2, means_200, width, yerr=errs_200,
                color=[colors[c] for c in compounds], alpha=0.85,
                capsize=5, label="h₀ = 200 mm", edgecolor="black", linewidth=0.7)
bars2 = ax1.bar(x + width/2, means_150, width, yerr=errs_150,
                color=[colors[c] for c in compounds], alpha=0.45,
                capsize=5, label="h₀ = 150 mm", edgecolor="black", linewidth=0.7,
                hatch="//")

ax1.set_xticks(x)
ax1.set_xticklabels(compounds, fontsize=11)
ax1.set_ylabel("Average drainage time (s)", fontsize=12)
ax1.set_xlabel("Fluid", fontsize=12)
ax1.set_title("Figure 1 — Average drainage time per fluid at two initial heights\n"
              "(error bars = SEM, n = 3)", fontsize=11, pad=10)
ax1.legend(fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_ylim(0, max(means_200) * 1.18)

# annotate viscosity on top of 200 mm bars
for i, cmp in enumerate(compounds):
    ax1.text(x[i] - width/2, means_200[i] + errs_200[i] + 3,
             f"η = {visc_given[cmp]:.2f} mPa·s",
             ha="center", va="bottom", fontsize=7.5, color="black")

fig1.tight_layout()
fig1.savefig("/mnt/user-data/outputs/fig1_drainage_time.png", dpi=150, bbox_inches="tight")
print("Figure 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Flow rate proxy vs viscosity  (Poiseuille fit)
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax_idx, ax in enumerate(axes):
    log_scale = (ax_idx == 1)

    # Fit line
    ax.plot(eta_fit, Q_fit, "k--", lw=1.5, zorder=1,
            label=f"Poiseuille fit:  Q = {C_fit:.2f}/η\n$R^2$ = {R2:.4f}")

    # 95 % confidence band (propagate C uncertainty)
    Q_upper = poiseuille(eta_fit, C_fit + C_err)
    Q_lower = poiseuille(eta_fit, C_fit - C_err)
    ax.fill_between(eta_fit, Q_lower, Q_upper, color="grey", alpha=0.20,
                    label="95 % confidence band")

    # Data points — one colour per compound, two heights
    for cmp in compounds:
        for i, h_str in enumerate(["200", "150"]):
            h = float(h_str)
            t = t_mean[(cmp, h_str)]
            sig_t = t_sem[(cmp, h_str)]
            Q = h / t
            sig_Q = Q * np.sqrt((sigma_h_mm/h)**2 + (sig_t/t)**2)
            eta_c = visc_calc[cmp]
            sig_eta_c = visc_err[cmp]

            lbl = f"{cmp} (h={int(h)} mm)" if ax_idx == 0 else None
            ax.errorbar(eta_c, Q,
                        xerr=sig_eta_c, yerr=sig_Q,
                        fmt=markers[cmp], color=colors[cmp],
                        markersize=7 if i == 0 else 5,
                        alpha=1.0 if i == 0 else 0.55,
                        capsize=4, elinewidth=1.2, zorder=3,
                        label=lbl)

    ax.set_xlabel("Dynamic viscosity η (mPa·s)", fontsize=12)
    ax.set_ylabel("Flow-rate proxy  Q = h/t  (mm s⁻¹)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)

    if log_scale:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title("(b) Log–log scale  (linear ↔ power law)", fontsize=11)
        # Add slope = −1 guide
        ax.text(0.97, 0.97, "Slope = −1 expected\nfrom Poiseuille's law",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="dimgrey",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    else:
        ax.set_title("(a) Linear scale", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right")

fig2.suptitle(
    "Figure 2 — Flow-rate proxy Q = h/t versus fluid viscosity\n"
    "Error bars: propagated SEM (time) and density measurement uncertainty",
    fontsize=11, y=1.01
)
fig2.tight_layout()
fig2.savefig("/mnt/user-data/outputs/fig2_flowrate_vs_viscosity.png",
             dpi=150, bbox_inches="tight")
print("Figure 2 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Calculated vs given viscosity  (validation / parity plot)
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 5.5))

eta_given_arr = np.array([viscosity_given[c] for c in compounds])
eta_calc_arr  = np.array([visc_calc[c]        for c in compounds])
eta_calc_err  = np.array([visc_err[c]         for c in compounds])

# Parity line
lim = max(eta_given_arr.max(), eta_calc_arr.max()) * 1.1
ax3.plot([0, lim], [0, lim], "k--", lw=1.2, label="1 : 1 parity line", zorder=1)

for cmp in compounds:
    ax3.errorbar(viscosity_given[cmp], visc_calc[cmp],
                 yerr=visc_err[cmp],
                 fmt=markers[cmp], color=colors[cmp],
                 markersize=9, capsize=5, elinewidth=1.4, zorder=3,
                 label=cmp)

ax3.set_xlabel("Viscosity given in Table 2 (mPa·s)", fontsize=12)
ax3.set_title("Figure 3 — Calculated vs. reported viscosity\n"
              "(error bars = propagated uncertainty, n = 3 per height)", fontsize=11, pad=8)
ax3.legend(fontsize=10)
ax3.grid(linestyle="--", alpha=0.35)
ax3.set_xlim(0, lim); ax3.set_ylim(0, lim)
fig3.tight_layout()
fig3.savefig("/mnt/user-data/outputs/fig3_viscosity_parity.png",
             dpi=150, bbox_inches="tight")
print("Figure 3 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Summary of derived quantities ──")
print(f"{'Compound':<16} {'η_given':>10} {'η_calc':>10} {'σ_η':>8}  {'Q@200mm':>9} {'Q@150mm':>9}")
print("-" * 68)
for cmp in compounds:
    Q200 = float(200) / t_mean[(cmp, "200")]
    Q150 = float(150) / t_mean[(cmp, "150")]
    print(f"{cmp:<16} {viscosity_given[cmp]:>10.2f} {visc_calc[cmp]:>10.3f} "
          f"{visc_err[cmp]:>8.3f}  {Q200:>9.4f} {Q150:>9.4f}")

print(f"\nPoiseuille fit:  C = {C_fit:.3f} ± {C_err:.3f}   R² = {R2:.4f}")
