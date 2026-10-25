import os
import sys
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Lineshape models
def lorentzian(x, amp, gamma, x0, bg):
    return bg + amp / (1.0 + 4.0 * ((x - x0) / gamma)**2)

def saturated_lamb_dip(x, amp, s, gamma, x0, bg):
    denom = 1.0 + 4.0 * ((x - x0) / gamma)**2
    val = 1.0 - 1.0 / np.sqrt(1.0 + s / denom)
    val_max = 1.0 - 1.0 / np.sqrt(1.0 + s)
    return bg + amp * (val / val_max)

def lorentzian_plus_pedestal(x, amp_L, gamma_L, amp_ped, sigma_ped, x0, bg):
    L = amp_L / (1.0 + 4.0 * ((x - x0) / gamma_L)**2)
    ped = amp_ped * np.exp(-0.5 * ((x - x0) / sigma_ped)**2)
    return bg + L + ped

protocols = [1, 3, 5, 7]
prot_names = {
    1: "Prot 1 (High Power)",
    3: "Prot 3 (Med-High)",
    5: "Prot 5 (Med-Low)",
    7: "Prot 7 (Low Power)"
}

fsr = 374.74
fig, axes = plt.subplots(4, 2, figsize=(14, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.35, 'wspace': 0.22})
fig.suptitle("Clean Peak Lineshape Comparison Across Protocols ($^{87}\\mathrm{Rb } \\mathrm{CO}(2,3)$)\n"
             "Standard Lorentzian vs. Saturated Lamb-Dip vs. Lorentzian + Doppler Pedestal",
             fontsize=12, fontweight='bold')

os.makedirs('figures/lineshape_protocols', exist_ok=True)
results_table = []

for idx, p_num in enumerate(protocols):
    fp = f"data/Session 9/F{p_num}.CSV"
    p = f"data/Session 9/P{p_num}.CSV"
    u = f"data/Session 9/U{p_num}.CSV"
    freq, _, pumped, unpumped = main.prep_data(fp, p, u, align_shift=True)
    diff = pumped - unpumped
    
    # Locate peak near 87Rb F=2 dominant peak (find global max of diff in the first manifold window)
    # Window for M1 in Prot p_num:
    from tests.compare_protocols_models import WINDOWS_ALL
    w0, w1 = WINDOWS_ALL[p_num][0]
    m_win = (freq >= w0) & (freq <= w1)
    x_win = freq[m_win] * fsr
    y_win = diff[m_win]
    
    # Peak is the highest point in window
    pk_idx = np.argmax(y_win)
    pk_x = x_win[pk_idx]
    
    # Slice +/- 35 MHz around the peak
    m_pk = (x_win >= pk_x - 35.0) & (x_win <= pk_x + 35.0)
    x = x_win[m_pk] - pk_x
    y = y_win[m_pk]
    
    # Fits
    # 1. Lorentzian
    p0_L = [np.max(y)-np.min(y), 24.0, 0.0, np.min(y)]
    fit_L, _ = curve_fit(lorentzian, x, y, p0=p0_L)
    y_L = lorentzian(x, *fit_L)
    ssr_L = np.sum((y - y_L)**2)
    
    # 2. Saturated Lamb-Dip
    p0_S = [np.max(y)-np.min(y), 2.0, 20.0, 0.0, np.min(y)]
    fit_S, _ = curve_fit(saturated_lamb_dip, x, y, p0=p0_S, bounds=([0, 0.01, 5, -5, -1], [5, 50, 50, 5, 1]))
    y_S = saturated_lamb_dip(x, *fit_S)
    ssr_S = np.sum((y - y_S)**2)
    
    # 3. Lorentzian + Pedestal
    p0_P = [np.max(y)-np.min(y)*0.8, 20.0, 0.05, 40.0, 0.0, np.min(y)]
    fit_P, _ = curve_fit(lorentzian_plus_pedestal, x, y, p0=p0_P, bounds=([0, 5, 0, 10, -5, -1], [5, 40, 1, 200, 5, 1]))
    y_P = lorentzian_plus_pedestal(x, *fit_P)
    ssr_P = np.sum((y - y_P)**2)
    
    results_table.append({
        'prot': p_num, 'name': prot_names[p_num],
        'L_gamma': fit_L[1], 'L_ssr': ssr_L,
        'S_gamma': fit_S[2], 'S_s': fit_S[1], 'S_ssr': ssr_S,
        'P_gamma': fit_P[1], 'P_ped_amp': fit_P[2], 'P_ped_sigma': fit_P[3], 'P_ssr': ssr_P
    })
    
    # Plot
    ax_fit, ax_res = axes[idx, 0], axes[idx, 1]
    ax_fit.plot(x, y, 'k.', markersize=3.0, alpha=0.45, label='Data')
    ax_fit.plot(x, y_L, '--', color='#1f77b4', lw=1.5, label=f"Lorentzian ($\\Gamma={fit_L[1]:.1f}\\text{{ MHz}}$, SSR={ssr_L:.4f})")
    ax_fit.plot(x, y_S, '-.', color='#d62728', lw=1.6, label=f"Saturated ($\\Gamma={fit_S[2]:.1f}, s={fit_S[1]:.1f}$, SSR={ssr_S:.4f})")
    ax_fit.plot(x, y_P, '-', color='#2ca02c', lw=1.8, label=f"Lorentzian+Pedestal ($\\Gamma={fit_P[1]:.1f}\\text{{ MHz}}$, SSR={ssr_P:.4f})")
    
    ax_fit.set_title(f"{prot_names[p_num]}: Peak Fit Comparison", fontweight='bold', fontsize=10, loc='left')
    ax_fit.set_ylabel("Diff Abs (V)")
    ax_fit.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax_fit.grid(True, linestyle=':', alpha=0.55)
    
    res_L = y - y_L
    res_S = y - y_S
    res_P = y - y_P
    ax_res.plot(x, res_L, color='#1f77b4', lw=1.0, alpha=0.7, label=f"Lorentzian")
    ax_res.plot(x, res_S, color='#d62728', lw=1.0, linestyle='-.', alpha=0.7, label=f"Saturated")
    ax_res.plot(x, res_P, color='#2ca02c', lw=1.1, alpha=0.8, label=f"Lor + Pedestal")
    ax_res.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax_res.set_ylabel("Residual (V)")
    ax_res.set_title(f"Residuals ({prot_names[p_num]})", fontsize=10, loc='left')
    ax_res.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax_res.grid(True, linestyle=':', alpha=0.55)
    
    if idx == 3:
        ax_fit.set_xlabel("Detuning from Peak (MHz)")
        ax_res.set_xlabel("Detuning from Peak (MHz)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
lineshape_fig_path = "figures/lineshape_protocols/lineshape_comparison_across_protocols.png"
plt.savefig(lineshape_fig_path, dpi=200)
plt.close()
print("Saved:", lineshape_fig_path)

print("\n=== ISOLATED PEAK LINESHAPE FITS ACROSS PROTOCOLS ===")
for r in results_table:
    print(f"{r['name']:25s}:")
    print(f"  Standard Lorentzian       : Gamma = {r['L_gamma']:5.2f} MHz | SSR = {r['L_ssr']:.5f}")
    print(f"  Saturated Lamb-Dip        : Gamma = {r['S_gamma']:5.2f} MHz, s = {r['S_s']:5.2f} | SSR = {r['S_ssr']:.5f} (Drop: {(r['L_ssr']-r['S_ssr'])/r['L_ssr']*100:.1f}%)")
    print(f"  Lorentzian + Pedestal     : Gamma_L = {r['P_gamma']:5.2f} MHz, Ped_Amp = {r['P_ped_amp']:.4f}, Sigma = {r['P_ped_sigma']:4.1f} MHz | SSR = {r['P_ssr']:.5f} (Drop: {(r['L_ssr']-r['P_ssr'])/r['L_ssr']*100:.1f}%)")
