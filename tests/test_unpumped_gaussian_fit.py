import os
import sys
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

# Configure style
plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11

# Load Protocol 1
fp_file = 'data/Session 9/F1.CSV'
p_file = 'data/Session 9/P1.CSV'
u_file = 'data/Session 9/U1.CSV'

freq, fabry, pumped, unpumped = main.prep_data(fp_file, p_file, u_file, align_shift=False)
fsr = 374.74 # MHz per fringe
nu = freq * fsr # in MHz

# Cut extreme edges where polyfit/laser ramp behaves irregularly
valid_mask = (freq >= 0.5) & (freq <= 26.5)
nu_v = nu[valid_mask]
freq_v = freq[valid_mask]
u_v = unpumped[valid_mask]

# Downsample slightly for fast, robust optimization (take every 5th point: ~26,000 points)
step = 5
nu_fit = nu_v[::step]
freq_fit = freq_v[::step]
u_fit = u_v[::step]

print(f"Points used for fit: {len(u_fit)}")

# -------------------------------------------------------------------------
# Global Model 1: 4 Gaussian Dips on a Cubic Polynomial Baseline
# y(nu) = (c0 + c1*nu + c2*nu^2 + c3*nu^3) - sum_{k=1}^4 A_k * exp(- (nu - nu_k)^2 / (2*sigma_k^2))
# -------------------------------------------------------------------------
# Initial guesses:
# Background points outside dips
m_bg = ((freq_fit >= 0.5) & (freq_fit <= 2.5)) | \
       ((freq_fit >= 10.0) & (freq_fit <= 12.0)) | \
       ((freq_fit >= 18.0) & (freq_fit <= 19.5)) | \
       ((freq_fit >= 24.5) & (freq_fit <= 26.5))

p_bg_init = np.polyfit(nu_fit[m_bg] * 1e-3, u_fit[m_bg], 3) # scaled in GHz for numerical stability

# 4 Dip centers (in MHz):
# 1: 87Rb F=2  ~ 1840 MHz
# 2: 85Rb F=3  ~ 2970 MHz
# 3: 85Rb F=2  ~ 5850 MHz
# 4: 87Rb F=1  ~ 8310 MHz
dip_centers_init = [1840.0, 2970.0, 5850.0, 8310.0]
# Initial depth A_k = I_bg(nu_k) - u_min(nu_k)
bg_at_dips = np.polyval(p_bg_init, np.array(dip_centers_init) * 1e-3)
u_min_dips = [-1.17, -2.84, -1.42, -0.31]
amps_init = bg_at_dips - u_min_dips

# Room temp Doppler sigma ~ 220 MHz (FWHM ~ 520 MHz)
# Due to unresolved excited hyperfine structure, apparent sigma ~ 260-350 MHz
sigma_init = [320.0, 300.0, 350.0, 320.0]

# Parameter vector: [c3, c2, c1, c0] (4 poly) + [A1, nu1, sig1, A2, nu2, sig2, A3, nu3, sig3, A4, nu4, sig4] (12 dip params) = 16 params
p0_global = list(p_bg_init)
for k in range(4):
    p0_global.extend([amps_init[k], dip_centers_init[k], sigma_init[k]])

def model_global_4gauss(p, x_mhz):
    c3, c2, c1, c0 = p[:4]
    x_ghz = x_mhz * 1e-3
    y_eval = c3 * x_ghz**3 + c2 * x_ghz**2 + c1 * x_ghz + c0
    for k in range(4):
        Ak = p[4 + k*3]
        nuk = p[4 + k*3 + 1]
        sigk = p[4 + k*3 + 2]
        y_eval -= Ak * np.exp(-0.5 * ((x_mhz - nuk) / sigk)**2)
    return y_eval

def res_global(p):
    return model_global_4gauss(p, nu_fit) - u_fit

b_low = [-1.0, -1.0, -1.0, -1.0]
b_high = [1.0, 1.0, 1.0, 2.0]
for k in range(4):
    b_low.extend([0.05, dip_centers_init[k] - 300.0, 100.0])
    b_high.extend([5.0, dip_centers_init[k] + 300.0, 600.0])

print("Running Global 4-Gaussian Fit...")
fit_glob = least_squares(res_global, p0_global, bounds=(b_low, b_high), ftol=1e-7, xtol=1e-7)

p_fit = fit_glob.x
p_poly_fit = p_fit[:4]
print("\n=== Global 4-Gaussian Fit Results ===")
print(f"Background baseline (GHz poly): c3={p_poly_fit[0]:.4f}, c2={p_poly_fit[1]:.4f}, c1={p_poly_fit[2]:.4f}, c0={p_poly_fit[3]:.4f}")
dip_names = ['87Rb F=2', '85Rb F=3', '85Rb F=2', '87Rb F=1']
for k in range(4):
    Ak = p_fit[4 + k*3]
    nuk = p_fit[4 + k*3 + 1]
    sigk = p_fit[4 + k*3 + 2]
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigk
    print(f"  Dip {k+1} ({dip_names[k]:10s}): Center = {nuk:6.1f} MHz (Fringe {nuk/fsr:5.2f}), Amp = {Ak:5.3f} V, Sigma = {sigk:5.1f} MHz, FWHM = {fwhm:5.1f} MHz")

# Evaluate on dense grid
y_glob_fit = model_global_4gauss(p_fit, nu_v)
res_glob = u_v - y_glob_fit
print(f"Global Fit RMS Error: {np.std(res_glob)*1e3:.2f} mV")

# -------------------------------------------------------------------------
# Local Fits per Dip (individual Gaussian + local cubic/quadratic baseline)
# -------------------------------------------------------------------------
local_windows = [
    (1100.0, 2550.0), # 87Rb F=2
    (2200.0, 3800.0), # 85Rb F=3
    (5000.0, 6700.0), # 85Rb F=2
    (7400.0, 9200.0), # 87Rb F=1
]

local_results = []
for k in range(4):
    w0, w1 = local_windows[k]
    m_loc = (nu_v >= w0) & (nu_v <= w1)
    x_loc = nu_v[m_loc]
    y_loc = u_v[m_loc]
    
    # Model: (b0 + b1*x + b2*x^2) - A * exp(-(x - x0)^2 / (2*sig^2))
    x_mid = 0.5 * (w0 + w1)
    x_rel = x_loc - x_mid
    
    def res_loc(p):
        b0, b1, b2, A, x0_rel, sig = p
        y_mod = (b0 + b1*(x_rel*1e-3) + b2*(x_rel*1e-3)**2) - A * np.exp(-0.5 * ((x_rel - x0_rel) / sig)**2)
        return y_mod - y_loc
        
    p0_loc = [np.max(y_loc), 0.0, 0.0, np.max(y_loc)-np.min(y_loc), 0.0, 300.0]
    b_low_loc = [-2.0, -5.0, -5.0, 0.05, -300.0, 100.0]
    b_high_loc = [2.0, 5.0, 5.0, 5.0, 300.0, 600.0]
    
    fit_loc = least_squares(res_loc, p0_loc, bounds=(b_low_loc, b_high_loc), ftol=1e-7, xtol=1e-7)
    
    b0, b1, b2, A_l, x0_rel_l, sig_l = fit_loc.x
    x0_l = x_mid + x0_rel_l
    fwhm_l = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig_l
    
    y_eval_loc = (b0 + b1*(x_rel*1e-3) + b2*(x_rel*1e-3)**2) - A_l * np.exp(-0.5 * ((x_rel - x0_rel_l) / sig_l)**2)
    bg_eval_loc = (b0 + b1*(x_rel*1e-3) + b2*(x_rel*1e-3)**2)
    res_l = y_loc - y_eval_loc
    
    local_results.append({
        'name': dip_names[k], 'x': x_loc, 'y': y_loc, 'y_fit': y_eval_loc,
        'bg_fit': bg_eval_loc, 'res': res_l, 'center': x0_l, 'amp': A_l,
        'sigma': sig_l, 'fwhm': fwhm_l, 'rms': np.std(res_l)
    })
    print(f"Local Dip {k+1} ({dip_names[k]:10s}): Center = {x0_l:6.1f} MHz, Amp = {A_l:5.3f} V, Sigma = {sig_l:5.1f} MHz, FWHM = {fwhm_l:5.1f} MHz, RMS = {np.std(res_l)*1e3:.2f} mV")

# -------------------------------------------------------------------------
# Plotting Figure 1: Full Sweep Unpumped Transmission & Global Gaussian Fit
# -------------------------------------------------------------------------
os.makedirs('figures/unpumped_fits', exist_ok=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.14})

# Evaluate background curve
x_ghz = nu_v * 1e-3
y_bg_curve = p_poly_fit[0] * x_ghz**3 + p_poly_fit[1] * x_ghz**2 + p_poly_fit[2] * x_ghz + p_poly_fit[3]

ax1.plot(nu_v * 1e-3, u_v, 'k.', markersize=1.0, alpha=0.35, label='Experimental Unpumped Transmission')
ax1.plot(nu_v * 1e-3, y_glob_fit, 'r-', lw=1.8, label='Global 4-Gaussian Fit')
ax1.plot(nu_v * 1e-3, y_bg_curve, 'b--', lw=1.5, alpha=0.8, label='Fitted Laser Power Baseline (Cubic)')

# Annotate the 4 dips
for k in range(4):
    nuk = p_fit[4 + k*3]
    Ak = p_fit[4 + k*3]
    sigk = p_fit[4 + k*3 + 2]
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigk
    ax1.axvline(nuk * 1e-3, color='gray', linestyle=':', alpha=0.6)
    ax1.text(nuk * 1e-3, np.min(u_v)*0.95, f"{dip_names[k]}\n$\\nu_0={nuk:.0f}$ MHz\nFWHM$={fwhm:.0f}$ MHz",
             ha='center', va='bottom', fontsize=8.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray'))

ax1.set_ylabel('Photodiode Signal (V)', fontsize=11)
ax1.set_title('Protocol 1: Full-Scan Unpumped Doppler Spectrum & Global 4-Gaussian Fit', fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', framealpha=0.9, fontsize=9.5)
ax1.grid(True, linestyle=':', alpha=0.55)

# Residuals on bottom
ax2.plot(nu_v * 1e-3, res_glob, color='crimson', lw=0.8, alpha=0.75, label=f'Residuals (RMS: {np.std(res_glob)*1e3:.1f} mV)')
ax2.axhline(0, color='black', linestyle='--', lw=0.8)
ax2.set_xlabel('Laser Frequency (GHz, calibrated from Fabry-Pérot fringes)', fontsize=11)
ax2.set_ylabel('Residual (V)', fontsize=11)
ax2.set_title('Global Fit Residuals ($y_{data} - y_{fit}$)', fontsize=10.5, fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
ax2.grid(True, linestyle=':', alpha=0.55)

fig1_path = 'figures/unpumped_fits/prot1_unpumped_global_gaussian_fit.png'
plt.tight_layout()
plt.savefig(fig1_path, dpi=200)
plt.close()
print(f"Saved: {fig1_path}")

# -------------------------------------------------------------------------
# Plotting Figure 2: Detailed 4-Manifold Subplots with Residuals
# -------------------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(14, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.38, 'wspace': 0.22})
fig.suptitle("Protocol 1: Individual Doppler Absorption Dips with Local Gaussian Fits & Residuals", fontsize=12, fontweight='bold')

for k in range(4):
    r = local_results[k]
    ax_fit, ax_res = axes[k, 0], axes[k, 1]
    
    # Plot on left
    ax_fit.plot(r['x'], r['y'], 'k.', markersize=2.0, alpha=0.4, label='Data')
    ax_fit.plot(r['x'], r['y_fit'], 'r-', lw=1.8, label=f"Gaussian Fit (FWHM={r['fwhm']:.0f} MHz)")
    ax_fit.plot(r['x'], r['bg_fit'], 'b--', lw=1.2, alpha=0.7, label='Local Baseline')
    ax_fit.axvline(r['center'], color='gray', linestyle=':', alpha=0.6)
    
    ax_fit.set_title(f"Dip {k+1}: {r['name']} | Center = {r['center']:.1f} MHz, $\\sigma={r['sigma']:.1f}$ MHz",
                     fontweight='bold', fontsize=10, loc='left')
    ax_fit.set_ylabel('Photodiode Signal (V)', fontsize=9.5)
    ax_fit.legend(loc='lower left', framealpha=0.9, fontsize=8)
    ax_fit.grid(True, linestyle=':', alpha=0.5)
    
    # Residuals on right
    ax_res.plot(r['x'], r['res'], color='crimson', lw=0.9, alpha=0.8, label=f"Residual (RMS: {r['rms']*1e3:.1f} mV)")
    ax_res.axhline(0, color='black', linestyle='--', lw=0.8)
    ax_res.set_title(f"Residuals ({r['name']})", fontweight='bold', fontsize=10, loc='left')
    ax_res.set_ylabel('Residual (V)', fontsize=9.5)
    ax_res.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax_res.grid(True, linestyle=':', alpha=0.5)
    
    if k == 3:
        ax_fit.set_xlabel('Frequency (MHz)', fontsize=10)
        ax_res.set_xlabel('Frequency (MHz)', fontsize=10)

fig2_path = 'figures/unpumped_fits/prot1_unpumped_local_dips_residuals.png'
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(fig2_path, dpi=200)
plt.close()
print(f"Saved: {fig2_path}")
