import os
import sys
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.special import voigt_profile

# Load Protocol 1
freq, _, p, u = main.prep_data('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', align_shift=True)
diff = p - u
fsr = 374.74

# Isolate the prominent, cleanest peak in 87Rb F=2: CO(2,3) around freq ~ 4.64
# Or let's look at the window (4.5, 4.8)
mask_pk = (freq >= 4.50) & (freq <= 4.75)
x = freq[mask_pk] * fsr
y = diff[mask_pk]
x = x - x[np.argmax(y)] # center peak at x = 0

print(f"Data range: {len(x)} points, peak height = {np.max(y):.4f}, min = {np.min(y):.4f}")

# Model 1: Pure Lorentzian
def lorentzian(x, amp, gamma, x0, bg):
    return bg + amp / (1.0 + 4.0 * ((x - x0) / gamma)**2)

# Model 2: Gaussian
def gaussian(x, amp, sigma, x0, bg):
    return bg + amp * np.exp(-0.5 * ((x - x0) / sigma)**2)

# Model 3: Pseudo-Voigt (linear combination of Lorentzian and Gaussian)
def pseudo_voigt(x, amp, fwhm, eta, x0, bg):
    # eta = 1 is pure Lorentzian, eta = 0 is pure Gaussian
    # For Gaussian with same FWHM: sigma = fwhm / (2 * sqrt(2 * ln 2))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    L = 1.0 / (1.0 + 4.0 * ((x - x0) / fwhm)**2)
    G = np.exp(-0.5 * ((x - x0) / sigma)**2)
    return bg + amp * (eta * L + (1.0 - eta) * G)

# Model 4: Lorentzian + Broad Doppler Pedestal (VCC / diffuse scattering)
def lorentzian_plus_pedestal(x, amp_L, gamma_L, amp_ped, sigma_ped, x0, bg):
    L = amp_L / (1.0 + 4.0 * ((x - x0) / gamma_L)**2)
    ped = amp_ped * np.exp(-0.5 * ((x - x0) / sigma_ped)**2)
    return bg + L + ped

# Model 5: Saturated Lamb-dip shape (1 - 1 / sqrt(1 + s / (1 + 4 x^2 / Gamma^2)))
def saturated_lamb_dip(x, amp, s, gamma, x0, bg):
    denom = 1.0 + 4.0 * ((x - x0) / gamma)**2
    # profile = (1 - 1 / sqrt(1 + s / denom)) normalized so peak at x0 is amp
    val = 1.0 - 1.0 / np.sqrt(1.0 + s / denom)
    val_max = 1.0 - 1.0 / np.sqrt(1.0 + s)
    return bg + amp * (val / val_max)

# Model 6: Power-law Lorentzian (exponent alpha: (1 + 4x^2/G^2)^(-alpha))
# alpha < 1 gives heavier wings and sharper cusped top!
def super_lorentzian(x, amp, gamma, alpha, x0, bg):
    return bg + amp / (1.0 + 4.0 * ((x - x0) / gamma)**2)**alpha

p0_L = [np.max(y)-np.min(y), 24.0, 0.0, np.min(y)]
fit_L, _ = curve_fit(lorentzian, x, y, p0=p0_L)

p0_G = [np.max(y)-np.min(y), 10.0, 0.0, np.min(y)]
fit_G, _ = curve_fit(gaussian, x, y, p0=p0_G)

p0_PV = [np.max(y)-np.min(y), 24.0, 0.5, 0.0, np.min(y)]
fit_PV, _ = curve_fit(pseudo_voigt, x, y, p0=p0_PV, bounds=([0, 5, 0, -5, -1], [5, 50, 1.0, 5, 1]))

p0_Ped = [np.max(y)-np.min(y)*0.8, 20.0, 0.1, 50.0, 0.0, np.min(y)]
fit_Ped, _ = curve_fit(lorentzian_plus_pedestal, x, y, p0=p0_Ped, bounds=([0, 5, 0, 15, -5, -1], [5, 40, 1, 200, 5, 1]))

p0_Sat = [np.max(y)-np.min(y), 1.0, 20.0, 0.0, np.min(y)]
fit_Sat, _ = curve_fit(saturated_lamb_dip, x, y, p0=p0_Sat, bounds=([0, 0.01, 5, -5, -1], [5, 20, 50, 5, 1]))

p0_SL = [np.max(y)-np.min(y), 24.0, 1.0, 0.0, np.min(y)]
fit_SL, _ = curve_fit(super_lorentzian, x, y, p0=p0_SL, bounds=([0, 5, 0.1, -5, -1], [5, 50, 3.0, 5, 1]))

y_L = lorentzian(x, *fit_L)
y_G = gaussian(x, *fit_G)
y_PV = pseudo_voigt(x, *fit_PV)
y_Ped = lorentzian_plus_pedestal(x, *fit_Ped)
y_Sat = saturated_lamb_dip(x, *fit_Sat)
y_SL = super_lorentzian(x, *fit_SL)

# Inspect residual signs for Pure Lorentzian (data - fit):
res_L = y - y_L
# Divide into 3 regions:
# Region 1: Peak top (|x| < 3 MHz)
top_mask = np.abs(x) < 3.0
# Region 2: Slopes (6 < |x| < 16 MHz)
slope_mask = (np.abs(x) >= 6.0) & (np.abs(x) <= 16.0)
# Region 3: Foot / Wings (|x| > 20 MHz)
foot_mask = np.abs(x) > 20.0

print("\n=== PURE LORENTZIAN RESIDUAL ANALYSIS (y_data - y_fit) ===")
print(f"Peak Top (|x| < 3 MHz)    : mean(y_data - y_fit) = {np.mean(res_L[top_mask]):+.5f} V -> Data is {'TALLER' if np.mean(res_L[top_mask])>0 else 'SHORTER'} than fit")
print(f"Slopes (6 < |x| < 16 MHz)  : mean(y_data - y_fit) = {np.mean(res_L[slope_mask]):+.5f} V -> Fit is {'ABOVE' if np.mean(res_L[slope_mask])<0 else 'BELOW'} data (overshooting slopes)")
print(f"Foot/Wings (|x| > 20 MHz)  : mean(y_data - y_fit) = {np.mean(res_L[foot_mask]):+.5f} V -> Data is {'HIGHER' if np.mean(res_L[foot_mask])>0 else 'LOWER'} than fit (heavy foot/wings)")

print("\n=== MODEL COMPARISON (Sum of Squared Residuals) ===")
print(f"1. Pure Lorentzian             : SSR = {np.sum(res_L**2):.5f}")
print(f"2. Pure Gaussian               : SSR = {np.sum((y - y_G)**2):.5f}")
print(f"3. Pseudo-Voigt (eta={fit_PV[2]:.2f})    : SSR = {np.sum((y - y_PV)**2):.5f}")
print(f"4. Saturated Lamb-Dip (s={fit_Sat[1]:.2f}): SSR = {np.sum((y - y_Sat)**2):.5f}")
print(f"5. Lorentzian + Doppler Pedestal: SSR = {np.sum((y - y_Ped)**2):.5f} (Gamma_L={fit_Ped[1]:.1f}, Sigma_ped={fit_Ped[3]:.1f})")
print(f"6. Generalized Power-Law (alpha={fit_SL[2]:.3f}): SSR = {np.sum((y - y_SL)**2):.5f}")

# Plot detailed visual comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2], 'hspace': 0.12})

ax1.plot(x, y, 'k.', markersize=3.5, alpha=0.5, label='Experimental Data (P - U)')
ax1.plot(x, y_L, 'b--', lw=1.8, label=f'Standard Lorentzian (SSR={np.sum(res_L**2):.4f})')
ax1.plot(x, y_Ped, 'g-', lw=2.0, label=f'Lorentzian + Doppler Pedestal (SSR={np.sum((y-y_Ped)**2):.4f})')
ax1.plot(x, y_SL, 'r-', lw=1.6, label=f'Cusped Profile (alpha={fit_SL[2]:.2f}, SSR={np.sum((y-y_SL)**2):.4f})')
ax1.set_ylabel('Differential Absorption (V)', fontsize=11)
ax1.set_title('Lineshape Discrepancy Analysis on Clean Saturated Absorption Peak ($^{87}\\mathrm{Rb } F=2 \\to \\mathrm{CO}(2,3)$)',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
ax1.grid(True, linestyle=':', alpha=0.6)

# Annotate the discrepancy points
ax1.annotate('Data top is SHARPER\n(Fit undershoots)', xy=(0, np.max(y)), xytext=(8, np.max(y)*0.95),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5), fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))
ax1.annotate('Fit is ABOVE data\non steep slopes', xy=(10, y_L[np.argmin(np.abs(x-10))]), xytext=(18, np.max(y)*0.65),
             arrowprops=dict(facecolor='blue', shrink=0.08, width=1, headwidth=5), fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
ax1.annotate('Data foot is HIGHER\n(Pedestal effect)', xy=(28, y[np.argmin(np.abs(x-28))]), xytext=(22, np.max(y)*0.30),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5), fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

# Plot Residuals (data - fit)
ax2.plot(x, res_L, 'b-', lw=1.5, label='Residual: Data - Standard Lorentzian')
ax2.plot(x, y - y_Ped, 'g-', lw=1.5, label='Residual: Data - (Lorentzian + Pedestal)')
ax2.plot(x, y - y_SL, 'r-', lw=1.2, linestyle=':', label='Residual: Data - Cusped Profile')
ax2.axhline(0, color='gray', linestyle='--', lw=0.8)
ax2.set_xlabel('Detuning from Peak Center (MHz)', fontsize=11)
ax2.set_ylabel('Residual (V)', fontsize=11)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.6)

os.makedirs('figures/lineshape', exist_ok=True)
out_fig = 'figures/lineshape/lineshape_discrepancy_inspection.png'
plt.tight_layout()
plt.savefig(out_fig, dpi=200)
plt.close()
print(f"Saved figure to: {out_fig}")
