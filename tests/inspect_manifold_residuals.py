import os
import sys
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Let's inspect the entire Manifold 1 (87Rb F=2) and Manifold 2 (85Rb F=3) in Prot 1
freq, _, p, u = main.prep_data('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', align_shift=True)
diff = p - u
fsr = 374.74

fits = main.fit_hyperfine_multiplets(freq, diff, windows=[(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)], fsr=fsr)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.28, 'wspace': 0.22})

for idx, m_idx in enumerate([0, 1]): # Manifold 1 & Manifold 2
    f = fits[m_idx]
    x_data = f['x_data']
    y_data = f['y_data']
    x_fit = f['x_fit']
    y_fit = f['y_fit']
    
    # Evaluate fit at x_data
    # Re-evaluate
    ax_top = axes[0, idx]
    ax_res = axes[1, idx]
    
    # We can evaluate model at x_data:
    # From fit dict:
    bg = f['bg']
    amps = f['amplitudes']
    gamma = f['gamma']
    peak_freqs = f['peak_freqs']
    
    y_model = np.full_like(x_data, bg)
    for k in range(len(amps)):
        y_model += amps[k] / (1.0 + 4.0 * ((x_data - peak_freqs[k]) / gamma)**2)
        
    res = y_data - y_model # Data minus fit
    
    ax_top.plot(x_data, y_data, 'k.', markersize=2.5, alpha=0.4, label='Data')
    ax_top.plot(x_data, y_model, 'r-', lw=1.8, label=f"Lorentzian Fit (\\Gamma={gamma:.1f} MHz)")
    ax_top.axhline(bg, color='gray', linestyle=':', label='Fitted Baseline')
    ax_top.set_title(f"{f['name']}: Data vs. Fit", fontweight='bold')
    ax_top.set_ylabel('Diff Absorption (V)')
    ax_top.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax_top.grid(True, linestyle=':', alpha=0.5)
    
    # Mark peaks
    for pk in peak_freqs:
        ax_top.axvline(pk, color='navy', linestyle=':', alpha=0.4)
        ax_res.axvline(pk, color='navy', linestyle=':', alpha=0.4)
        
    ax_res.plot(x_data, res, 'b-', lw=1.2, label='Data - Fit')
    ax_res.axhline(0, color='red', linestyle='--', lw=1.0)
    ax_res.set_ylabel('Residual (V)')
    ax_res.set_xlabel('Frequency (MHz)')
    ax_res.set_title(f"{f['name']}: Residuals (Data - Fit)", fontweight='bold')
    ax_res.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax_res.grid(True, linestyle=':', alpha=0.5)

os.makedirs('figures/lineshape', exist_ok=True)
out_path = 'figures/lineshape/manifold_residuals_detailed.png'
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)
