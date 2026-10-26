import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'tests')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import main
from plot_multiplet_residuals import load_and_calibrate, fit_manifold, offsets_87

t, nu, diff = load_and_calibrate(1)
m87_loc = (t >= -0.0035) & (t <= -0.0031)
t87 = t[m87_loc][np.argmax(diff[m87_loc])]
nu87_ref = nu[np.argmin(np.abs(t - t87))]
mask = (nu >= nu87_ref - 470) & (nu <= nu87_ref + 30)
x = nu[mask] - nu87_ref
y = diff[mask] * 1e3
popt, model_fn = fit_manifold(nu[mask], diff[mask], offsets_87, nu87_ref)
res = y - model_fn(popt, nu[mask]) * 1e3

print('Maximum positive residual:', np.max(res), 'mV at x =', x[np.argmax(res)], 'MHz')
print('Maximum negative residual:', np.min(res), 'mV at x =', x[np.argmin(res)], 'MHz')

names = ["F'=1", "F'=2", "F'=3", "CO(1,2)", "CO(1,3)", "CO(2,3)"]
for k in range(6):
    x_k = offsets_87[k] + (popt[0] - nu87_ref)
    idx_k = np.argmin(np.abs(x - x_k))
    print(f'{names[k]:10s} at {x_k:6.1f} MHz: data = {y[idx_k]:5.2f} mV, residual = {res[idx_k]:+5.2f} mV')
