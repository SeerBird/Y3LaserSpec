import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

fsr = 374.74  # MHz

# Let's inspect F1 to F7 in data/Session 9
s9_dir = r'data\Session 9'
f_files = [f'F{i}.CSV' for i in range(1, 8)]

fig_dir = r'figures\fp_analysis'
os.makedirs(fig_dir, exist_ok=True)
artifact_dir = r'C:\Users\ilyag\Desktop\G\Offline\Code\Y3LaserSpec'

fp_data = {}
for fname in f_files:
    path = os.path.join(s9_dir, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Check column names
        t_col = df.columns[0]
        v_col = df.columns[-1]
        t = df[t_col].values
        v = df[v_col].values
        peaks, props = find_peaks(v, prominence=0.015, distance=100)
        fp_data[fname] = {
            't': t,
            'v': v,
            'peaks': peaks,
            't_peaks': t[peaks],
            'n_peaks': len(peaks)
        }
        print(f"{fname:8s}: {len(t)} samples, {len(peaks)} fringes found. First fringe at t={t[peaks[0]]:.5f}s (idx {peaks[0]}), last at t={t[peaks[-1]]:.5f}s (idx {peaks[-1]})")

# Also check GF1 to GF5 in Magnetic variation
mag_dir = r'data\Session 9\Magnetic variation'
gf_files = [f'GF{i}.CSV' for i in range(1, 6)]
gf_data = {}
for fname in gf_files:
    path = os.path.join(mag_dir, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        t_col = df.columns[0]
        v_col = df.columns[-1]
        t = df[t_col].values
        v = df[v_col].values
        peaks, props = find_peaks(v, prominence=0.01, distance=100)
        gf_data[fname] = {
            't': t,
            'v': v,
            'peaks': peaks,
            't_peaks': t[peaks],
            'n_peaks': len(peaks)
        }
        print(f"{fname:8s} (Mag): {len(t)} samples, {len(peaks)} fringes found. First at t={t[peaks[0]]:.5f}s (idx {peaks[0]}), last at t={t[peaks[-1]]:.5f}s (idx {peaks[-1]})")
