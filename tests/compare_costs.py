import sys
sys.path.insert(0, '.')
import numpy as np
import tests.saturation_investigation as si

slices1 = si.prepare_manifold_slices('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', si.windows1)
fits1 = si.run_all_fits(slices1)

slices5 = si.prepare_manifold_slices('data/Session 9/F5.CSV', 'data/Session 9/P5.CSV', 'data/Session 9/U5.CSV', si.windows5)
fits5 = si.run_all_fits(slices5)

print("=== PROTOCOL 1 COST & PARAMETERS ===")
cost_m0 = sum(m['fit'].cost for m in fits1['M0'])
gammas_m0 = [m['gamma'] for m in fits1['M0']]
print(f"Model 0 (4 Independent Gammas, one per manifold):")
print(f"  Cost: {cost_m0:.4f}")
print(f"  Gammas: {', '.join(f'{g:.2f} MHz' for g in gammas_m0)}")

cost_m2 = fits1['M2_fit87'].cost + fits1['M2_fit85'].cost
g0_87 = fits1['M2_fit87'].x[8]
s_87 = fits1['M2_fit87'].x[9]
g0_85 = fits1['M2_fit85'].x[8]
s_85 = fits1['M2_fit85'].x[9]
print(f"\nModel 2 (2 Gammas & 2 s parameters, shared per isotope):")
print(f"  Cost: {cost_m2:.4f}")
print(f"  Rb87: Gamma_0 = {g0_87:.2f} MHz, s = {s_87:.3f} | Gammas in [{(g0_87*np.sqrt(1+s_87*0.057)):.1f}, {(g0_87*np.sqrt(1+s_87)):.1f}] MHz")
print(f"  Rb85: Gamma_0 = {g0_85:.2f} MHz, s = {s_85:.3f} | Gammas in [{(g0_85*np.sqrt(1+s_85*0.063)):.1f}, {(g0_85*np.sqrt(1+s_85)):.1f}] MHz")

cost_m3 = fits1['M3_fit'].cost
g0_glob = fits1['M3_fit'].x[16]
s_glob = fits1['M3_fit'].x[17]
print(f"\nModel 3 (1 Global Gamma & 1 s parameter, shared across both isotopes):")
print(f"  Cost: {cost_m3:.4f}")
print(f"  Global: Gamma_0 = {g0_glob:.2f} MHz, s_beam = {s_glob:.3f}")

print("\n=== PROTOCOL 5 COST & PARAMETERS ===")
cost_m0_5 = sum(m['fit'].cost for m in fits5['M0'])
gammas_m0_5 = [m['gamma'] for m in fits5['M0']]
print(f"Model 0 (4 Independent Gammas, one per manifold):")
print(f"  Cost: {cost_m0_5:.4f}")
print(f"  Gammas: {', '.join(f'{g:.2f} MHz' for g in gammas_m0_5)}")

cost_m2_5 = fits5['M2_fit87'].cost + fits5['M2_fit85'].cost
g0_87_5 = fits5['M2_fit87'].x[8]
s_87_5 = fits5['M2_fit87'].x[9]
g0_85_5 = fits5['M2_fit85'].x[8]
s_85_5 = fits5['M2_fit85'].x[9]
print(f"\nModel 2 (2 Gammas & 2 s parameters, shared per isotope):")
print(f"  Cost: {cost_m2_5:.4f}")
print(f"  Rb87: Gamma_0 = {g0_87_5:.2f} MHz, s = {s_87_5:.3f}")
print(f"  Rb85: Gamma_0 = {g0_85_5:.2f} MHz, s = {s_85_5:.3f}")

cost_m3_5 = fits5['M3_fit'].cost
g0_glob_5 = fits5['M3_fit'].x[16]
s_glob_5 = fits5['M3_fit'].x[17]
print(f"\nModel 3 (1 Global Gamma & 1 s parameter, shared across both isotopes):")
print(f"  Cost: {cost_m3_5:.4f}")
print(f"  Global: Gamma_0 = {g0_glob_5:.2f} MHz, s_beam = {s_glob_5:.3f}")
