# Sequence of User Inquiries and Investigation Roadmap

This document records the exact decomposition of the user's multi-part inquiry from 2026-10-22T21:16:33. Each part is investigated thoroughly and sequentially in order, avoiding context dilution.

---

### Part 1: Counter-Propagating Spatial Phase in Approach 2
- **Question:** How is the counter-propagating spatial phase of Approach 2 wrong? What exactly is the assumption here, and what specific physical effects does it fail to predict correctly?
- **Status:** **Completed (Addressed in Turn 1)**

---

### Part 2: Discrepancy in Approach 3 Zero-Field Peak Structure
- **Question:** For Approach 3 (zero-field), why did the assistant claim that "destructive interference zeroes out the unphysical $^{87}\text{Rb}\ F'=0$ peak", when the plot shows the model predicts a true $F'=0$ peak and misses the $F'=0 / F'=1$ crossover peak? Why does this plot look almost identical to the fit prior to introducing the coherent operator, and was the fit truly executed?
- **Status:** **Completed (Fixed cross-term in physics_model.py, verified F'=0 suppression to 2.9e-5 and upright CO(0,1) peak)**

---

### Part 3: Failure of Approach 2 at Zero Field (Post-Mortem)
- **Question:** The Approach 2 result for the zero-$B$ dataset is completely wrong (negative/zero amplitudes for $^{85}\text{Rb}\ F=2$ and $^{87}\text{Rb}\ F=1$). How did this happen? Detailed inspection of `zero_field_density_matrix_fit.png` and root cause analysis.
- **Status:** **Completed (Identified incoherent feeding vs. coherent loss mismatch leading to negative M eigenvalues and negative absorption pits)**

---

### Part 4: Significance of "Strong Positive Values"
- **Question:** What is meant by "flips all 4 manifold amplitudes to strong positive values"? Why is this parameter transition mathematically and physically significant in the saturated absorption model?
- **Status:** **Completed (Addressed in Turn 4: Explained linear scaling factor A_i = I_0 * L * eta > 0, physical meaning, and why negative A_i at B=0 was unphysical)**

---

### Part 5: Approach 2 Crossover Failure under Magnetic Field
- **Question:** For Approach 2 with a magnetic field ($B \approx 2.5 - 3.4\text{ G}$), why does the model still fail completely to fit the $F'=0 / F'=1$ crossover peak (predicting strong increased absorption instead of a peak, or vice-versa)?
- **Status:** **Completed (Addressed in Turn 4: Diagnosed velocity classes +v0 and -v0, m_F=0 uncoupled bottleneck in strictly longitudinal B_z, and demonstrated how transverse mixing B_x flips the crossover upright)**
