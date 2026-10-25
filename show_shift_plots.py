"""
Interactive script to display background baseline zoom plots and 
unshifted vs. shift-aligned manifold fit comparisons with matplotlib GUI (plt.show()),
allowing interactive zooming, panning, and inspecting.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import main

WINDOWS_ALL = [
    [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)],  # Prot1
    [(2.0, 3.3), (5.5, 6.2), (13.4, 14.0), (19.9, 20.8)],  # Prot2
    [(1.9, 3.2), (5.5, 6.2), (13.4, 14.0), (19.8, 20.7)],  # Prot3
    [(2.4, 3.8), (6.1, 6.8), (14.0, 14.5), (20.4, 21.3)],  # Prot4
    [(3.3, 4.7), (7.0, 7.7), (14.9, 15.4), (21.2, 22.1)],  # Prot5
    [(2.4, 3.7), (6.0, 6.7), (13.9, 14.5), (20.3, 21.2)],  # Prot6
    [(2.7, 4.0), (6.3, 7.0), (14.2, 14.7), (20.5, 21.5)]   # Prot7
]

FSR = 374.74

LIT_CONSTANTS = {
    '87Rb F=2': (84.72, 12.50),
    '85Rb F=3': (25.00, 25.79),
    '85Rb F=2': (25.00, 25.79),
    '87Rb F=1': (84.72, 12.50)
}


def show_prot5_background():
    """Figure 1: Protocol 5 Full-Scan Zoomed Background Baseline vs. Raw Transmission."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.canvas.manager.set_window_title('Protocol 5: Zoomed Background Baseline')
    fig.suptitle('Protocol 5: Full-Scan Zoomed Background Baseline vs. Raw Transmission', 
                 fontsize=13, fontweight='bold')

    f5, p5, u5 = "data/Session 9/F5.CSV", "data/Session 9/P5.CSV", "data/Session 9/U5.CSV"
    freq5_u, _, p5_u, u5_u = main.prep_data(f5, p5, u5, align_shift=False)
    freq5_s, _, p5_s, u5_s = main.prep_data(f5, p5, u5, align_shift=True)
    diff5_u = p5_u - u5_u
    diff5_s = p5_s - u5_s

    # Raw Transmission
    ax1.plot(freq5_u, p5_u, color='crimson', lw=1.2, alpha=0.7, label='Pumped Transmission')
    ax1.plot(freq5_u, u5_u, color='royalblue', lw=1.2, alpha=0.7, label='Unpumped Transmission')
    ax1.set_ylabel('Photodiode Signal (V)', fontsize=10)
    ax1.set_title('Raw Transmission Signals (Full Vertical Scale)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=9)

    # Zoomed Differential Baseline
    ax2.plot(freq5_u, diff5_u, color='lightcoral', lw=1.0, alpha=0.8, label='Unshifted Diff (P - U)')
    ax2.plot(freq5_s, diff5_s, color='navy', lw=1.3, label='Shift-Aligned Diff (+78.5 samples / +8.3 MHz)')
    for w_idx, (w0, w1) in enumerate(WINDOWS_ALL[4]):
        m_name = list(LIT_CONSTANTS.keys())[w_idx]
        ax2.axvspan(w0, w1, color='gold', alpha=0.25, label='Manifold Window' if w_idx == 0 else "")
        ax2.text(0.5 * (w0 + w1), 0.022, f"{m_name}\n({w0}, {w1})", ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color='darkgoldenrod')

    ax2.set_ylim(-0.035, 0.035)
    ax2.set_xlabel('Fabry-Perot Fringe Index', fontsize=11)
    ax2.set_ylabel('Differential Voltage (V)', fontsize=10)
    ax2.set_title('Zoomed-In Background Baseline (Y-Limits: -35 mV to +35 mV, Peak Tops Truncated)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=9)

    sec_ax2 = ax2.secondary_xaxis('top', functions=(lambda fringe: fringe * FSR, lambda nu: nu / FSR))
    sec_ax2.set_xlabel('Frequency (MHz)', fontsize=10)

    plt.tight_layout()
    return fig


def show_protocol_fits(prot_index=5):
    """Figure: 4-Manifold Fits (Unshifted vs. Shift-Aligned) for a given protocol (1 to 7)."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.canvas.manager.set_window_title(f'Protocol {prot_index}: Manifold Fits Comparison')
    fig.suptitle(f"Protocol {prot_index}: Hyperfine Manifold Fits (Unshifted vs. Shift-Aligned)", 
                 fontsize=14, fontweight='bold', y=0.995)

    f_path = f"data/Session 9/F{prot_index}.CSV"
    p_path = f"data/Session 9/P{prot_index}.CSV"
    u_path = f"data/Session 9/U{prot_index}.CSV"
    wins = WINDOWS_ALL[prot_index - 1]

    freq_u, _, p_u, u_u = main.prep_data(f_path, p_path, u_path, align_shift=False)
    diff_u = p_u - u_u
    freq_s, _, p_s, u_s = main.prep_data(f_path, p_path, u_path, align_shift=True)
    diff_s = p_s - u_s

    res_u = main.fit_hyperfine_multiplets(freq_u, diff_u, windows=wins, fsr=FSR)
    res_s = main.fit_hyperfine_multiplets(freq_s, diff_s, windows=wins, fsr=FSR)

    for m_idx in range(4):
        ax = axs.flat[m_idx]
        ru = res_u[m_idx]
        rs = res_s[m_idx]
        name = ru['name']
        A_lit, B_lit = LIT_CONSTANTS[name]

        # Unshifted
        ax.plot(ru['x_data'], ru['y_data'], color='lightcoral', lw=1.2, alpha=0.6, label='Unshifted Data')
        ax.plot(ru['x_fit'], ru['y_fit'], color='firebrick', ls='--', lw=1.8, 
                label=f"Unshifted: A={ru['A']:.2f}, B={ru['B']:.2f} MHz (cost={ru['cost']:.4f})")
        ax.plot(ru['x_fit'], ru['bg_dense'], color='firebrick', ls=':', lw=1.0, alpha=0.6)

        # Shifted
        ax.plot(rs['x_data'], rs['y_data'], color='black', lw=1.2, alpha=0.7, label='Shift-Aligned Data')
        ax.plot(rs['x_fit'], rs['y_fit'], color='royalblue', ls='-', lw=2.0, 
                label=f"Shifted: A={rs['A']:.2f}, B={rs['B']:.2f} MHz (cost={rs['cost']:.4f})")
        ax.plot(rs['x_fit'], rs['bg_dense'], color='royalblue', ls=':', lw=1.2, alpha=0.8, label='Shifted Baseline')

        sec_ax = ax.secondary_xaxis('top', functions=(lambda nu: nu / FSR, lambda fringe: fringe * FSR))
        sec_ax.set_xlabel('Fringe Index', fontsize=9)

        ax.set_title(f"{name} (Literature: A={A_lit:.2f}, B={B_lit:.2f} MHz)", fontsize=11, fontweight='bold', pad=22)
        ax.set_xlabel('Frequency (MHz)', fontsize=10)
        ax.set_ylabel('Voltage (V)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig


def show_all_protocols_background():
    """Figure: Stacked Full-Scan Zoomed Background Baseline across all 7 protocols."""
    fig, axs = plt.subplots(7, 1, figsize=(16, 20), sharex=False)
    fig.canvas.manager.set_window_title('All Protocols: Zoomed Background Comparison')
    fig.suptitle('Full-Scan Differential Transmission: Zoomed In On Background Baseline (Peak Tops Excluded)',
                 fontsize=14, fontweight='bold', y=0.995)

    for i in range(7):
        ax = axs[i]
        f_path = f"data/Session 9/F{i+1}.CSV"
        p_path = f"data/Session 9/P{i+1}.CSV"
        u_path = f"data/Session 9/U{i+1}.CSV"

        freq_u, _, p_u, u_u = main.prep_data(f_path, p_path, u_path, align_shift=False)
        diff_u = p_u - u_u
        freq_s, _, p_s, u_s = main.prep_data(f_path, p_path, u_path, align_shift=True)
        diff_s = p_s - u_s

        ax.plot(freq_u, diff_u, color='lightcoral', lw=0.9, alpha=0.7, label='Unshifted Diff')
        ax.plot(freq_s, diff_s, color='navy', lw=1.1, alpha=0.85, label='Shift-Aligned Diff')

        wins = WINDOWS_ALL[i]
        for w_idx, (w0, w1) in enumerate(wins):
            m_name = list(LIT_CONSTANTS.keys())[w_idx]
            ax.axvspan(w0, w1, color='gold', alpha=0.2, label='Manifold Window' if w_idx == 0 else "")
            ax.text(0.5 * (w0 + w1), 0.025, m_name.replace(' ', '\n'), ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='darkgoldenrod')

        ax.set_ylim(-0.045, 0.045)
        ax.set_ylabel(f'Prot{i+1} (V)', fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
        if i == 6:
            ax.set_xlabel('Fabry-Perot Fringe Index', fontsize=11)

    plt.tight_layout()
    return fig


def show_prot1_residuals():
    """Figure: Protocol 1 Manifold Fits with Sub-Panel Residuals."""
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.manager.set_window_title('Protocol 1: Fits and Residuals')
    fig.suptitle('Protocol 1: Hyperfine Manifold Fits & Residuals (Unshifted vs. Shift-Aligned)', 
                 fontsize=15, fontweight='bold', y=0.995)
    outer_grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)

    freq_u1, _, pu1, uu1 = main.prep_data('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', align_shift=False)
    freq_s1, _, ps1, us1 = main.prep_data('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', align_shift=True)
    res_u1 = main.fit_hyperfine_multiplets(freq_u1, pu1 - uu1, windows=WINDOWS_ALL[0], fsr=FSR)
    res_s1 = main.fit_hyperfine_multiplets(freq_s1, ps1 - us1, windows=WINDOWS_ALL[0], fsr=FSR)

    for i in range(4):
        ru = res_u1[i]
        rs = res_s1[i]
        name = ru['name']
        A_lit, B_lit = LIT_CONSTANTS[name]
        sub_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[i], height_ratios=[3, 1], hspace=0.08)
        ax_main = fig.add_subplot(sub_grid[0])
        ax_res = fig.add_subplot(sub_grid[1], sharex=ax_main)

        sec_ax = ax_main.secondary_xaxis('top', functions=(lambda nu: nu / FSR, lambda fringe: fringe * FSR))
        sec_ax.set_xlabel('Fabry-Perot Fringe Index', fontsize=9)

        ax_main.plot(ru['x_data'], ru['y_data'], color='lightcoral', alpha=0.5, lw=1.2, label='Unshifted Data')
        ax_main.plot(ru['x_fit'], ru['y_fit'], color='firebrick', ls='--', lw=1.8, 
                     label=f"Unshifted: A={ru['A']:.2f}, B={ru['B']:.2f} MHz (cost={ru['cost']:.4f})")
        ax_main.plot(ru['x_fit'], ru['bg_dense'], color='firebrick', ls=':', lw=1.0, alpha=0.6)

        ax_main.plot(rs['x_data'], rs['y_data'], color='black', alpha=0.6, lw=1.2, label='Shift-Aligned Data')
        ax_main.plot(rs['x_fit'], rs['y_fit'], color='royalblue', ls='-', lw=2.0, 
                     label=f"Shift-Aligned: A={rs['A']:.2f}, B={rs['B']:.2f} MHz (cost={rs['cost']:.4f})")
        ax_main.plot(rs['x_fit'], rs['bg_dense'], color='royalblue', ls=':', lw=1.2, alpha=0.8, label='Shifted Baseline')

        ax_main.set_ylabel('Voltage (V)', fontsize=10)
        title_str = f"{name} (Literature: A={A_lit:.2f}, B={B_lit:.2f} MHz)\n"
        if name == '85Rb F=2':
            title_str += f"★ A improved: {ru['A']:.2f} -> {rs['A']:.2f} MHz (Cost -{100*(ru['cost']-rs['cost'])/ru['cost']:.1f}%)"
        else:
            title_str += f"Cost: {ru['cost']:.4f} -> {rs['cost']:.4f}"
        ax_main.set_title(title_str, fontsize=10, fontweight='bold', pad=22)
        ax_main.grid(True, linestyle=':', alpha=0.5)
        ax_main.legend(loc='upper right', fontsize=8, framealpha=0.9)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        def eval_model(res_dict):
            peak_freqs = res_dict['peak_freqs']
            gamma = res_dict['gamma']
            bg = res_dict['bg']
            amps = res_dict['amplitudes']
            x = res_dict['x_data']
            y = np.full_like(x, bg)
            for k in range(len(amps)):
                y += amps[k] / (1.0 + 4.0 * ((x - peak_freqs[k]) / gamma) ** 2)
            return res_dict['y_data'] - y

        resid_u = eval_model(ru)
        resid_s = eval_model(rs)

        ax_res.plot(ru['x_data'], resid_u, color='firebrick', alpha=0.6, lw=1.0, label='Unshifted Resid')
        ax_res.plot(rs['x_data'], resid_s, color='royalblue', alpha=0.8, lw=1.0, label='Shifted Resid')
        ax_res.axhline(0, color='gray', linestyle='--', lw=0.8)
        ax_res.set_ylabel('Residual (V)', fontsize=8)
        ax_res.set_xlabel('Frequency (MHz)', fontsize=10)
        ax_res.grid(True, linestyle=':', alpha=0.5)
        ax_res.legend(loc='lower right', fontsize=7)

    # Using gridspec hspace/wspace instead of tight_layout to avoid sub-gridspec warning
    return fig


if __name__ == '__main__':
    print("Building interactive plots...")
    
    # 1. Protocol 5 Zoomed Background (requested specifically)
    print("  -> Creating Protocol 5 Background Zoom window...")
    show_prot5_background()

    # 2. Protocol 5 Manifold Fits (requested specifically)
    print("  -> Creating Protocol 5 Manifold Fits window...")
    show_protocol_fits(5)

    # 3. All Protocols Zoomed Background Comparison
    print("  -> Creating All Protocols Background window...")
    show_all_protocols_background()

    # 4. Protocol 1 Fits & Residuals breakdown
    print("  -> Creating Protocol 1 Fits & Residuals window...")
    show_prot1_residuals()

    # If you want to also inspect other protocols, you can uncomment any of:
    # show_protocol_fits(1)
    # show_protocol_fits(2)
    # show_protocol_fits(3)
    # show_protocol_fits(4)
    # show_protocol_fits(6)
    # show_protocol_fits(7)

    print("\nDisplaying all figure windows. Use matplotlib GUI tools to pan, zoom, and inspect.")
    plt.show()
