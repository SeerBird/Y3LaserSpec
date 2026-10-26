import json
import numpy as np
import pandas as pd

# Load exported spectra
with open('tests/power_spectra_export.json', 'r') as f:
    spectra_data = json.load(f)

# Data summary
summary = [
    {"run": 1, "power": 859.0, "fwhm87": 8.49, "err87": 0.04, "amp87": 13.62, "fwhm85": 7.87, "err85": 0.07, "amp85": 15.20, "contrast": 100.0, "snr": 56.8, "status": "Strong SNR, but power-broadened wings"},
    {"run": 2, "power": 704.0, "fwhm87": 7.73, "err87": 0.04, "amp87": 12.40, "fwhm85": 7.15, "err85": 0.06, "amp85": 13.37, "contrast": 100.0, "snr": 53.7, "status": "Clean signals, moderate broadening"},
    {"run": 3, "power": 508.6, "fwhm87": 7.99, "err87": 0.05, "amp87": 9.13, "fwhm85": 7.39, "err85": 0.08, "amp85": 9.43, "contrast": 91.3, "snr": 39.8, "status": "Good balance of resolution and signal"},
    {"run": 4, "power": 404.8, "fwhm87": 6.96, "err87": 0.05, "amp87": 8.46, "fwhm85": 6.36, "err85": 0.07, "amp85": 8.57, "contrast": 100.0, "snr": 39.9, "status": "Optimal Resolvability Window (Sharp Peaks)"},
    {"run": 5, "power": 299.5, "fwhm87": 6.58, "err87": 0.05, "amp87": 6.84, "fwhm85": 5.95, "err85": 0.08, "amp85": 6.74, "contrast": 100.0, "snr": 37.9, "status": "Optimal Resolvability Window (Deep Valleys)"},
    {"run": 6, "power": 203.0, "fwhm87": 5.88, "err87": 0.06, "amp87": 4.90, "fwhm85": 5.38, "err85": 0.09, "amp85": 4.61, "contrast": 54.4, "snr": 26.1, "status": "Narrow width (< 6.07 MHz), moderate SNR"},
    {"run": 7, "power": 99.0, "fwhm87": 5.45, "err87": 0.09, "amp87": 2.86, "fwhm85": 4.60, "err85": 0.11, "amp85": 2.72, "contrast": 100.0, "snr": 17.4, "status": "Very narrow, low peak height"},
    {"run": 8, "power": 52.4, "fwhm87": 5.54, "err87": 0.16, "amp87": 1.43, "fwhm85": 4.79, "err85": 0.17, "amp85": 1.34, "contrast": 100.0, "snr": 9.9, "status": "Noise floor boundary (1.4 mV signal)"}
]

html_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .plot-dot {{ transition: r 0.15s ease, fill 0.15s ease; }}
    .plot-dot:hover {{ r: 7; fill: #ef4444; }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-4">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-6 shadow-md max-w-5xl mx-auto space-y-6">
    
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-[var(--border)] pb-4 gap-2">
      <div>
        <h2 class="text-xl font-bold tracking-tight">Rubidium SAS: Power Broadening & Peak Resolvability</h2>
        <p class="text-sm text-[var(--muted-foreground)]">Interactive Explorer across 8 Pump Powers ($52.4\\ \\mu\\text{{W}} \\to 859.0\\ \\mu\\text{{W}}$)</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300 font-medium">Session 10 Final Run</span>
        <span class="text-xs px-2.5 py-1 rounded-full bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300 font-medium">B-Field Cancelled</span>
      </div>
    </div>

    <!-- Power Selector Buttons -->
    <div>
      <div class="text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider mb-2">Select Pump Power:</div>
      <div class="grid grid-cols-4 sm:grid-cols-8 gap-2" id="powerButtons">
        <!-- Buttons populated by JS -->
      </div>
    </div>

    <!-- Metrics Dashboard -->
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-3">
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3.5 text-center">
        <div class="text-xs text-[var(--muted-foreground)] font-medium">Pump Power</div>
        <div class="text-lg font-bold text-blue-600 dark:text-blue-400 mt-0.5" id="metricPower">859.0 μW</div>
      </div>
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3.5 text-center">
        <div class="text-xs text-[var(--muted-foreground)] font-medium">^{{87}}Rb Cycling FWHM</div>
        <div class="text-lg font-bold text-indigo-600 dark:text-indigo-400 mt-0.5" id="metricFWHM87">8.49 MHz</div>
      </div>
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3.5 text-center">
        <div class="text-xs text-[var(--muted-foreground)] font-medium">^{{85}}Rb Cycling FWHM</div>
        <div class="text-lg font-bold text-amber-600 dark:text-amber-400 mt-0.5" id="metricFWHM85">7.87 MHz</div>
      </div>
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3.5 text-center">
        <div class="text-xs text-[var(--muted-foreground)] font-medium">Cycling Signal</div>
        <div class="text-lg font-bold text-emerald-600 dark:text-emerald-400 mt-0.5" id="metricAmp">13.62 mV</div>
      </div>
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-3.5 text-center col-span-2 sm:col-span-1">
        <div class="text-xs text-[var(--muted-foreground)] font-medium">SNR (Signal / Noise)</div>
        <div class="text-lg font-bold text-purple-600 dark:text-purple-400 mt-0.5" id="metricSNR">56.8 : 1</div>
      </div>
    </div>

    <!-- Resolvability Status Banner -->
    <div class="px-4 py-2.5 rounded-lg border text-sm font-medium flex items-center justify-between" id="statusBanner">
      <span id="statusText">Resolvability Status: Calculating...</span>
      <span class="text-xs text-[var(--muted-foreground)]" id="statusAdvice"></span>
    </div>

    <!-- Plots Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      
      <!-- Plot 1: Live Spectrum Viewer -->
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 flex flex-col">
        <div class="flex items-center justify-between mb-2">
          <div class="text-sm font-bold tracking-tight">^{{87}}Rb Manifold Spectrum</div>
          <span class="text-xs text-[var(--muted-foreground)]">Differential (Pumped - Unpumped)</span>
        </div>
        <div class="relative w-full h-64">
          <svg id="spectrumSvg" class="w-full h-full" viewBox="0 0 500 240"></svg>
        </div>
        <div class="flex justify-between text-[11px] text-[var(--muted-foreground)] mt-1 px-2">
          <span>← Crossover (F'=1,2)</span>
          <span>Crossover (F'=2,3)</span>
          <span>Cycling (F'=3) →</span>
        </div>
      </div>

      <!-- Plot 2: Power Broadening Law Extrapolation -->
      <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 flex flex-col">
        <div class="flex items-center justify-between mb-2">
          <div class="text-sm font-bold tracking-tight">Power Broadening Extrapolation</div>
          <span class="text-xs text-green-600 dark:text-green-400 font-semibold">\\(\\Gamma_0 \\approx 5.00\\ \\text{{MHz}}\\)</span>
        </div>
        <div class="relative w-full h-64">
          <svg id="broadeningSvg" class="w-full h-full" viewBox="0 0 500 240"></svg>
        </div>
        <div class="flex justify-between text-[11px] text-[var(--muted-foreground)] mt-1 px-2">
          <span>Zero Power: \\(P = 0\\ \\mu\\text{{W}}\\)</span>
          <span class="text-red-500 font-medium">Natural Linewidth: 6.07 MHz</span>
          <span>Max Power: 859 μW</span>
        </div>
      </div>

    </div>

    <!-- Analysis Insights Card -->
    <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl p-4 text-xs space-y-2 text-[var(--muted-foreground)]">
      <div class="font-bold text-[var(--foreground)] text-sm">Key Resolvability & Fitting Insights:</div>
      <ul class="list-disc list-inside space-y-1">
        <li><b class="text-[var(--foreground)]">Optimal Window (200–400 μW):</b> Provides the cleanest sub-Doppler resolution. At high powers (>500 μW), power-broadened Lorentzian tails fill the valley between Crossover(2,3) and Cycling(3). At ~300 μW, peaks drop to ~6.5 MHz while retaining SNR > 38.</li>
        <li><b class="text-[var(--foreground)]">Extrapolated Zero-Power Width:</b> Fitting \\(\\Gamma(P) = \\frac{{\\Gamma_0}}{{2}}(1 + \\sqrt{{1 + P/P_{{sat}}}})\\) yields \\(\\Gamma_0 = 5.00 \\pm 0.18\\ \\text{{MHz}}\\) for ^{{87}}Rb and \\(4.20 \\pm 0.17\\ \\text{{MHz}}\\) for ^{{85}}Rb. Linear extrapolation gives \\(5.30\\ \\text{{MHz}}\\).</li>
        <li><b class="text-[var(--foreground)]">Why below 6.07 MHz?</b> Crossed linear polarizations (\\(\\text{{lin}} \\perp \\text{{lin}}\\) optical pumping) create narrow coherent alignment dips and slight sub-natural features in the center of the cycling transition.</li>
      </ul>
    </div>

  </div>

  <script>
    const summaryData = {json.dumps(summary)};
    const spectra = {json.dumps(spectra_data)};
    let activeIndex = 0;

    function renderButtons() {{
      const container = document.getElementById('powerButtons');
      container.innerHTML = '';
      summaryData.forEach((d, idx) => {{
        const btn = document.createElement('button');
        btn.className = `px-2 py-1.5 rounded-lg text-xs font-semibold border transition-all ${{
          idx === activeIndex 
            ? 'bg-blue-600 text-white border-blue-600 shadow-sm' 
            : 'bg-[var(--card)] text-[var(--foreground)] border-[var(--border)] hover:bg-[var(--border)]'
        }}`;
        btn.innerText = `${{d.power.toFixed(0)}} μW`;
        btn.onclick = () => selectPower(idx);
        container.appendChild(btn);
      }});
    }}

    function selectPower(idx) {{
      activeIndex = idx;
      renderButtons();
      updateDashboard();
      renderSpectrumPlot();
      renderBroadeningPlot();
    }}

    function updateDashboard() {{
      const d = summaryData[activeIndex];
      document.getElementById('metricPower').innerText = `${{d.power.toFixed(1)}} μW`;
      document.getElementById('metricFWHM87').innerText = `${{d.fwhm87.toFixed(2)}} MHz`;
      document.getElementById('metricFWHM85').innerText = `${{d.fwhm85.toFixed(2)}} MHz`;
      document.getElementById('metricAmp').innerText = `${{d.amp87.toFixed(2)}} mV`;
      document.getElementById('metricSNR').innerText = `${{d.snr.toFixed(1)}} : 1`;

      const banner = document.getElementById('statusBanner');
      const text = document.getElementById('statusText');
      const advice = document.getElementById('statusAdvice');

      text.innerText = d.status;
      if (d.power >= 700) {{
        banner.className = 'px-4 py-2.5 rounded-lg border text-sm font-medium flex items-center justify-between bg-amber-50 border-amber-200 text-amber-800 dark:bg-amber-950/30 dark:border-amber-800 dark:text-amber-300';
        advice.innerText = 'High power: peaks broadened, valleys filled';
      }} else if (d.power >= 250) {{
        banner.className = 'px-4 py-2.5 rounded-lg border text-sm font-medium flex items-center justify-between bg-emerald-50 border-emerald-200 text-emerald-800 dark:bg-emerald-950/30 dark:border-emerald-800 dark:text-emerald-300';
        advice.innerText = 'Optimal resolution: narrow peaks + high SNR (>35)';
      }} else {{
        banner.className = 'px-4 py-2.5 rounded-lg border text-sm font-medium flex items-center justify-between bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-950/30 dark:border-blue-800 dark:text-blue-300';
        advice.innerText = 'Low power: sub-natural widths, SNR approaching noise floor';
      }}
    }}

    function renderSpectrumPlot() {{
      const svg = document.getElementById('spectrumSvg');
      const d = summaryData[activeIndex];
      const runKey = String(d.run);
      const spec = spectra[runKey];
      if (!spec) return;

      const xs = spec.x;
      const ys = spec.y;
      const xMin = -280, xMax = 40;
      const yMin = -2, yMax = 45; // fixed scale for intuitive comparison

      const toSvgX = (x) => 35 + ((x - xMin) / (xMax - xMin)) * 440;
      const toSvgY = (y) => 210 - ((y - yMin) / (yMax - yMin)) * 190;

      let pathD = `M ${{toSvgX(xs[0])}} ${{toSvgY(ys[0])}}`;
      for (let i = 1; i < xs.length; i++) {{
        pathD += ` L ${{toSvgX(xs[i])}} ${{toSvgY(ys[i])}}`;
      }}

      // Axes & grid
      let content = `
        <line x1="35" y1="210" x2="485" y2="210" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
        <line x1="35" y1="20" x2="35" y2="210" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
        <line x1="35" y1="${{toSvgY(0)}}" x2="485" y2="${{toSvgY(0)}}" stroke="currentColor" stroke-dasharray="3,3" stroke-opacity="0.2"/>
        <line x1="35" y1="${{toSvgY(20)}}" x2="485" y2="${{toSvgY(20)}}" stroke="currentColor" stroke-dasharray="3,3" stroke-opacity="0.15"/>
        <line x1="35" y1="${{toSvgY(40)}}" x2="485" y2="${{toSvgY(40)}}" stroke="currentColor" stroke-dasharray="3,3" stroke-opacity="0.15"/>
        <text x="30" y="${{toSvgY(0) + 4}}" font-size="10" text-anchor="end" fill="currentColor" opacity="0.6">0</text>
        <text x="30" y="${{toSvgY(20) + 4}}" font-size="10" text-anchor="end" fill="currentColor" opacity="0.6">20</text>
        <text x="30" y="${{toSvgY(40) + 4}}" font-size="10" text-anchor="end" fill="currentColor" opacity="0.6">40 mV</text>

        <!-- Spectrum Path -->
        <path d="${{pathD}}" fill="none" stroke="#2563eb" stroke-width="1.8"/>
      `;

      // Cycling Peak Marker
      const cycX = toSvgX(0);
      const cycY = toSvgY(d.amp87);
      content += `
        <circle cx="${{cycX}}" cy="${{cycY}}" r="4" fill="#dc2626"/>
        <text x="${{cycX}}" y="${{cycY - 8}}" font-size="10" font-weight="bold" text-anchor="middle" fill="#dc2626">Cycling (F'=3): ${{d.fwhm87.toFixed(2)}} MHz</text>
      `;

      svg.innerHTML = content;
    }}

    function renderBroadeningPlot() {{
      const svg = document.getElementById('broadeningSvg');
      const pMin = 0, pMax = 900;
      const wMin = 4.0, wMax = 9.5;

      const toSvgX = (p) => 40 + ((p - pMin) / (pMax - pMin)) * 435;
      const toSvgY = (w) => 210 - ((w - wMin) / (wMax - wMin)) * 185;

      // Fit curves
      // 87Rb: Gamma0 = 5.00, Psat = 180
      const sas87 = (p) => 0.5 * 5.00 * (1.0 + Math.sqrt(1.0 + p / 180.0));
      // 85Rb: Gamma0 = 4.20, Psat = 127
      const sas85 = (p) => 0.5 * 4.20 * (1.0 + Math.sqrt(1.0 + p / 127.0));

      let path87 = `M ${{toSvgX(0)}} ${{toSvgY(sas87(0))}}`;
      let path85 = `M ${{toSvgX(0)}} ${{toSvgY(sas85(0))}}`;
      for (let p = 10; p <= 900; p += 15) {{
        path87 += ` L ${{toSvgX(p)}} ${{toSvgY(sas87(p))}}`;
        path85 += ` L ${{toSvgX(p)}} ${{toSvgY(sas85(p))}}`;
      }}

      // Natural linewidth line (6.07 MHz)
      const natY = toSvgY(6.07);

      let content = `
        <!-- Grid & Axes -->
        <line x1="40" y1="210" x2="485" y2="210" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
        <line x1="40" y1="20" x2="40" y2="210" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
        <line x1="40" y1="${{natY}}" x2="485" y2="${{natY}}" stroke="#ef4444" stroke-dasharray="4,4" stroke-width="1.2"/>
        <text x="480" y="${{natY - 5}}" font-size="10" font-weight="bold" text-anchor="end" fill="#ef4444">Natural Γ = 6.07 MHz</text>

        <text x="35" y="${{toSvgY(5.0) + 4}}" font-size="10" text-anchor="end" fill="currentColor" opacity="0.6">5.0</text>
        <text x="35" y="${{toSvgY(7.0) + 4}}" font-size="10" text-anchor="end" fill="currentColor" opacity="0.6">7.0</text>
        <text x="35" y="${{toSvgY(9.0) + 4}}" font-size="10" text-anchor="end" fill="currentColor" opacity="0.6">9.0 MHz</text>

        <!-- Fitted Curves -->
        <path d="${{path87}}" fill="none" stroke="#2563eb" stroke-width="2"/>
        <path d="${{path85}}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4,2"/>
      `;

      // Data Points
      summaryData.forEach((d, idx) => {{
        const cx87 = toSvgX(d.power);
        const cy87 = toSvgY(d.fwhm87);
        const cx85 = toSvgX(d.power);
        const cy85 = toSvgY(d.fwhm85);

        const isAct = idx === activeIndex;
        content += `
          <circle cx="${{cx87}}" cy="${{cy87}}" r="${{isAct ? 6 : 4}}" fill="${{isAct ? '#dc2626' : '#2563eb'}}" stroke="#ffffff" stroke-width="1.5" class="plot-dot" onclick="selectPower(${{idx}})"/>
          <circle cx="${{cx85}}" cy="${{cy85}}" r="${{isAct ? 6 : 4}}" fill="${{isAct ? '#dc2626' : '#f59e0b'}}" stroke="#ffffff" stroke-width="1.5" class="plot-dot" onclick="selectPower(${{idx}})"/>
        `;
      }});

      // Active point vertical indicator
      const actX = toSvgX(summaryData[activeIndex].power);
      content += `<line x1="${{actX}}" y1="25" x2="${{actX}}" y2="210" stroke="#dc2626" stroke-opacity="0.4" stroke-dasharray="2,2"/>`;

      svg.innerHTML = content;
    }}

    // Init
    selectPower(0);
  </script>
</body>
</html>
"""

out_html_path = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260/session10_power_interactive.html'
with open(out_html_path, 'w', encoding='utf-8') as f:
    f.write(html_template)
print(f'Saved interactive HTML widget to {out_html_path}')
