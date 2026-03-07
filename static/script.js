/* ═══════════════════════════════════════════
   Galaxy Predictor — App Logic
   ═══════════════════════════════════════════ */

// ── Presets ──
const presets = {
    star_forming: {
        u: 20.7468, g: 19.5216, r: 18.8356, i: 18.4295, z: 18.158, redshift: 0.0394
    },
    transition: {
        u: 21.4997, g: 19.7431, r: 18.7274, i: 18.2965, z: 18.0088, redshift: 0.1034
    },
    quenched: {
        u: 22.4709, g: 20.15, r: 18.7786, i: 18.2661, z: 18.0014, redshift: 0.1852
    }
};

let evolutionChart = null;
let lossChartLoaded = false;
let evolutionChartLoaded = false;

// ── Chart.js Global Theme ──
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

// ── Preset loading ──
function loadPreset(name) {
    const p = presets[name];
    if (!p) return;

    document.getElementById('u').value = p.u;
    document.getElementById('g').value = p.g;
    document.getElementById('r').value = p.r;
    document.getElementById('i').value = p.i;
    document.getElementById('z').value = p.z;
    document.getElementById('redshift').value = p.redshift;

    // Highlight active preset
    document.querySelectorAll('.btn-preset').forEach(btn => btn.classList.remove('active'));
    const presetMap = { star_forming: 'preset-sf', transition: 'preset-tr', quenched: 'preset-qu' };
    const activeBtn = document.getElementById(presetMap[name]);
    if (activeBtn) activeBtn.classList.add('active');
}

// ── Form submission ──
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const data = {
        u: parseFloat(document.getElementById('u').value),
        g: parseFloat(document.getElementById('g').value),
        r: parseFloat(document.getElementById('r').value),
        i: parseFloat(document.getElementById('i').value),
        z: parseFloat(document.getElementById('z').value),
        redshift: parseFloat(document.getElementById('redshift').value)
    };

    setLoading(true);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!res.ok) throw new Error("API request failed");

        const result = await res.json();
        displayResults(result);
    } catch (err) {
        alert("Prediction failed: " + err.message);
    } finally {
        setLoading(false);
    }
});

// ── Loading state ──
function setLoading(isLoading) {
    const loader = document.getElementById('loading');
    if (isLoading) loader.classList.remove('hidden');
    else loader.classList.add('hidden');
}

// ── Interpretation text ──
function generateInterpretation(r) {
    const q = r.quenching_prob_mean;
    const mass = r.mass_log_mean;
    const sfr = r.sfr_log_mean;

    let state, detail;

    if (q < 0.3) {
        state = "actively star-forming";
        detail = "The galaxy is likely part of the blue-cloud population, exhibiting significant ongoing stellar mass assembly. Its photometric colours are consistent with a young stellar population with active star formation.";
    } else if (q > 0.7) {
        state = "quenched";
        detail = "The galaxy appears to have ceased significant star formation and is likely part of the red-sequence population. Its photometric colours indicate an older, evolved stellar population.";
    } else {
        state = "in a transitional evolutionary state";
        detail = "The galaxy occupies the green valley between star-forming and quenched populations. This transitional phase may indicate recent or ongoing quenching mechanisms.";
    }

    return `This galaxy has an estimated stellar mass of ${mass.toFixed(2)} log M☉ ` +
           `and a star formation rate of ${sfr.toFixed(2)} log SFR. ` +
           `The model classifies it as ${state} (Q = ${q.toFixed(2)}).\n\n${detail}\n\n` +
           `The quenching probability reflects the model's inferred evolutionary status ` +
           `based on photometric colours, luminosity, and redshift.`;
}

// ══════════════════════
//  DISPLAY RESULTS
// ══════════════════════
function displayResults(r) {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('resultsContent').classList.remove('hidden');

    // ── KPI Metrics ──
    animateValue('massVal', r.mass_log_mean, 2);
    animateValue('sfrVal', r.sfr_log_mean, 2);
    document.getElementById('massUncertainty').textContent = `± ${r.mass_log_std.toFixed(2)}`;
    document.getElementById('sfrUncertainty').textContent = `± ${r.sfr_log_std.toFixed(2)}`;

    // ── Quenching ──
    const q = r.quenching_prob_mean;
    const q_std = r.quenching_prob_std;
    document.getElementById('probText').textContent = `Q = ${q.toFixed(2)} ± ${q_std.toFixed(2)}`;
    document.getElementById('probBar').style.width = '100%';
    document.getElementById('probMarker').style.left = `${Math.min(Math.max(q * 100, 0), 100)}%`;

    const badge = document.getElementById('quenchingStatus');
    badge.className = 'badge';
    if (q < 0.3) {
        badge.textContent = 'Star Forming';
        badge.classList.add('blue');
    } else if (q > 0.7) {
        badge.textContent = 'Quenched';
        badge.classList.add('red');
    } else {
        badge.textContent = 'Transitional';
        badge.classList.add('yellow');
    }

    // Histogram
    renderHistogram(r.quenching_posterior);

    // ── Interpretation tab ──
    document.getElementById('interpretationText').textContent = generateInterpretation(r);

    // ── Explainability tab ──
    if (r.mass_feature_importance) renderFeatureImportance('massImportance', r.mass_feature_importance);
    if (r.sfr_feature_importance) renderFeatureImportance('sfrImportance', r.sfr_feature_importance);

    // ── Comparison tab ──
    if (r.rf_mass_log_mean !== null) {
        document.getElementById('pinnCompMass').textContent = `${r.mass_log_mean.toFixed(2)} ± ${r.mass_log_std.toFixed(2)}`;
        document.getElementById('pinnCompSfr').textContent = `${r.sfr_log_mean.toFixed(2)} ± ${r.sfr_log_std.toFixed(2)}`;
        document.getElementById('rfCompMass').textContent = `${r.rf_mass_log_mean.toFixed(2)} ± ${r.rf_mass_log_std.toFixed(2)}`;
        document.getElementById('rfCompSfr').textContent = `${r.rf_sfr_log_mean.toFixed(2)} ± ${r.rf_sfr_log_std.toFixed(2)}`;
    }

    // ── Evolution chart (lazy) ──
    evolutionChartLoaded = false; // re-render on next tab visit
    window._pendingEvolution = { mass: r.mass_log_mean, sfr: r.sfr_log_mean };

    // If the evolution tab is currently active, render immediately
    if (document.getElementById('panel-evolution').classList.contains('active')) {
        renderEvolutionChart(r.mass_log_mean, r.sfr_log_mean);
        evolutionChartLoaded = true;
    }

    // Reset to interpretation tab
    switchTab('interpretation');
}

// ── Animated numeric value ──
function animateValue(id, target, decimals) {
    const el = document.getElementById(id);
    const duration = 600;
    const start = performance.now();
    const from = parseFloat(el.textContent) || 0;

    function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = from + (target - from) * eased;
        el.textContent = current.toFixed(decimals);
        if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
}

// ── Histogram ──
function renderHistogram(samples) {
    const container = document.getElementById('histogram');
    container.innerHTML = '';

    const bins = 30;
    const bucketCounts = new Array(bins).fill(0);
    samples.forEach(v => {
        const idx = Math.floor(v * bins);
        if (idx >= 0 && idx < bins) bucketCounts[idx]++;
    });
    const maxCount = Math.max(...bucketCounts);

    bucketCounts.forEach((count, i) => {
        const bar = document.createElement('div');
        bar.className = 'hist-bar';
        const h = maxCount > 0 ? (count / maxCount) * 100 : 0;
        // Stagger the animation
        bar.style.height = '0%';
        bar.style.transitionDelay = `${i * 15}ms`;
        container.appendChild(bar);
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                bar.style.height = `${h}%`;
            });
        });
    });
}

// ── Feature Importance ──
function renderFeatureImportance(containerId, importance) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const sorted = Object.entries(importance)
        .filter(([k]) => k !== 'Mr_placeholder')
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6);

    sorted.forEach(([feature, value], i) => {
        const percent = (value * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'feature-bar';
        row.innerHTML = `
            <div class="feature-name">${feature}</div>
            <div class="feature-bar-bg">
                <div class="feature-bar-fill" style="width:0%"></div>
            </div>
            <div class="feature-value">${percent}%</div>
        `;
        container.appendChild(row);

        // Animate bar fill
        const fill = row.querySelector('.feature-bar-fill');
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                fill.style.width = `${percent}%`;
                fill.style.transitionDelay = `${i * 60}ms`;
            });
        });
    });
}

// ══════════════════════
//  TAB SYSTEM
// ══════════════════════
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');
const tabIndicator = document.getElementById('tabIndicator');

function switchTab(tabName) {
    tabBtns.forEach(btn => {
        const isActive = btn.dataset.tab === tabName;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive);
    });

    tabPanels.forEach(panel => {
        panel.classList.toggle('active', panel.id === `panel-${tabName}`);
    });

    updateTabIndicator();

    // Lazy-load charts
    if (tabName === 'evolution' && !evolutionChartLoaded && window._pendingEvolution) {
        renderEvolutionChart(window._pendingEvolution.mass, window._pendingEvolution.sfr);
        evolutionChartLoaded = true;
    }
    if (tabName === 'diagnostics' && !lossChartLoaded) {
        loadTrainingLoss();
        lossChartLoaded = true;
    }
}

function updateTabIndicator() {
    const activeBtn = document.querySelector('.tab-btn.active');
    if (!activeBtn || !tabIndicator) return;
    tabIndicator.style.left = `${activeBtn.offsetLeft}px`;
    tabIndicator.style.width = `${activeBtn.offsetWidth}px`;
}

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Re-position indicator on resize
window.addEventListener('resize', updateTabIndicator);

// ══════════════════════
//  CHARTS
// ══════════════════════
function renderEvolutionChart(mass, sfr) {
    const ctx = document.getElementById('evolutionChart');

    const mainSequence = [];
    for (let m = 8; m <= 12; m += 0.2) {
        const base = 0.7 * m - 7;
        for (let i = 0; i < 5; i++) {
            mainSequence.push({
                x: m + (Math.random() - 0.5) * 0.2,
                y: base + (Math.random() - 0.5) * 0.4
            });
        }
    }

    if (evolutionChart) evolutionChart.destroy();

    evolutionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Star-forming main sequence',
                    data: mainSequence,
                    showLine: false,
                    borderColor: 'rgba(96,165,250,0.6)',
                    backgroundColor: 'rgba(96,165,250,0.5)',
                    pointRadius: 2.5,
                    pointHoverRadius: 4
                },
                {
                    label: 'Predicted Galaxy',
                    data: [{ x: mass, y: sfr }],
                    backgroundColor: '#f59e0b',
                    borderColor: '#fff',
                    pointRadius: 9,
                    pointBorderWidth: 2,
                    pointHoverRadius: 12,
                    pointHoverBorderWidth: 3,
                    pointHoverBorderColor: '#fff',
                    pointHoverBackgroundColor: '#f59e0b'
                }
            ]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: 'easeOutQuart' },
            scales: {
                x: {
                    title: { display: true, text: 'log(M*/M☉)', color: '#94a3b8', font: { weight: 500 } },
                    min: 8, max: 12.5,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    title: { display: true, text: 'log(SFR / M☉ yr⁻¹)', color: '#94a3b8', font: { weight: 500 } },
                    min: -3, max: 2,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748b' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#cbd5e1', usePointStyle: true, pointStyle: 'circle', padding: 16 }
                },
                tooltip: {
                    backgroundColor: '#151d2e',
                    titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10
                }
            }
        }
    });
}

async function loadTrainingLoss() {
    try {
        const res = await fetch('/training-loss');
        const data = await res.json();
        const ctx = document.getElementById('lossChart').getContext('2d');

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.epochs,
                datasets: [
                    {
                        label: 'Total Loss',
                        data: data.total_loss,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59,130,246,0.08)',
                        fill: true,
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHitRadius: 10
                    },
                    {
                        label: 'Physics Loss',
                        data: data.physics_loss,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245,158,11,0.06)',
                        fill: true,
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHitRadius: 10
                    }
                ]
            },
            options: {
                responsive: true,
                animation: { duration: 800, easing: 'easeOutQuart' },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: {
                        title: { display: true, text: 'Epoch', color: '#94a3b8', font: { weight: 500 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#64748b', maxTicksLimit: 10 }
                    },
                    y: {
                        title: { display: true, text: 'Loss', color: '#94a3b8', font: { weight: 500 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#64748b' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#cbd5e1', usePointStyle: true, pointStyle: 'circle', padding: 16 }
                    },
                    tooltip: {
                        backgroundColor: '#151d2e',
                        titleColor: '#f1f5f9',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(255,255,255,0.08)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 10
                    }
                }
            }
        });
    } catch (err) {
        console.warn('Training loss data unavailable:', err.message);
    }
}

// ── Init ──
loadPreset('star_forming');

// Position tab indicator once DOM is settled
requestAnimationFrame(() => {
    requestAnimationFrame(() => {
        updateTabIndicator();
    });
});