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

function loadPreset(name) {
    const p = presets[name];
    if (p) {
        document.getElementById('u').value = p.u;
        document.getElementById('g').value = p.g;
        document.getElementById('r').value = p.r;
        document.getElementById('i').value = p.i;
        document.getElementById('z').value = p.z;
        document.getElementById('redshift').value = p.redshift;
    }
}

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

function setLoading(isLoading) {
    const loader = document.getElementById('loading');
    if (isLoading) loader.classList.remove('hidden');
    else loader.classList.add('hidden');
}

function displayResults(r) {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('resultsContent').classList.remove('hidden');

    // Mass & SFR
    document.getElementById('massVal').innerText = r.mass_log_mean.toFixed(2);
    document.getElementById('massErr').innerText = r.mass_log_std.toFixed(2);
    document.getElementById('sfrVal').innerText = r.sfr_log_mean.toFixed(2);
    document.getElementById('sfrErr').innerText = r.sfr_log_std.toFixed(2);

    // Quenching
    const q = r.quenching_prob_mean;
    const q_std = r.quenching_prob_std;
    document.getElementById('probText').innerText = `Q = ${q.toFixed(2)} ± ${q_std.toFixed(2)}`;
    
    // Bar
    // We want the fill to just visually represent 100% full, but maybe we show the marker position?
    // Let's make the fill full gradient background, and use a marker for the value.
    // CSS already has gradient bg for prob-bar-fill, but let's just make it 100% width always 
    // and use the marker? No, wait.
    // Let's just fix the width of bar-fill to 100% static in CSS? 
    // Actually, let's use the marker for the mean.
    document.getElementById('probBar').style.width = '100%';
    document.getElementById('probMarker').style.left = `${Math.min(Math.max(q*100, 0), 100)}%`;

    // Status Badge
    const badge = document.getElementById('quenchingStatus');
    badge.className = 'badge';
    if (q < 0.3) {
        badge.innerText = 'Star Forming';
        badge.classList.add('blue');
    } else if (q > 0.7) {
        badge.innerText = 'Quenched';
        badge.classList.add('red');
    } else {
        badge.innerText = 'Transitional / Uncertain';
        badge.classList.add('yellow');
    }

    // Histogram
    const histContainer = document.getElementById('histogram');
    histContainer.innerHTML = '';
    // Generate buckets
    const samples = r.quenching_posterior;
    const bins = 30;
    const bucketCounts = new Array(bins).fill(0);
    samples.forEach(v => {
        const binIdx = Math.floor(v * bins);
        if (binIdx >= 0 && binIdx < bins) bucketCounts[binIdx]++;
    });
    const maxCount = Math.max(...bucketCounts);
    
    bucketCounts.forEach(count => {
        const bar = document.createElement('div');
        bar.className = 'hist-bar';
        // Normalize height
        const h = maxCount > 0 ? (count / maxCount) * 100 : 0;
        bar.style.height = `${h}%`;
        histContainer.appendChild(bar);
    });

    // Comparison
    if (r.rf_mass_log_mean !== null) {
        document.getElementById('pinnCompMass').innerText = `${r.mass_log_mean.toFixed(2)} ± ${r.mass_log_std.toFixed(2)}`;
        document.getElementById('pinnCompSfr').innerText = `${r.sfr_log_mean.toFixed(2)} ± ${r.sfr_log_std.toFixed(2)}`;
        document.getElementById('rfCompMass').innerText = `${r.rf_mass_log_mean.toFixed(2)} ± ${r.rf_mass_log_std.toFixed(2)}`;
        document.getElementById('rfCompSfr').innerText = `${r.rf_sfr_log_mean.toFixed(2)} ± ${r.rf_sfr_log_std.toFixed(2)}`;
    }
}

// Init with star forming values but don't submit
loadPreset('star_forming');
