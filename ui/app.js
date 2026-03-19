let API_URL = localStorage.getItem('vectordb_api_url') || ""; // Defaults to relative

// Init
document.addEventListener('DOMContentLoaded', () => {
    // Set initial input value
    document.getElementById('api-url-input').value = API_URL || window.location.origin;

    fetchStats();
    setInterval(fetchStats, 5000); // Refresh every 5s
    generateRandomVector(); // Pre-fill
});

function switchView(viewId, navElement) {
    // Nav active state
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    navElement.classList.add('active');

    // View visibility
    document.querySelectorAll('.view-section').forEach(el => el.style.display = 'none');
    document.getElementById(`view-${viewId}`).style.display = 'block';
}

function saveSettings() {
    const input = document.getElementById('api-url-input').value;
    // Remove trailing slash
    const cleanUrl = input.replace(/\/$/, "");
    localStorage.setItem('vectordb_api_url', cleanUrl);
    API_URL = cleanUrl;

    alert("Settings saved!");
    fetchStats(); // Update status immediately
}

async function fetchStats() {
    try {
        const res = await fetch(`${API_URL}/stats`);
        const data = await res.json();

        document.getElementById('stat-count').textContent = data.vector_count.toLocaleString();
        document.getElementById('stat-dim').textContent = data.dimension;
        document.getElementById('stat-status').textContent = data.status;

        document.getElementById('conn-status').innerHTML = '<span class="dot"></span> Connected';
        document.getElementById('conn-status').classList.remove('error');
    } catch (e) {
        document.getElementById('conn-status').innerHTML = '<span class="dot" style="background:red; box-shadow:none;"></span> Disconnected';
        document.getElementById('conn-status').classList.add('error');
    }
}

function generateRandomVector() {
    // Generate 128 dim vector
    const vec = Array.from({ length: 128 }, () => Math.random());
    document.getElementById('search-vector').value = JSON.stringify(vec);
}

async function performSearch() {
    const input = document.getElementById('search-vector').value;
    let vector;
    try {
        vector = JSON.parse(input);
    } catch (e) {
        alert("Invalid JSON vector");
        return;
    }

    const container = document.getElementById('results-container');
    container.innerHTML = '<div class="empty-state">Searching...</div>';

    try {
        const payload = {
            vector: vector,
            k: 10,
            filter: {}
        };

        const res = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        container.innerHTML = '';
        if (data.results.length === 0) {
            container.innerHTML = '<div class="empty-state">No results found</div>';
            return;
        }

        data.results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card';
            // Scale score for display (HNSW score is dist, so lower is better for L2, 
            // but API returns raw distance usually? 
            // Test showed score ~0.56. Let's just display it.)

            card.innerHTML = `
                <div class="result-header">
                    <span>ID: <span class="result-id">${item.id}</span></span>
                    <span class="result-score">Dist: ${item.score.toFixed(4)}</span>
                </div>
                <div style="height: 4px; background: #334155; border-radius: 2px; overflow: hidden; margin-top: 10px;">
                    <div style="width: ${Math.max(5, (1 - item.score) * 100)}%; height: 100%; background: #10b981;"></div>
                </div>
            `;
            container.appendChild(card);
        });

    } catch (e) {
        container.innerHTML = `<div class="empty-state" style="color:red">Error: ${e.message}</div>`;
    }
}
