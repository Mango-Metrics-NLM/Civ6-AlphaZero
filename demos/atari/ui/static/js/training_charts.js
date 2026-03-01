/**
 * Training Dashboard - Live Chart.js training curves with polling.
 */
class TrainingDashboard {
    constructor() {
        this.polling = false;
        this.pollInterval = null;
        this.charts = {};

        this._initCharts();
        this._bindControls();
    }

    _initCharts() {
        const chartConfigs = [
            { id: 'policy-loss-chart', label: 'Policy Loss', color: '#e94560' },
            { id: 'value-loss-chart', label: 'Value Loss', color: '#3498db' },
            { id: 'total-loss-chart', label: 'Total Loss', color: '#2ecc71' },
            { id: 'examples-chart', label: 'Examples', color: '#f39c12' },
        ];

        for (const cfg of chartConfigs) {
            const ctx = document.getElementById(cfg.id);
            if (!ctx) continue;
            this.charts[cfg.id] = new Chart(ctx, {
                type: cfg.id === 'examples-chart' ? 'bar' : 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: cfg.label,
                        data: [],
                        borderColor: cfg.color,
                        backgroundColor: cfg.color + '40',
                        fill: true,
                        tension: 0.3,
                    }]
                },
                options: {
                    responsive: true,
                    animation: false,
                    scales: {
                        y: { beginAtZero: true, ticks: { color: '#8e8e9e' }, grid: { color: '#2a2a4a' } },
                        x: { ticks: { color: '#8e8e9e' }, grid: { display: false } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }
    }

    _bindControls() {
        document.getElementById('btn-start-train')?.addEventListener('click', () => this.startTraining());
        document.getElementById('btn-stop-train')?.addEventListener('click', () => this.stopTraining());
    }

    async startTraining() {
        const iters = parseInt(document.getElementById('train-iters')?.value || '10');
        const episodes = parseInt(document.getElementById('train-episodes')?.value || '2');

        try {
            const resp = await fetch(`/api/training/start?num_iterations=${iters}&episodes_per_iter=${episodes}`, {
                method: 'POST'
            });
            const data = await resp.json();
            console.log('Training started:', data);

            document.getElementById('btn-start-train').disabled = true;
            document.getElementById('btn-stop-train').disabled = false;
            document.getElementById('train-status').textContent = 'Running';

            this._startPolling();
        } catch (err) {
            console.error('Failed to start training:', err);
        }
    }

    async stopTraining() {
        try {
            await fetch('/api/training/stop', { method: 'POST' });
            document.getElementById('train-status').textContent = 'Stopping...';
        } catch (err) {
            console.error('Failed to stop:', err);
        }
    }

    _startPolling() {
        this._stopPolling();
        this.polling = true;
        this.pollInterval = setInterval(() => this._poll(), 2000);
    }

    _stopPolling() {
        this.polling = false;
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async _poll() {
        try {
            const resp = await fetch('/api/training/status');
            const data = await resp.json();

            document.getElementById('train-iteration').textContent = data.iteration;
            document.getElementById('train-policy-loss').textContent =
                data.metrics?.policy_loss?.toFixed(4) || '-';
            document.getElementById('train-value-loss').textContent =
                data.metrics?.value_loss?.toFixed(4) || '-';

            if (!data.running) {
                document.getElementById('train-status').textContent = 'Idle';
                document.getElementById('btn-start-train').disabled = false;
                document.getElementById('btn-stop-train').disabled = true;
                this._stopPolling();
            }

            // Update charts from history
            if (data.history && data.history.length > 0) {
                const labels = data.history.map(h => h.iteration);
                this._updateChart('policy-loss-chart', labels, data.history.map(h => h.policy_loss));
                this._updateChart('value-loss-chart', labels, data.history.map(h => h.value_loss));
                this._updateChart('total-loss-chart', labels, data.history.map(h => h.total_loss));
                this._updateChart('examples-chart', labels, data.history.map(h => h.num_examples));
            }
        } catch (err) {
            console.error('Poll error:', err);
        }
    }

    _updateChart(chartId, labels, data) {
        const chart = this.charts[chartId];
        if (!chart) return;
        chart.data.labels = labels;
        chart.data.datasets[0].data = data;
        chart.update('none');
    }
}
