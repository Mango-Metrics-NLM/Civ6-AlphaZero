/**
 * Episode Replay Controller - play/pause/scrub through saved episodes.
 */
class ReplayController {
    constructor() {
        this.frames = [];
        this.actions = [];
        this.rewards = [];
        this.policies = [];
        this.values = [];
        this.currentFrame = 0;
        this.playing = false;
        this.speed = 1;
        this.playTimer = null;

        this.canvas = document.getElementById('replay-canvas');
        this.ctx = this.canvas?.getContext('2d');

        this.policyChart = null;
        this._initPolicyChart();
        this._bindControls();
    }

    _initPolicyChart() {
        const ctx = document.getElementById('replay-policy-chart');
        if (!ctx) return;
        this.policyChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Policy', data: [], backgroundColor: '#e94560' }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1, ticks: { color: '#8e8e9e' }, grid: { color: '#2a2a4a' } },
                    x: { ticks: { color: '#8e8e9e' }, grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    _bindControls() {
        document.getElementById('replay-file')?.addEventListener('change', (e) => this.loadFile(e));
        document.getElementById('btn-replay-play')?.addEventListener('click', () => this.togglePlay());
        document.getElementById('btn-replay-back')?.addEventListener('click', () => this.stepBack());
        document.getElementById('btn-replay-forward')?.addEventListener('click', () => this.stepForward());
        document.getElementById('btn-replay-start')?.addEventListener('click', () => this.goToStart());
        document.getElementById('btn-replay-end')?.addEventListener('click', () => this.goToEnd());
        document.getElementById('replay-scrubber')?.addEventListener('input', (e) => {
            this.goToFrame(parseInt(e.target.value));
        });
        document.getElementById('replay-speed')?.addEventListener('change', (e) => {
            this.speed = parseFloat(e.target.value);
        });
    }

    async loadFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Read .npz file using JSZip or parse with fetch
        // For simplicity, we'll upload to server and get frames back
        // But since .npz is numpy format, we handle it differently:
        // We'll use a simpler approach - the UI can load via API
        alert('Replay loading from .npz files requires server-side parsing.\nUse the API: POST /api/game/start to create a session and save episodes for replay.');
    }

    loadData(data) {
        this.frames = data.frames || [];
        this.actions = data.actions || [];
        this.rewards = data.rewards || [];
        this.policies = data.policies || [];
        this.values = data.values || [];
        this.currentFrame = 0;

        const scrubber = document.getElementById('replay-scrubber');
        if (scrubber) {
            scrubber.max = Math.max(0, this.frames.length - 1);
            scrubber.value = 0;
        }

        this._displayFrame(0);
    }

    togglePlay() {
        if (this.playing) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        this.playing = true;
        document.getElementById('btn-replay-play').textContent = 'Pause';
        this._scheduleNext();
    }

    pause() {
        this.playing = false;
        document.getElementById('btn-replay-play').textContent = 'Play';
        if (this.playTimer) {
            clearTimeout(this.playTimer);
            this.playTimer = null;
        }
    }

    _scheduleNext() {
        if (!this.playing || this.currentFrame >= this.frames.length - 1) {
            this.pause();
            return;
        }
        const delay = Math.max(16, 100 / this.speed); // ~10fps base
        this.playTimer = setTimeout(() => {
            this.stepForward();
            this._scheduleNext();
        }, delay);
    }

    stepForward() {
        if (this.currentFrame < this.frames.length - 1) {
            this.currentFrame++;
            this._displayFrame(this.currentFrame);
        }
    }

    stepBack() {
        if (this.currentFrame > 0) {
            this.currentFrame--;
            this._displayFrame(this.currentFrame);
        }
    }

    goToStart() { this.goToFrame(0); }
    goToEnd() { this.goToFrame(this.frames.length - 1); }

    goToFrame(idx) {
        this.currentFrame = Math.max(0, Math.min(idx, this.frames.length - 1));
        this._displayFrame(this.currentFrame);
    }

    _displayFrame(idx) {
        // Update scrubber
        const scrubber = document.getElementById('replay-scrubber');
        if (scrubber) scrubber.value = idx;

        document.getElementById('replay-frame-num').textContent =
            `${idx} / ${this.frames.length - 1}`;

        // Frame rendering (base64 frames from API)
        if (this.frames[idx]) {
            const img = new Image();
            img.onload = () => {
                this.ctx.imageSmoothingEnabled = false;
                this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            };
            img.src = `data:image/png;base64,${this.frames[idx]}`;
        }

        // Info
        const actionNames = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'];
        const action = this.actions[idx];
        document.getElementById('replay-action').textContent =
            action !== undefined ? (actionNames[action] || `${action}`) : '-';
        document.getElementById('replay-reward').textContent =
            this.rewards[idx] !== undefined ? this.rewards[idx] : '-';

        // Cumulative reward
        let cumulative = 0;
        for (let i = 0; i <= idx && i < this.rewards.length; i++) {
            cumulative += this.rewards[i] || 0;
        }
        document.getElementById('replay-cumulative').textContent = cumulative;

        // Policy chart
        if (this.policies[idx] && this.policyChart) {
            const probs = this.policies[idx];
            this.policyChart.data.labels = probs.map((_, i) => `A${i}`);
            this.policyChart.data.datasets[0].data = probs;
            this.policyChart.update('none');
        }
    }
}
