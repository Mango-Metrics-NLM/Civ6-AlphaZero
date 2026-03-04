/**
 * Atari Game - HTML5 Canvas game rendering + keyboard input + WebSocket.
 */
class AtariGame {
    constructor(gameName) {
        this.gameName = gameName;
        this.sessionId = null;
        this.ws = null;
        this.mode = 'human'; // 'human', 'ai', 'step'
        this.running = false;
        this.aiInterval = null;

        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('game-overlay');
        this.overlayText = document.getElementById('overlay-text');

        // Policy chart
        this.policyChart = null;
        this._initPolicyChart();

        this._bindControls();
        this._bindKeyboard();
    }

    _initPolicyChart() {
        const ctx = document.getElementById('policy-chart');
        if (!ctx) return;
        this.policyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Action Probability',
                    data: [],
                    backgroundColor: '#e94560',
                }]
            },
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
        document.getElementById('btn-new-game')?.addEventListener('click', () => this.startGame());
        document.getElementById('btn-reset')?.addEventListener('click', () => this.resetGame());
        document.getElementById('btn-save')?.addEventListener('click', () => this.saveEpisode());
        document.getElementById('btn-ai-step')?.addEventListener('click', () => this.requestAiStep());

        document.querySelectorAll('input[name="play-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => this.setMode(e.target.value));
        });
    }

    _bindKeyboard() {
        // Atari action mapping: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
        const keyMap = {
            'ArrowRight': 2,
            'ArrowLeft': 3,
            ' ': 1,       // Space = FIRE
            'ArrowUp': 1,  // Up = FIRE
        };

        document.addEventListener('keydown', (e) => {
            if (!this.running || this.mode !== 'human') return;
            const action = keyMap[e.key];
            if (action !== undefined) {
                e.preventDefault();
                this.sendAction(action);
            }
            if (e.key === 'Enter') {
                e.preventDefault();
                this.resetGame();
            }
        });
    }

    async startGame() {
        try {
            const resp = await fetch('/api/game/start', { method: 'POST' });
            const data = await resp.json();
            this.sessionId = data.session_id;
            this.running = true;
            this.overlay.classList.add('hidden');

            // Render initial frame
            if (data.frame) this._renderFrame(data.frame);
            this._updateInfo(data);

            // Connect WebSocket
            this._connectWebSocket();
            this._setConnectionStatus(true);

            document.getElementById('mode-display').textContent =
                this.mode === 'human' ? 'Human' : this.mode === 'ai' ? 'AI Autopilot' : 'Step AI';

            if (this.mode === 'ai') this._startAiLoop();
        } catch (err) {
            console.error('Failed to start game:', err);
        }
    }

    _connectWebSocket() {
        if (this.ws) this.ws.close();

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws/game/${this.sessionId}`);

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            this._handleMessage(msg);
        };

        this.ws.onclose = () => {
            this._setConnectionStatus(false);
        };
    }

    _handleMessage(msg) {
        switch (msg.type) {
            case 'state':
                if (msg.data?.frame) this._renderFrame(msg.data.frame);
                this._updateInfo(msg.data);
                break;
            case 'frame':
                if (msg.data?.frame) this._renderFrame(msg.data.frame);
                this._updateInfo(msg.data);
                break;
            case 'ai_move':
                if (msg.frame) this._renderFrame(msg.frame);
                this._updateAiAnalysis(msg);
                this._updateInfo(msg);
                break;
            case 'ai_thinking':
                document.getElementById('ai-action-name').textContent = 'Thinking...';
                break;
            case 'episode_end':
                this.running = false;
                this._stopAiLoop();
                this.overlayText.textContent = `Episode End! Score: ${msg.total_reward}`;
                this.overlay.classList.remove('hidden');
                break;
            case 'error':
                console.error('Server error:', msg.message);
                break;
        }
    }

    sendAction(action) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'action', action }));
        }
    }

    requestAiStep() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'ai_move' }));
        }
    }

    resetGame() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'reset' }));
            this.running = true;
            this.overlay.classList.add('hidden');
            if (this.mode === 'ai') this._startAiLoop();
        }
    }

    async saveEpisode() {
        if (!this.sessionId) return;
        try {
            const resp = await fetch(`/api/game/save_episode?session_id=${this.sessionId}`, { method: 'POST' });
            const data = await resp.json();
            alert(`Episode saved to: ${data.path}`);
        } catch (err) {
            console.error('Failed to save:', err);
        }
    }

    setMode(mode) {
        this.mode = mode;
        document.getElementById('mode-display').textContent =
            mode === 'human' ? 'Human' : mode === 'ai' ? 'AI Autopilot' : 'Step AI';

        const stepControls = document.getElementById('step-controls');
        if (stepControls) {
            stepControls.classList.toggle('hidden', mode !== 'step');
        }

        if (mode === 'ai' && this.running) {
            this._startAiLoop();
        } else {
            this._stopAiLoop();
        }
    }

    _startAiLoop() {
        this._stopAiLoop();
        this.aiInterval = setInterval(() => {
            if (this.running && this.ws?.readyState === WebSocket.OPEN) {
                this.requestAiStep();
            }
        }, 200); // ~5 fps AI play
    }

    _stopAiLoop() {
        if (this.aiInterval) {
            clearInterval(this.aiInterval);
            this.aiInterval = null;
        }
    }

    _renderFrame(base64Data) {
        if (!base64Data) return;
        const img = new Image();
        img.onload = () => {
            this.ctx.imageSmoothingEnabled = false;
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
        };
        img.src = `data:image/png;base64,${base64Data}`;
    }

    _updateInfo(data) {
        if (data?.total_reward !== undefined) {
            document.getElementById('score-display').textContent = data.total_reward;
        }
        if (data?.step !== undefined) {
            document.getElementById('step-display').textContent = data.step;
        }
    }

    _updateAiAnalysis(data) {
        if (data.value !== undefined) {
            document.getElementById('value-estimate').textContent = data.value.toFixed(3);
        }
        if (data.action !== undefined) {
            const actionNames = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'];
            document.getElementById('ai-action-name').textContent =
                actionNames[data.action] || `Action ${data.action}`;
        }
        if (data.probs && this.policyChart) {
            const labels = data.probs.map((_, i) => `A${i}`);
            this.policyChart.data.labels = labels;
            this.policyChart.data.datasets[0].data = data.probs;
            this.policyChart.update('none');
        }
    }

    _setConnectionStatus(connected) {
        const dot = document.getElementById('connection-status');
        if (dot) {
            dot.classList.toggle('connected', connected);
            dot.classList.toggle('disconnected', !connected);
        }
    }
}
