/**
 * Civ6 AlphaZero - WebSocket game client
 *
 * Provides real-time communication with the game server.
 * Handles AI thinking indicators and move results.
 */

class GameClient {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.onStateUpdate = null;
        this.onAiThinking = null;
        this.onAiMove = null;
        this.onGameOver = null;
        this.onError = null;

        this.connect();
    }

    connect() {
        try {
            this.ws = new WebSocket(this.wsUrl);
        } catch (e) {
            console.warn('WebSocket not available, using HTMX fallback');
            return;
        }

        this.ws.onopen = () => {
            this.connected = true;
            this.reconnectAttempts = 0;
            console.log('WebSocket connected');
        };

        this.ws.onclose = () => {
            this.connected = false;
            console.log('WebSocket disconnected');
            this._tryReconnect();
        };

        this.ws.onerror = (err) => {
            console.warn('WebSocket error:', err);
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this._handleMessage(msg);
            } catch (e) {
                console.warn('Failed to parse WebSocket message:', e);
            }
        };
    }

    sendMove(actionIndex) {
        if (!this.connected) return false;
        this.ws.send(JSON.stringify({
            type: 'move',
            action: actionIndex
        }));
        return true;
    }

    sendUndo() {
        if (!this.connected) return false;
        this.ws.send(JSON.stringify({ type: 'undo' }));
        return true;
    }

    sendResign() {
        if (!this.connected) return false;
        this.ws.send(JSON.stringify({ type: 'resign' }));
        return true;
    }

    ping() {
        if (!this.connected) return;
        this.ws.send(JSON.stringify({ type: 'ping' }));
    }

    disconnect() {
        this.maxReconnectAttempts = 0;
        if (this.ws) {
            this.ws.close();
        }
    }

    _handleMessage(msg) {
        switch (msg.type) {
            case 'state':
                if (this.onStateUpdate) this.onStateUpdate(msg.data);
                break;
            case 'ai_thinking':
                this._showAiThinking();
                if (this.onAiThinking) this.onAiThinking();
                break;
            case 'ai_move':
                this._hideAiThinking();
                if (this.onAiMove) this.onAiMove(msg.data);
                break;
            case 'move_result':
                break;
            case 'game_over':
                if (this.onGameOver) this.onGameOver(msg.data);
                break;
            case 'error':
                console.error('Server error:', msg.message);
                if (this.onError) this.onError(msg.message);
                break;
            case 'pong':
                break;
            default:
                console.log('Unknown message type:', msg.type);
        }
    }

    _showAiThinking() {
        const status = document.querySelector('.phase-ai_turn, .info-value');
        if (status && status.textContent.includes('AI')) {
            status.textContent = 'AI Thinking...';
            status.classList.add('thinking');
        }
    }

    _hideAiThinking() {
        const status = document.querySelector('.thinking');
        if (status) {
            status.classList.remove('thinking');
        }
    }

    _tryReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        setTimeout(() => this.connect(), delay);
    }
}
