// vLLM WebUI - Main JavaScript
class VLLMWebUI {
    constructor() {
        this.ws = null;
        this.chatHistory = [];
        this.serverRunning = false;
        this.autoScroll = true;
        this.benchmarkRunning = false;
        this.benchmarkPollInterval = null;
        
        // Resize state
        this.isResizing = false;
        this.currentResizer = null;
        this.resizeDirection = null;
        
        this.init();
    }

    init() {
        // Get DOM elements
        this.elements = {
            // Configuration
            modelSelect: document.getElementById('model-select'),
            customModel: document.getElementById('custom-model'),
            host: document.getElementById('host'),
            port: document.getElementById('port'),
            tensorParallel: document.getElementById('tensor-parallel'),
            gpuMemory: document.getElementById('gpu-memory'),
            dtype: document.getElementById('dtype'),
            maxModelLen: document.getElementById('max-model-len'),
            trustRemoteCode: document.getElementById('trust-remote-code'),
            enablePrefixCaching: document.getElementById('enable-prefix-caching'),
            disableLogStats: document.getElementById('disable-log-stats'),
            
            // Command Preview
            commandText: document.getElementById('command-text'),
            copyCommandBtn: document.getElementById('copy-command-btn'),
            
            // Buttons
            startBtn: document.getElementById('start-btn'),
            stopBtn: document.getElementById('stop-btn'),
            sendBtn: document.getElementById('send-btn'),
            clearChatBtn: document.getElementById('clear-chat-btn'),
            clearLogsBtn: document.getElementById('clear-logs-btn'),
            
            // Chat
            chatContainer: document.getElementById('chat-container'),
            chatInput: document.getElementById('chat-input'),
            temperature: document.getElementById('temperature'),
            maxTokens: document.getElementById('max-tokens'),
            tempValue: document.getElementById('temp-value'),
            tokensValue: document.getElementById('tokens-value'),
            
            // Logs
            logsContainer: document.getElementById('logs-container'),
            autoScrollCheckbox: document.getElementById('auto-scroll'),
            
            // Status
            statusDot: document.getElementById('status-dot'),
            statusText: document.getElementById('status-text'),
            uptime: document.getElementById('uptime'),
            
            // Benchmark
            runBenchmarkBtn: document.getElementById('run-benchmark-btn'),
            stopBenchmarkBtn: document.getElementById('stop-benchmark-btn'),
            benchmarkRequests: document.getElementById('benchmark-requests'),
            benchmarkRate: document.getElementById('benchmark-rate'),
            benchmarkPromptTokens: document.getElementById('benchmark-prompt-tokens'),
            benchmarkOutputTokens: document.getElementById('benchmark-output-tokens'),
            metricsDisplay: document.getElementById('metrics-display'),
            metricsGrid: document.getElementById('metrics-grid'),
            benchmarkProgress: document.getElementById('benchmark-progress'),
            progressFill: document.getElementById('progress-fill'),
            progressStatus: document.getElementById('progress-status'),
            progressPercent: document.getElementById('progress-percent')
        };

        // Attach event listeners
        this.attachListeners();
        
        // Initialize resize functionality
        this.initResize();
        
        // Update command preview initially
        this.updateCommandPreview();
        
        // Connect WebSocket for logs
        this.connectWebSocket();
        
        // Start status polling
        this.pollStatus();
        setInterval(() => this.pollStatus(), 3000);
    }

    attachListeners() {
        // Server control
        this.elements.startBtn.addEventListener('click', () => this.startServer());
        this.elements.stopBtn.addEventListener('click', () => this.stopServer());
        
        // Chat
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        this.elements.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.sendMessage();
            }
        });
        this.elements.clearChatBtn.addEventListener('click', () => this.clearChat());
        
        // Logs
        this.elements.clearLogsBtn.addEventListener('click', () => this.clearLogs());
        this.elements.autoScrollCheckbox.addEventListener('change', (e) => {
            this.autoScroll = e.target.checked;
        });
        
        // Generation parameters
        this.elements.temperature.addEventListener('input', (e) => {
            this.elements.tempValue.textContent = e.target.value;
        });
        this.elements.maxTokens.addEventListener('input', (e) => {
            this.elements.tokensValue.textContent = e.target.value;
        });
        
        // Command preview - update when any config changes
        const configElements = [
            this.elements.modelSelect,
            this.elements.customModel,
            this.elements.host,
            this.elements.port,
            this.elements.tensorParallel,
            this.elements.gpuMemory,
            this.elements.dtype,
            this.elements.maxModelLen,
            this.elements.trustRemoteCode,
            this.elements.enablePrefixCaching,
            this.elements.disableLogStats
        ];
        
        configElements.forEach(element => {
            element.addEventListener('input', () => this.updateCommandPreview());
            element.addEventListener('change', () => this.updateCommandPreview());
        });
        
        // Copy command button
        this.elements.copyCommandBtn.addEventListener('click', () => this.copyCommand());
        
        // Benchmark
        this.elements.runBenchmarkBtn.addEventListener('click', () => this.runBenchmark());
        this.elements.stopBenchmarkBtn.addEventListener('click', () => this.stopBenchmark());
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.addLog('WebSocket connected', 'success');
            this.updateStatus('connected', 'Connected');
        };
        
        this.ws.onmessage = (event) => {
            if (event.data) {
                this.addLog(event.data);
            }
        };
        
        this.ws.onerror = (error) => {
            this.addLog(`WebSocket error: ${error.message}`, 'error');
        };
        
        this.ws.onclose = () => {
            this.addLog('WebSocket disconnected', 'warning');
            this.updateStatus('disconnected', 'Disconnected');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }

    async pollStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.running) {
                this.serverRunning = true;
                this.updateStatus('running', 'Server Running');
                this.elements.startBtn.disabled = true;
                this.elements.stopBtn.disabled = false;
                this.elements.sendBtn.disabled = false;
                this.elements.runBenchmarkBtn.disabled = false;
                
                if (data.uptime) {
                    this.elements.uptime.textContent = `(${data.uptime})`;
                }
            } else {
                this.serverRunning = false;
                this.updateStatus('connected', 'Server Stopped');
                this.elements.startBtn.disabled = false;
                this.elements.stopBtn.disabled = true;
                this.elements.sendBtn.disabled = true;
                this.elements.runBenchmarkBtn.disabled = true;
                this.elements.uptime.textContent = '';
            }
        } catch (error) {
            console.error('Failed to poll status:', error);
        }
    }

    updateStatus(state, text) {
        this.elements.statusDot.className = `status-dot ${state}`;
        this.elements.statusText.textContent = text;
    }

    getConfig() {
        const model = this.elements.customModel.value.trim() || this.elements.modelSelect.value;
        const maxModelLen = this.elements.maxModelLen.value;
        
        return {
            model: model,
            host: this.elements.host.value,
            port: parseInt(this.elements.port.value),
            tensor_parallel_size: parseInt(this.elements.tensorParallel.value),
            gpu_memory_utilization: parseFloat(this.elements.gpuMemory.value) / 100,
            max_model_len: maxModelLen ? parseInt(maxModelLen) : null,
            dtype: this.elements.dtype.value,
            trust_remote_code: this.elements.trustRemoteCode.checked,
            enable_prefix_caching: this.elements.enablePrefixCaching.checked,
            disable_log_stats: this.elements.disableLogStats.checked,
            load_format: "auto"
        };
    }

    async startServer() {
        const config = this.getConfig();
        
        this.elements.startBtn.disabled = true;
        this.elements.startBtn.textContent = '⏳ Starting...';
        
        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start server');
            }
            
            const data = await response.json();
            this.addLog(`Server started with PID: ${data.pid}`, 'success');
            this.showNotification('Server started successfully', 'success');
            
        } catch (error) {
            this.addLog(`Failed to start server: ${error.message}`, 'error');
            this.showNotification(`Failed to start: ${error.message}`, 'error');
            this.elements.startBtn.disabled = false;
        } finally {
            this.elements.startBtn.textContent = '▶️ Start Server';
        }
    }

    async stopServer() {
        this.elements.stopBtn.disabled = true;
        this.elements.stopBtn.textContent = '⏳ Stopping...';
        
        try {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to stop server');
            }
            
            this.addLog('Server stopped', 'success');
            this.showNotification('Server stopped', 'success');
            
        } catch (error) {
            this.addLog(`Failed to stop server: ${error.message}`, 'error');
            this.showNotification(`Failed to stop: ${error.message}`, 'error');
            this.elements.stopBtn.disabled = false;
        } finally {
            this.elements.stopBtn.textContent = '⏹️ Stop Server';
        }
    }

    async sendMessage() {
        const message = this.elements.chatInput.value.trim();
        
        if (!message) {
            return;
        }
        
        if (!this.serverRunning) {
            this.showNotification('Please start the server first', 'warning');
            return;
        }
        
        // Add user message to chat
        this.addChatMessage('user', message);
        this.chatHistory.push({role: 'user', content: message});
        
        // Clear input
        this.elements.chatInput.value = '';
        
        // Disable send button
        this.elements.sendBtn.disabled = true;
        this.elements.sendBtn.textContent = '⏳ Sending...';
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: this.chatHistory,
                    temperature: parseFloat(this.elements.temperature.value),
                    max_tokens: parseInt(this.elements.maxTokens.value),
                    stream: false
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to send message');
            }
            
            const data = await response.json();
            
            // Extract assistant response
            let assistantMessage = '';
            if (data.choices && data.choices.length > 0) {
                assistantMessage = data.choices[0].message.content;
            } else if (data.response) {
                assistantMessage = data.response;
            } else {
                assistantMessage = 'No response from model';
            }
            
            // Add assistant message to chat
            this.addChatMessage('assistant', assistantMessage);
            this.chatHistory.push({role: 'assistant', content: assistantMessage});
            
        } catch (error) {
            this.addLog(`Chat error: ${error.message}`, 'error');
            this.showNotification(`Error: ${error.message}`, 'error');
            this.addChatMessage('system', `Error: ${error.message}`);
        } finally {
            this.elements.sendBtn.disabled = false;
            this.elements.sendBtn.textContent = 'Send';
        }
    }

    addChatMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (role !== 'system') {
            const roleLabel = document.createElement('strong');
            roleLabel.textContent = role.charAt(0).toUpperCase() + role.slice(1) + ':';
            contentDiv.appendChild(roleLabel);
        }
        
        const text = document.createElement('div');
        text.textContent = content;
        contentDiv.appendChild(text);
        
        messageDiv.appendChild(contentDiv);
        this.elements.chatContainer.appendChild(messageDiv);
        
        // Auto-scroll
        this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
    }

    clearChat() {
        this.chatHistory = [];
        this.elements.chatContainer.innerHTML = `
            <div class="chat-message system">
                <div class="message-content">
                    <strong>System:</strong> Chat cleared. Start a new conversation.
                </div>
            </div>
        `;
    }

    addLog(message, type = 'info') {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        this.elements.logsContainer.appendChild(logEntry);
        
        // Auto-scroll if enabled
        if (this.autoScroll) {
            this.elements.logsContainer.scrollTop = this.elements.logsContainer.scrollHeight;
        }
        
        // Limit log entries to prevent memory issues
        const maxLogs = 1000;
        const logs = this.elements.logsContainer.querySelectorAll('.log-entry');
        if (logs.length > maxLogs) {
            logs[0].remove();
        }
    }

    clearLogs() {
        this.elements.logsContainer.innerHTML = `
            <div class="log-entry info">Logs cleared.</div>
        `;
    }

    showNotification(message, type = 'info') {
        // Simple notification using browser notification API
        // You could also implement a custom toast notification
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Optional: Add a temporary notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#f59e0b'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    updateCommandPreview() {
        const model = this.elements.customModel.value.trim() || this.elements.modelSelect.value;
        const host = this.elements.host.value;
        const port = this.elements.port.value;
        const tensorParallel = this.elements.tensorParallel.value;
        const gpuMemory = parseFloat(this.elements.gpuMemory.value) / 100;
        const dtype = this.elements.dtype.value;
        const maxModelLen = this.elements.maxModelLen.value;
        const trustRemoteCode = this.elements.trustRemoteCode.checked;
        const enablePrefixCaching = this.elements.enablePrefixCaching.checked;
        const disableLogStats = this.elements.disableLogStats.checked;
        
        // Build command string
        let cmd = `python -m vllm.entrypoints.openai.api_server`;
        cmd += ` \\\n  --model ${model}`;
        cmd += ` \\\n  --host ${host}`;
        cmd += ` \\\n  --port ${port}`;
        cmd += ` \\\n  --tensor-parallel-size ${tensorParallel}`;
        cmd += ` \\\n  --gpu-memory-utilization ${gpuMemory}`;
        cmd += ` \\\n  --dtype ${dtype}`;
        cmd += ` \\\n  --load-format auto`;
        
        if (maxModelLen) {
            cmd += ` \\\n  --max-model-len ${maxModelLen}`;
        }
        
        if (trustRemoteCode) {
            cmd += ` \\\n  --trust-remote-code`;
        }
        
        if (enablePrefixCaching) {
            cmd += ` \\\n  --enable-prefix-caching`;
        }
        
        if (disableLogStats) {
            cmd += ` \\\n  --disable-log-stats`;
        }
        
        // Update the display
        this.elements.commandText.textContent = cmd;
    }

    async copyCommand() {
        const commandText = this.elements.commandText.textContent;
        
        try {
            await navigator.clipboard.writeText(commandText);
            
            // Visual feedback
            const originalText = this.elements.copyCommandBtn.textContent;
            this.elements.copyCommandBtn.textContent = '✓ Copied!';
            this.elements.copyCommandBtn.classList.add('copied');
            
            setTimeout(() => {
                this.elements.copyCommandBtn.textContent = originalText;
                this.elements.copyCommandBtn.classList.remove('copied');
            }, 2000);
            
            this.showNotification('Command copied to clipboard!', 'success');
        } catch (err) {
            console.error('Failed to copy command:', err);
            this.showNotification('Failed to copy command', 'error');
        }
    }

    async runBenchmark() {
        if (!this.serverRunning) {
            this.showNotification('Server must be running to benchmark', 'warning');
            return;
        }

        const config = {
            total_requests: parseInt(this.elements.benchmarkRequests.value),
            request_rate: parseFloat(this.elements.benchmarkRate.value),
            prompt_tokens: parseInt(this.elements.benchmarkPromptTokens.value),
            output_tokens: parseInt(this.elements.benchmarkOutputTokens.value)
        };

        this.benchmarkRunning = true;
        this.benchmarkStartTime = Date.now();
        this.elements.runBenchmarkBtn.disabled = true;
        this.elements.runBenchmarkBtn.style.display = 'none';
        this.elements.stopBenchmarkBtn.disabled = false;
        this.elements.stopBenchmarkBtn.style.display = 'inline-block';

        // Hide placeholder, show progress
        this.elements.metricsDisplay.style.display = 'none';
        this.elements.metricsGrid.style.display = 'none';
        this.elements.benchmarkProgress.style.display = 'block';

        try {
            const response = await fetch('/api/benchmark/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start benchmark');
            }

            // Start polling for status
            this.benchmarkPollInterval = setInterval(() => this.pollBenchmarkStatus(), 1000);

        } catch (err) {
            console.error('Failed to start benchmark:', err);
            this.showNotification(`Failed to start benchmark: ${err.message}`, 'error');
            this.resetBenchmarkUI();
        }
    }

    async stopBenchmark() {
        try {
            await fetch('/api/benchmark/stop', {method: 'POST'});
            this.showNotification('Benchmark stopped', 'info');
        } catch (err) {
            console.error('Failed to stop benchmark:', err);
        }
        this.resetBenchmarkUI();
    }

    async pollBenchmarkStatus() {
        try {
            const response = await fetch('/api/benchmark/status');
            const data = await response.json();

            if (data.running) {
                // Update progress (estimate based on time)
                // This is approximate since we don't have real-time progress
                const elapsed = Date.now() - this.benchmarkStartTime;
                const estimated = (this.elements.benchmarkRequests.value / this.elements.benchmarkRate.value) * 1000;
                const progress = Math.min(95, (elapsed / estimated) * 100);
                
                this.elements.progressFill.style.width = `${progress}%`;
                this.elements.progressPercent.textContent = `${progress.toFixed(0)}%`;
            } else {
                // Benchmark complete
                clearInterval(this.benchmarkPollInterval);
                this.benchmarkPollInterval = null;

                if (data.results) {
                    this.displayBenchmarkResults(data.results);
                    this.showNotification('Benchmark completed!', 'success');
                } else {
                    this.showNotification('Benchmark failed', 'error');
                }

                this.resetBenchmarkUI();
            }
        } catch (err) {
            console.error('Failed to poll benchmark status:', err);
        }
    }

    displayBenchmarkResults(results) {
        // Hide progress, show metrics
        this.elements.benchmarkProgress.style.display = 'none';
        this.elements.metricsGrid.style.display = 'grid';

        // Update metric cards
        document.getElementById('metric-throughput').textContent = `${results.throughput} req/s`;
        document.getElementById('metric-latency').textContent = `${results.avg_latency} ms`;
        document.getElementById('metric-tokens-per-sec').textContent = `${results.tokens_per_second} tok/s`;
        document.getElementById('metric-p50').textContent = `${results.p50_latency} ms`;
        document.getElementById('metric-p95').textContent = `${results.p95_latency} ms`;
        document.getElementById('metric-p99').textContent = `${results.p99_latency} ms`;
        document.getElementById('metric-total-tokens').textContent = results.total_tokens.toLocaleString();
        document.getElementById('metric-success-rate').textContent = `${results.success_rate} %`;

        // Animate cards
        document.querySelectorAll('.metric-card').forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('updated');
                setTimeout(() => card.classList.remove('updated'), 500);
            }, index * 50);
        });
    }

    resetBenchmarkUI() {
        this.benchmarkRunning = false;
        this.elements.runBenchmarkBtn.disabled = !this.serverRunning;
        this.elements.runBenchmarkBtn.style.display = 'inline-block';
        this.elements.stopBenchmarkBtn.disabled = true;
        this.elements.stopBenchmarkBtn.style.display = 'none';
        this.elements.progressFill.style.width = '0%';
        this.elements.progressPercent.textContent = '0%';
        
        if (this.benchmarkPollInterval) {
            clearInterval(this.benchmarkPollInterval);
            this.benchmarkPollInterval = null;
        }
    }

    // ============ Resize Functionality ============
    initResize() {
        const resizeHandles = document.querySelectorAll('.resize-handle');
        
        resizeHandles.forEach(handle => {
            handle.addEventListener('mousedown', (e) => this.startResize(e, handle));
        });
        
        document.addEventListener('mousemove', (e) => this.resize(e));
        document.addEventListener('mouseup', () => this.stopResize());
    }

    startResize(e, handle) {
        e.preventDefault();
        this.isResizing = true;
        this.currentResizer = handle;
        this.resizeDirection = handle.dataset.direction;
        
        // Add resizing class to body
        document.body.classList.add(
            this.resizeDirection === 'horizontal' ? 'resizing' : 'resizing-vertical'
        );
        
        // Store initial positions
        this.startX = e.clientX;
        this.startY = e.clientY;
        
        // Get the panel being resized
        if (this.resizeDirection === 'horizontal') {
            // Find which resizable section this handle belongs to
            const parentResizable = handle.closest('.resizable');
            
            // Determine which panel to resize based on the parent's ID
            if (parentResizable.id === 'config-panel') {
                // Left handle: resize config panel (normal direction)
                this.resizingPanel = parentResizable;
                this.resizeMode = 'left';
            } else if (parentResizable.id === 'chat-panel') {
                // Right handle: resize logs panel (need to find it)
                this.resizingPanel = document.getElementById('logs-panel');
                this.resizeMode = 'right';
            }
            
            this.startWidth = this.resizingPanel.offsetWidth;
        } else {
            // Vertical resize: metrics section at the bottom
            // The handle is between main-content and metrics-section
            // We always resize the metrics panel
            this.resizingPanel = document.getElementById('metrics-panel');
            this.resizeMode = 'bottom';
            this.startHeight = this.resizingPanel.offsetHeight;
        }
    }

    resize(e) {
        if (!this.isResizing) return;
        
        e.preventDefault();
        
        if (this.resizeDirection === 'horizontal') {
            // Horizontal resize (columns)
            const deltaX = e.clientX - this.startX;
            let newWidth;
            
            // For the right panel (logs), we resize in reverse direction
            if (this.resizeMode === 'right') {
                newWidth = this.startWidth - deltaX; // Dragging left makes logs bigger
            } else {
                newWidth = this.startWidth + deltaX; // Dragging right makes config bigger
            }
            
            // Apply minimum width
            if (newWidth >= 200) {
                this.resizingPanel.style.width = `${newWidth}px`;
                this.resizingPanel.style.flexShrink = '0';
                
                // Ensure the chat section remains flexible
                const chatSection = document.querySelector('.chat-section');
                chatSection.style.flex = '1';
                chatSection.style.width = 'auto';
                chatSection.style.minWidth = '200px';
                
                // Force layout recalculation for better responsiveness
                this.resizingPanel.offsetWidth;
            }
        } else {
            // Vertical resize (bottom metrics section)
            const deltaY = e.clientY - this.startY;
            const newHeight = this.startHeight + deltaY; // Dragging down makes metrics bigger
            
            // Apply minimum height
            if (newHeight >= 200) {
                // Set height on both the outer section and inner panel
                this.resizingPanel.style.height = `${newHeight}px`;
                
                const metricsInnerPanel = this.resizingPanel.querySelector('.panel');
                if (metricsInnerPanel) {
                    metricsInnerPanel.style.height = `${newHeight}px`;
                }
                
                // Force layout recalculation
                this.resizingPanel.offsetHeight;
            }
        }
    }

    stopResize() {
        if (!this.isResizing) return;
        
        this.isResizing = false;
        this.currentResizer = null;
        
        // Remove resizing class
        document.body.classList.remove('resizing', 'resizing-vertical');
        
        // Save layout preferences to localStorage
        this.saveLayoutPreferences();
    }

    saveLayoutPreferences() {
        const layout = {
            configWidth: document.getElementById('config-panel')?.offsetWidth,
            logsWidth: document.getElementById('logs-panel')?.offsetWidth,
            metricsHeight: document.querySelector('.metrics-section .panel')?.offsetHeight
        };
        
        try {
            localStorage.setItem('vllm-webui-layout', JSON.stringify(layout));
        } catch (e) {
            console.warn('Could not save layout preferences:', e);
        }
    }

    loadLayoutPreferences() {
        try {
            const saved = localStorage.getItem('vllm-webui-layout');
            if (saved) {
                const layout = JSON.parse(saved);
                
                if (layout.configWidth) {
                    const configPanel = document.getElementById('config-panel');
                    if (configPanel) configPanel.style.width = `${layout.configWidth}px`;
                }
                
                if (layout.logsWidth) {
                    const logsPanel = document.getElementById('logs-panel');
                    if (logsPanel) logsPanel.style.width = `${layout.logsWidth}px`;
                }
                
                if (layout.metricsHeight) {
                    const metricsPanel = document.querySelector('.metrics-section .panel');
                    if (metricsPanel) metricsPanel.style.height = `${layout.metricsHeight}px`;
                }
            }
        } catch (e) {
            console.warn('Could not load layout preferences:', e);
        }
    }
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.vllmUI = new VLLMWebUI();
    
    // Load saved layout preferences
    window.vllmUI.loadLayoutPreferences();
});

