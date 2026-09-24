/**
 * app.js
 * Cognitive Coordinator for AERO Idea B.
 * Handles Natural Language Chat with Gemma Brain, real-time WebSocket frames,
 * multi-room canvas updates, and semantic spatial memory telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Subsystems
  const simRenderer = new window.SimCanvas3DRenderer('sim-3d-container');
  const telemetry = new window.TelemetryManager();

  // DOM Elements
  const chatInput = document.getElementById('chat-input');
  const btnSendGemma = document.getElementById('btn-send-gemma');
  const btnStopNav = document.getElementById('btn-stop-nav');
  const btnCamOrbit = document.getElementById('btn-cam-orbit');
  const btnCamTop = document.getElementById('btn-cam-top');
  const btnCamFollow = document.getElementById('btn-cam-follow');
  const btnResetCam = document.getElementById('btn-reset-cam');
  const btnClearLogs = document.getElementById('btn-clear-logs');
  const thoughtText = document.getElementById('gemma-thought-text');
  const activeGoalText = document.getElementById('gemma-active-goal');
  const landmarksList = document.getElementById('landmarks-list');
  const presetChips = document.querySelectorAll('.preset-chip');

  // Preset chips handler
  presetChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const cmd = chip.dataset.cmd;
      if (cmd) {
        chatInput.value = cmd;
        sendChatCommand(cmd);
      }
    });
  });

  // WebSocket Connection
  let ws = null;
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      telemetry.setStatus('ONLINE');
      telemetry.log('⚡ Connected to AERO Cognitive Telemetry Stream', 'success');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'sim_frame') {
          simRenderer.updateFrame(data);
          telemetry.updateMetrics(data);

          if (data.gemma_thought && thoughtText) {
            thoughtText.textContent = data.gemma_thought;
          }
          if (data.target && activeGoalText) {
            activeGoalText.textContent = `Target: ${data.target.name || 'Goal'} at (${data.target.x.toFixed(2)}, ${data.target.y.toFixed(2)})`;
          }

          // Update landmarks list if present
          if (data.semantic_objects && landmarksList) {
            updateLandmarksUI(data.semantic_objects);
          }
        } else if (data.type === 'log') {
          telemetry.log(data.message, data.level || 'info');
        } else if (data.type === 'status_update') {
          telemetry.setStatus(data.status, data.reason);
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onclose = () => {
      telemetry.setStatus('OFFLINE');
      telemetry.log('WebSocket disconnected. Reconnecting in 2s...', 'warning');
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => console.error('WS Error:', err);
  }

  connectWebSocket();

  // Send Natural Language Command to Gemma Brain
  async function sendChatCommand(messageText) {
    const text = messageText || chatInput.value.trim();
    if (!text) return;

    btnSendGemma.disabled = true;
    btnSendGemma.style.opacity = '0.6';

    try {
      telemetry.log(`🗣️ Command sent to Gemma: "${text}"`, 'info');
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (res.ok) {
        const decision = await res.json();
        if (thoughtText) thoughtText.textContent = decision.thought;
        if (activeGoalText && decision.coordinates) {
          activeGoalText.textContent = `Target: ${decision.target} at (${decision.coordinates[0]}, ${decision.coordinates[1]})`;
        }
      } else {
        const err = await res.json();
        telemetry.log(`Gemma Error: ${err.detail || 'Failed to process command'}`, 'error');
      }
    } catch (e) {
      telemetry.log(`Network error: ${e.message}`, 'error');
    } finally {
      btnSendGemma.disabled = false;
      btnSendGemma.style.opacity = '1';
    }
  }

  btnSendGemma.addEventListener('click', () => sendChatCommand());
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatCommand();
    }
  });

  // Stop button
  btnStopNav.addEventListener('click', async () => {
    try {
      await fetch('/api/stop', { method: 'POST' });
      telemetry.setStatus('STOPPED');
    } catch (e) {
      console.error('Stop error:', e);
    }
  });

  // Camera View Switcher
  function setCameraActive(activeBtn, mode) {
    [btnCamOrbit, btnCamTop, btnCamFollow].forEach((b) => {
      if (b) {
        b.style.background = 'transparent';
        b.style.color = 'var(--text-secondary)';
        b.style.fontWeight = 'normal';
      }
    });
    if (activeBtn) {
      activeBtn.style.background = 'rgba(0, 229, 255, 0.18)';
      activeBtn.style.color = '#00e5ff';
      activeBtn.style.fontWeight = '600';
    }
    simRenderer.setCameraView(mode);
  }

  if (btnCamOrbit) btnCamOrbit.addEventListener('click', () => setCameraActive(btnCamOrbit, 'orbit'));
  if (btnCamTop) btnCamTop.addEventListener('click', () => setCameraActive(btnCamTop, 'top'));
  if (btnCamFollow) btnCamFollow.addEventListener('click', () => setCameraActive(btnCamFollow, 'follow'));
  if (btnResetCam) btnResetCam.addEventListener('click', () => setCameraActive(btnCamOrbit, 'orbit'));

  // Clear Logs
  btnClearLogs.addEventListener('click', () => telemetry.clearLogs());

  // Render Landmarks in UI
  function updateLandmarksUI(objects) {
    landmarksList.innerHTML = '';
    objects.forEach((obj) => {
      const row = document.createElement('div');
      row.style.cssText =
        'display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem; padding: 6px 10px; background: rgba(255,255,255,0.03); border-radius: 6px; cursor: pointer;';

      const left = document.createElement('span');
      left.textContent = `${obj.icon || '📍'} ${obj.name}`;

      const right = document.createElement('span');
      right.style.fontFamily = 'var(--font-mono)';
      right.style.color = obj.discovered ? 'var(--accent-cyan)' : 'var(--text-muted)';
      right.textContent = obj.discovered
        ? `(${obj.x.toFixed(1)}, ${obj.y.toFixed(1)}) ${obj.room}`
        : '○ UNCHARTED';

      row.appendChild(left);
      row.appendChild(right);

      // Clicking landmark dispatches navigation to it
      row.addEventListener('click', () => {
        chatInput.value = `Navigate to the ${obj.name}`;
        sendChatCommand(`Navigate to the ${obj.name}`);
      });

      landmarksList.appendChild(row);
    });
  }
});
