/**
 * app.js
 * Main Dashboard Coordinator for AERO.
 * Manages WebSocket connection, UI events, simulation controls, and telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Subsystems
  const simRenderer = new window.SimCanvasRenderer('sim-canvas');
  const telemetry = new window.TelemetryManager();
  const codeDiff = new window.CodeDiffRenderer('code-container');

  // DOM Elements
  const promptInput = document.getElementById('problem-prompt');
  const modelSelect = document.getElementById('model-select');
  const btnRunLoop = document.getElementById('btn-run-loop');
  const btnRunSingle = document.getElementById('btn-run-single');
  const btnStop = document.getElementById('btn-stop');
  const btnToggleDiff = document.getElementById('btn-toggle-diff');
  const btnResetCam = document.getElementById('btn-reset-cam');
  const btnClearLogs = document.getElementById('btn-clear-logs');
  const presetChips = document.querySelectorAll('.preset-chip');

  // Presets mapping for Cognitive Explorer
  const presets = {
    'find_sofa': 'Find the red sofa and navigate to it',
    'explore_house': 'Explore unmapped rooms and chart the house',
    'kitchen': 'Navigate to the kitchen table',
    'dock': 'Return to charging dock',
  };

  presetChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const key = chip.dataset.preset;
      if (presets[key]) {
        promptInput.value = presets[key];
        telemetry.log(`Selected preset: ${chip.textContent}`, 'info');
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
      telemetry.setStatus('IDLE');
      telemetry.log('⚡ Connected to AERO Live Telemetry Stream', 'success');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'init':
            if (data.problem_statement) promptInput.value = data.problem_statement;
            if (data.selected_model) modelSelect.value = data.selected_model;
            telemetry.setIteration(data.iteration || 1, data.max_iterations || 5);
            codeDiff.updateCode(data.current_code, data.previous_code);
            if (data.sim_frame) {
              simRenderer.updateFrame(data.sim_frame);
              telemetry.updateMetrics(data.sim_frame);
            }
            break;

          case 'sim_frame':
            simRenderer.updateFrame(data);
            telemetry.updateMetrics(data);
            break;

          case 'status_update':
            telemetry.setStatus(data.status);
            telemetry.setIteration(data.iteration, data.max_iterations);
            updateControlButtons(data.status === 'RUNNING');
            break;

          case 'code_update':
            codeDiff.updateCode(data.current_code, data.previous_code);
            telemetry.log(`[Iteration ${data.iteration}] Controller code updated`, 'info');
            break;

          case 'log':
            telemetry.log(data.message, data.level || 'info');
            break;

          case 'trial_end':
            telemetry.setStatus(data.verdict, data.reason);
            updateControlButtons(false);
            if (data.verdict === 'PASSED') {
              telemetry.log(
                `🎉 BENCHMARK PASSED! Final distance: ${data.final_distance.toFixed(3)}m in ${data.time_elapsed.toFixed(1)}s`,
                'success'
              );
            } else {
              telemetry.log(`❌ Trial failed: ${data.reason}`, 'error');
            }
            break;
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onclose = () => {
      telemetry.setStatus('DISCONNECTED');
      telemetry.log('WebSocket disconnected. Reconnecting in 2s...', 'warning');
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error('WS Error:', err);
    };
  }

  connectWebSocket();

  // Control Handlers
  async function triggerRun(mode) {
    const payload = {
      problem_statement: promptInput.value.trim(),
      model: modelSelect.value,
      max_retries: 5,
      mode: mode,
    };

    simRenderer.resetTrajectory();
    updateControlButtons(true);

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        telemetry.log(`Run error: ${err.detail || 'Failed to start'}`, 'error');
        updateControlButtons(false);
      }
    } catch (e) {
      telemetry.log(`Network error: ${e.message}`, 'error');
      updateControlButtons(false);
    }
  }

  btnRunLoop.addEventListener('click', () => triggerRun('loop'));
  btnRunSingle.addEventListener('click', () => triggerRun('single'));

  btnStop.addEventListener('click', async () => {
    try {
      await fetch('/api/stop', { method: 'POST' });
      updateControlButtons(false);
    } catch (e) {
      console.error('Stop error:', e);
    }
  });

  function updateControlButtons(isRunning) {
    btnRunLoop.disabled = isRunning;
    btnRunSingle.disabled = isRunning;
    btnStop.disabled = !isRunning;
    btnRunLoop.style.opacity = isRunning ? '0.5' : '1';
    btnRunSingle.style.opacity = isRunning ? '0.5' : '1';
  }

  // Code Diff Toggle View
  btnToggleDiff.addEventListener('click', () => {
    const newMode = codeDiff.toggleViewMode();
    btnToggleDiff.textContent = newMode === 'diff' ? 'Diff View' : 'Full View';
  });

  // Reset Camera
  btnResetCam.addEventListener('click', () => {
    simRenderer.zoom = 1.0;
    simRenderer.panX = 0;
    simRenderer.panY = 0;
    simRenderer.resize();
  });

  // Clear Logs
  btnClearLogs.addEventListener('click', () => telemetry.clearLogs());
});
