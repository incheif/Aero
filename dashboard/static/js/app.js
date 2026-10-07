/**
 * Main Web Dashboard Controller & WebSocket Client
 * Google VLA Robot Arm Autonomous Manipulation Studio
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize 3D Viewport
  const canvasEl = document.getElementById('canvas-3d');
  const simCanvas = new window.SimCanvas3D(canvasEl);

  // 2. WebSocket Telemetry Client
  let ws = null;
  let isConnected = false;

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      isConnected = true;
      updateConnectionStatus(true);
      console.log('[WebSocket] Connected to simulation server.');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'telemetry') {
          handleTelemetry(data);
        }
      } catch (err) {
        console.error('Error parsing telemetry JSON:', err);
      }
    };

    ws.onclose = () => {
      isConnected = false;
      updateConnectionStatus(false);
      console.warn('[WebSocket] Disconnected. Reconnecting in 1.5s...');
      setTimeout(connectWebSocket, 1500);
    };

    ws.onerror = (err) => {
      console.error('[WebSocket] Error:', err);
      ws.close();
    };
  }

  connectWebSocket();

  // 3. UI State Handlers
  function updateConnectionStatus(connected) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (dot) {
      if (connected) {
        dot.style.backgroundColor = '#ffffff';
        dot.style.boxShadow = '0 0 8px rgba(255, 255, 255, 0.8)';
      } else {
        dot.style.backgroundColor = '#52525b';
        dot.style.boxShadow = 'none';
      }
    }
    if (text) {
      text.innerText = connected ? 'ONLINE / 60 FPS' : 'RECONNECTING...';
    }
  }

  function handleTelemetry(data) {
    // 1. Pass to 3D Three.js Canvas
    simCanvas.updateFromTelemetry(data);

    // 2. Update Header FPS & Ticks
    const fpsEl = document.getElementById('fps-metric');
    if (fpsEl) fpsEl.innerText = `${data.fps || 60} FPS`;

    // 3. Update VLA Thought Stream Card
    const thoughtBox = document.getElementById('vla-thought-box');
    const vlaBadge = document.getElementById('vla-status-badge');
    const modelPill = document.getElementById('model-pill-text');

    if (data.vla) {
      if (thoughtBox) thoughtBox.innerText = data.vla.thought || 'Awaiting instruction...';
      if (vlaBadge) {
        vlaBadge.innerText = data.vla.status || 'IDLE';
        if (data.vla.status === 'EXECUTING') {
          vlaBadge.style.color = '#09090b';
          vlaBadge.style.backgroundColor = '#ffffff';
          vlaBadge.style.borderColor = '#ffffff';
        } else {
          vlaBadge.style.color = '#ffffff';
          vlaBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.08)';
          vlaBadge.style.borderColor = 'rgba(255, 255, 255, 0.28)';
        }
      }
      if (modelPill && data.vla.model) {
        modelPill.innerText = `Google VLA (${data.vla.model})`;
      }

      // Update VLA Mode Badge (CLOUD vs EMBEDDED)
      const modeBadge = document.getElementById('vla-mode-badge');
      if (modeBadge) {
        if (data.vla.mode === 'live_cloud') {
          modeBadge.innerText = 'LIVE CLOUD';
          modeBadge.style.color = '#ffffff';
          modeBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.22)';
          modeBadge.title = 'Connected to Google Gemini Cloud Multimodal VLA';
        } else if (data.vla.has_api_key) {
          modeBadge.innerText = 'KEY READY';
          modeBadge.style.color = '#ffffff';
          modeBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.12)';
          modeBadge.title = 'API key saved. Ready for cloud inference.';
        } else {
          modeBadge.innerText = 'EMBEDDED';
          modeBadge.style.color = '#a1a1aa';
          modeBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
          modeBadge.title = 'Running Embedded Deterministic VLA Runtime (No API Key)';
        }
      }

      // Progress bar
      const prog = data.vla.progress;
      if (prog) {
        const barFill = document.getElementById('vla-progress-fill');
        const progText = document.getElementById('vla-progress-text');
        if (barFill) barFill.style.width = `${prog.pct || 0}%`;
        if (progText) progText.innerText = prog.is_active ? `Step ${prog.step}/${prog.total_steps} (${prog.action})` : 'Idle';
      }
    }

    // 4. Update 3D Semantic Spatial Memory Table
    if (data.semantic_map && data.semantic_map.entities) {
      updateSemanticTable(data.semantic_map.entities);
    }

    // 5. Update Frontier Exploration Card
    if (data.frontier) {
      const pctEl = document.getElementById('frontier-pct');
      const barEl = document.getElementById('frontier-bar-fill');
      if (pctEl) pctEl.innerText = `${data.frontier.exploration_pct || 25}%`;
      if (barEl) barEl.style.width = `${data.frontier.exploration_pct || 25}%`;

      const qCoverage = data.frontier.quadrant_coverage || {};
      ['nw', 'ne', 'sw', 'se'].forEach((q) => {
        const cell = document.getElementById(`quad-${q}`);
        if (cell) cell.innerText = `${qCoverage[`quadrant_${q}`] || 25}%`;
      });
    }

    // 6. Update Joint Telemetry Gauges
    if (data.joints) {
      const j = data.joints;
      setVal('joint-yaw', `${(j.baseYaw * 57.2958).toFixed(1)}° (${j.baseYaw.toFixed(2)})`);
      setVal('joint-shoulder', `${(j.shoulderPitch * 57.2958).toFixed(1)}° (${j.shoulderPitch.toFixed(2)})`);
      setVal('joint-elbow', `${(j.elbowPitch * 57.2958).toFixed(1)}° (${j.elbowPitch.toFixed(2)})`);
      setVal('joint-wrist', `${(j.wristPitch * 57.2958).toFixed(1)}° (${j.wristPitch.toFixed(2)})`);
      setVal('joint-gripper', `${((1.0 - j.gripper) * 100).toFixed(0)}% Open`);
    }

    if (data.tcp && data.tcp.position) {
      const tcp = data.tcp.position;
      setVal('tcp-coords', `[${tcp[0].toFixed(3)}, ${tcp[1].toFixed(3)}, ${tcp[2].toFixed(3)}]`);
      setVal('dock-tcp-pos', `X: ${tcp[0].toFixed(2)} · Y: ${tcp[1].toFixed(2)} · Z: ${tcp[2].toFixed(2)}`);
    }

    if (data.joints && data.joints.gripper !== undefined) {
      const gripPct = ((1.0 - data.joints.gripper) * 100).toFixed(0);
      const gripState = data.joints.gripper > 0.6 ? 'CLOSED' : (data.joints.gripper < 0.35 ? 'OPEN' : 'HOLDING');
      setVal('dock-gripper-state', `${gripPct}% (${gripState})`);
    }

    if (data.fps !== undefined) {
      setVal('dock-fps-val', `${data.fps} FPS`);
    }

    if (data.vla && data.vla.status) {
      setVal('dock-motion-status', data.vla.status);
    }

    // 7. Update LLM VLA Cost Tracking Metrics
    if (data.cost) {
      updateCostMetrics(data.cost);
    }

    // 8. Update Pause Button Icon
    const pauseBtn = document.getElementById('btn-pause');
    if (pauseBtn) {
      pauseBtn.innerText = data.is_paused ? '▶ Resume' : '⏸ Pause';
    }
  }

  function updateCostMetrics(cost) {
    if (!cost) return;

    // Header Pill
    const headerCost = document.getElementById('header-cost-val');
    if (headerCost) headerCost.innerText = cost.formatted_cost || `$${(cost.total_cost_usd || 0).toFixed(5)}`;

    // Main Card Hero
    const totalDisplay = document.getElementById('cost-total-display');
    if (totalDisplay) totalDisplay.innerText = cost.formatted_cost || `$${(cost.total_cost_usd || 0).toFixed(5)}`;

    // Token & Call Metrics
    setVal('input-tokens-display', (cost.total_input_tokens || 0).toLocaleString());
    setVal('output-tokens-display', (cost.total_output_tokens || 0).toLocaleString());
    setVal('calls-count-display', `${cost.total_calls || 0} calls (${(cost.total_tokens || 0).toLocaleString()} tok)`);

    // Last Call Cost & Model
    if (cost.last_call) {
      const lastCost = cost.last_call.formatted_cost || `$${(cost.last_call.cost_usd || 0).toFixed(5)}`;
      setVal('last-call-cost-display', lastCost);
      const modelBadge = document.getElementById('cost-model-badge');
      if (modelBadge && cost.last_call.model) {
        modelBadge.innerText = cost.last_call.model;
      }
    }

    // Average Cost per call
    const avgLabel = document.getElementById('avg-cost-label');
    if (avgLabel) {
      avgLabel.innerText = `Avg: $${(cost.avg_cost_per_call_usd || 0).toFixed(5)}/call`;
    }

    // Recent Inference History Feed
    const historyList = document.getElementById('cost-history-list');
    if (historyList && cost.history) {
      if (cost.history.length === 0) {
        historyList.innerHTML = '<div class="cost-history-empty">No inferences yet this session</div>';
      } else {
        historyList.innerHTML = cost.history.slice().reverse().map(h => `
          <div class="cost-history-item">
            <span class="cost-history-prompt" title="${h.prompt_snippet}">[${h.timestamp}] ${h.prompt_snippet}</span>
            <span class="cost-history-cost">+$${h.cost_usd.toFixed(5)} (${h.total_tokens}t)</span>
          </div>
        `).join('');
      }
    }
  }

  function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.innerText = val;
  }

  function updateSemanticTable(entities) {
    const tbody = document.getElementById('semantic-table-body');
    if (!tbody) return;

    let html = '';
    Object.values(entities).forEach((ent) => {
      if (ent.category === 'tcp') return; // Skip TCP in list for brevity
      const pos = ent.position;
      html += `
        <tr>
          <td><span class="color-dot" style="background-color: ${ent.color};"></span>${ent.name}</td>
          <td>${ent.semantic_state}</td>
          <td>[${pos[0].toFixed(2)}, ${pos[2].toFixed(2)}]</td>
          <td><span style="color: ${ent.grasp_affordance > 0.5 ? '#ffffff' : '#71717a'}">${ent.grasp_affordance}</span></td>
        </tr>
      `;
    });
    tbody.innerHTML = html;
  }

  // 4. Command Input Handling
  const cmdInput = document.getElementById('command-input');
  const sendBtn = document.getElementById('send-command-btn');

  async function submitCommand(commandText) {
    const text = (commandText || cmdInput.value || '').trim();
    if (!text) return;

    const apiKey = localStorage.getItem('GEMINI_API_KEY') || '';
    const model = localStorage.getItem('GEMINI_MODEL') || 'gemini-2.5-flash';

    sendBtn.disabled = true;
    sendBtn.innerText = 'Reasoning...';

    try {
      const resp = await fetch('/api/vla/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: text,
          api_key: apiKey,
          model: model,
        }),
      });
      const res = await resp.json();
      console.log('[Google VLA] Command executed:', res);
      if (res.api_warning) {
        console.warn('[Google VLA Warning]', res.api_warning);
      }
      cmdInput.value = '';
    } catch (err) {
      console.error('Command submission error:', err);
    } finally {
      sendBtn.disabled = false;
      sendBtn.innerHTML = '<span>Execute</span>';
    }
  }

  sendBtn.addEventListener('click', () => submitCommand());
  cmdInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitCommand();
  });

  // 5. Preset Action Chips
  document.querySelectorAll('.prompt-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        cmdInput.value = prompt;
        submitCommand(prompt);
      }
    });
  });

  // 6. Camera Switching Toolbar & Zoom Controls
  document.querySelectorAll('.cam-btn[data-view]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cam-btn[data-view]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const view = btn.getAttribute('data-view');
      simCanvas.setCameraView(view);
    });
  });

  // Interactive Zoom & Reset Buttons
  document.getElementById('btn-zoom-in')?.addEventListener('click', () => {
    simCanvas.zoomIn();
  });

  document.getElementById('btn-zoom-out')?.addEventListener('click', () => {
    simCanvas.zoomOut();
  });

  document.getElementById('btn-cam-reset')?.addEventListener('click', () => {
    simCanvas.resetCamera();
  });

  // Viewport Size Toggle (Compact -> Standard -> Expanded -> Compact)
  const sizeToggleBtn = document.getElementById('btn-toggle-viewport-size');
  const sizeLabel = document.getElementById('viewport-size-label');
  const viewportBox = document.getElementById('viewport-container');
  const sizeModes = ['compact', 'standard', 'expanded'];
  let currentSizeModeIdx = 0; // Starts at compact

  if (sizeToggleBtn && viewportBox) {
    sizeToggleBtn.addEventListener('click', () => {
      currentSizeModeIdx = (currentSizeModeIdx + 1) % sizeModes.length;
      const nextMode = sizeModes[currentSizeModeIdx];

      viewportBox.classList.remove('viewport-compact', 'viewport-standard', 'viewport-expanded');
      viewportBox.classList.add(`viewport-${nextMode}`);

      if (sizeLabel) {
        sizeLabel.innerText = nextMode.charAt(0).toUpperCase() + nextMode.slice(1);
      }

      // Allow DOM reflow then recalculate Three.js projection and render buffer
      setTimeout(() => {
        simCanvas.onWindowResize();
      }, 50);
    });
  }

  // 7. Toggle Chips
  const trailChip = document.getElementById('toggle-trails');
  if (trailChip) {
    trailChip.addEventListener('click', () => {
      const state = simCanvas.toggleTrails();
      trailChip.classList.toggle('active', state);
    });
  }

  // 8. Rest & Simulation Controls
  async function triggerRestArm() {
    try {
      await fetch('/api/sim/reset', { method: 'POST' });
      console.log('[System] Arm returned to rest pose and workspace reset.');
    } catch (err) {
      console.error('Error resting arm:', err);
    }
  }

  document.getElementById('btn-rest-header')?.addEventListener('click', triggerRestArm);
  document.getElementById('btn-rest-inline')?.addEventListener('click', triggerRestArm);
  document.getElementById('btn-rest-main')?.addEventListener('click', triggerRestArm);
  document.getElementById('btn-reset')?.addEventListener('click', triggerRestArm);

  document.getElementById('btn-reset-cost')?.addEventListener('click', async () => {
    await fetch('/api/vla/cost/reset', { method: 'POST' });
  });

  // 9. Speech Recognition (Web Speech API)
  const micBtn = document.getElementById('mic-btn');
  if (micBtn && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRec();
    recognition.continuous = false;
    recognition.interimResults = false;

    micBtn.addEventListener('click', () => {
      micBtn.style.color = '#ffffff';
      recognition.start();
    });

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      cmdInput.value = transcript;
      micBtn.style.color = '';
      submitCommand(transcript);
    };

    recognition.onerror = () => {
      micBtn.style.color = '';
    };

    recognition.onend = () => {
      micBtn.style.color = '';
    };
  }

  // 10. Settings Modal for Google Gemini API Key
  const settingsModal = document.getElementById('settings-modal');
  const openSettingsBtn = document.getElementById('open-settings-btn');
  const saveSettingsBtn = document.getElementById('save-settings-btn');
  const closeSettingsBtn = document.getElementById('close-settings-btn');
  const apiKeyInput = document.getElementById('api-key-input');
  const modelSelect = document.getElementById('model-select');

  // Load saved settings
  if (apiKeyInput) apiKeyInput.value = localStorage.getItem('GEMINI_API_KEY') || '';
  if (modelSelect) modelSelect.value = localStorage.getItem('GEMINI_MODEL') || 'gemini-2.5-flash';

  openSettingsBtn?.addEventListener('click', () => settingsModal.classList.add('active'));
  closeSettingsBtn?.addEventListener('click', () => settingsModal.classList.remove('active'));

  saveSettingsBtn?.addEventListener('click', () => {
    localStorage.setItem('GEMINI_API_KEY', apiKeyInput.value.trim());
    localStorage.setItem('GEMINI_MODEL', modelSelect.value);
    settingsModal.classList.remove('active');
  });

  // Test API Key Button
  const testKeyBtn = document.getElementById('test-key-btn');
  const keyFeedback = document.getElementById('key-test-feedback');

  testKeyBtn?.addEventListener('click', async () => {
    const testKey = apiKeyInput.value.trim();
    if (!testKey) {
      if (keyFeedback) {
        keyFeedback.style.color = '#a1a1aa';
        keyFeedback.innerText = 'Please enter an API key to test.';
      }
      return;
    }

    testKeyBtn.disabled = true;
    testKeyBtn.innerText = 'Pinging...';
    if (keyFeedback) {
      keyFeedback.style.color = '#a1a1aa';
      keyFeedback.innerText = 'Testing connection to Google Gemini Cloud...';
    }

    try {
      const resp = await fetch('/api/vla/test_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: testKey, model: modelSelect.value }),
      });
      const data = await resp.json();
      if (data.valid) {
        if (keyFeedback) {
          keyFeedback.style.color = '#ffffff';
          keyFeedback.innerText = '✓ ' + data.message;
        }
      } else {
        if (keyFeedback) {
          keyFeedback.style.color = '#a1a1aa';
          keyFeedback.innerText = '✗ ' + (data.message || 'Validation failed');
        }
      }
    } catch (err) {
      if (keyFeedback) {
        keyFeedback.style.color = '#a1a1aa';
        keyFeedback.innerText = '✗ Network error: ' + err.message;
      }
    } finally {
      testKeyBtn.disabled = false;
      testKeyBtn.innerText = 'Test API Key';
    }
  });

  // 11. Refresh Mini VLA Camera Preview every 500ms
  const vlaImg = document.getElementById('vla-pip-img');
  setInterval(() => {
    if (vlaImg) {
      vlaImg.src = `/api/vla/camera_image?t=${Date.now()}`;
    }
  }, 600);

  // 12. Auto-execute command if passed via URL ?cmd=...
  const urlParams = new URLSearchParams(window.location.search);
  const initialCmd = urlParams.get('cmd');
  if (initialCmd && cmdInput) {
    cmdInput.value = initialCmd;
    setTimeout(() => {
      submitCommand(initialCmd);
    }, 1200);
  }
});
