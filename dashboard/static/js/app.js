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

  let lastDomUpdateTime = 0;

  function handleTelemetry(data) {
    // 1. Pass to 3D Three.js Canvas (Real-time 60 FPS continuous update)
    simCanvas.updateFromTelemetry(data);

    // 2. Throttle heavy DOM string parsing and text node mutations to ~10 Hz (every 90ms)
    // This frees the browser main thread completely for 60 FPS WebGL rendering without layout thrashing
    const now = performance.now();
    if (now - lastDomUpdateTime < 90) {
      return;
    }
    lastDomUpdateTime = now;

    // 2. Update Header FPS & Ticks
    const fpsEl = document.getElementById('fps-metric');
    if (fpsEl) fpsEl.innerText = `${data.fps || 60} FPS`;

    // 3. Update VLA Thought Stream Card
    const thoughtBox = document.getElementById('vla-thought-box');
    const vlaBadge = document.getElementById('vla-status-badge');
    const modelPill = document.getElementById('model-pill-text');

    if (data.vla) {
      if (thoughtBox) {
        thoughtBox.innerText = data.vla.thought || 'Awaiting instruction...';
        if (data.vla.status === 'REJECTED') {
          thoughtBox.classList.add('rejected');
          thoughtBox.classList.remove('gesture');
        } else if (data.vla.thought && (data.vla.thought.toLowerCase().includes('gesture') || data.vla.thought.toLowerCase().includes('wave'))) {
          thoughtBox.classList.add('gesture');
          thoughtBox.classList.remove('rejected');
        } else {
          thoughtBox.classList.remove('rejected', 'gesture');
        }
      }
      if (vlaBadge) {
        vlaBadge.innerText = data.vla.status || 'IDLE';
        if (data.vla.status === 'REJECTED') {
          vlaBadge.className = 'thought-status-badge badge-rejected';
          vlaBadge.style.color = '';
          vlaBadge.style.backgroundColor = '';
          vlaBadge.style.borderColor = '';
        } else if (data.vla.status === 'EXECUTING') {
          vlaBadge.className = 'thought-status-badge';
          vlaBadge.style.color = '#09090b';
          vlaBadge.style.backgroundColor = '#ffffff';
          vlaBadge.style.borderColor = '#ffffff';
        } else {
          vlaBadge.className = 'thought-status-badge';
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

    // 6B. Update Playground Evaluation & Stability Hold Telemetry
    if (data.scores) {
      const sc = data.scores;
      if (sc.spatial_accuracy !== undefined) {
        const spatPct = (sc.spatial_accuracy * 100).toFixed(1) + '%';
        setVal('dock-spatial-accuracy', spatPct);
        setVal('spatial-acc-display', spatPct);
      }
      if (sc.task_completion !== undefined) {
        setVal('completion-score-display', `${sc.task_completion.toFixed(2)} / 1.00`);
      }
      if (sc.torque) {
        setVal('peak-torque-display', `${(sc.torque.peak || 0).toFixed(2)}`);
        setVal('avg-torque-display', `${(sc.torque.avg || 0).toFixed(2)}`);
      }
      if (sc.stability) {
        const stab = sc.stability;
        const holdText = `${stab.hold_ticks || 0} / ${stab.required_ticks || 18}`;
        setVal('hold-ticks-display', holdText);
        setVal('stability-text-display', stab.status || 'Pending');
        setVal('dock-stability-status', stab.is_verified ? 'VERIFIED (0.3s)' : (stab.status || 'IDLE'));

        const stabBadge = document.getElementById('telemetry-stability-badge');
        if (stabBadge) {
          if (stab.is_verified) {
            stabBadge.innerText = 'STABILITY: VERIFIED (0.3s)';
            stabBadge.style.color = '#10b981';
            stabBadge.style.backgroundColor = 'rgba(16, 185, 129, 0.15)';
            stabBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
          } else if (stab.hold_ticks > 0) {
            stabBadge.innerText = `HOLDING: ${stab.hold_ticks}/18`;
            stabBadge.style.color = '#38bdf8';
            stabBadge.style.backgroundColor = 'rgba(56, 189, 248, 0.15)';
            stabBadge.style.borderColor = 'rgba(56, 189, 248, 0.4)';
          } else {
            stabBadge.innerText = 'STABILITY: IDLE';
            stabBadge.style.color = '#a1a1aa';
            stabBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
            stabBadge.style.borderColor = 'rgba(255, 255, 255, 0.2)';
          }
        }
      }
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

    // 9. Update Execution Logs
    if (data.logs && Array.isArray(data.logs)) {
      updateLogsFromTelemetry(data.logs);
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

  // --------------------------------------------------------------------------
  // Execution & Audit Logs Panel Management
  // --------------------------------------------------------------------------
  const tabBtnTelemetry = document.getElementById('tab-btn-telemetry');
  const tabBtnLogs = document.getElementById('tab-btn-logs');
  const telemetryView = document.getElementById('telemetry-view');
  const logsView = document.getElementById('logs-view');
  const logsBadgeCount = document.getElementById('logs-badge-count');
  const logsTotalCount = document.getElementById('logs-total-count');
  const logsList = document.getElementById('logs-list');
  const logsContainer = document.getElementById('logs-stream-container');
  const btnToggleAutoScroll = document.getElementById('btn-toggle-autoscroll');
  const autoscrollLabel = document.getElementById('autoscroll-label');
  const btnClearLogs = document.getElementById('btn-clear-logs');
  const filterChips = document.querySelectorAll('.log-filter-chip');

  let activeTab = 'telemetry';
  let autoScrollEnabled = true;
  let activeLogFilter = 'all';
  let localLogs = [];
  let lastLogsSignature = '';

  function switchTab(tabName) {
    activeTab = tabName;
    if (tabName === 'telemetry') {
      tabBtnTelemetry?.classList.add('active');
      tabBtnLogs?.classList.remove('active');
      if (telemetryView) telemetryView.style.display = 'flex';
      if (logsView) logsView.style.display = 'none';
    } else {
      tabBtnLogs?.classList.add('active');
      tabBtnTelemetry?.classList.remove('active');
      if (telemetryView) telemetryView.style.display = 'none';
      if (logsView) logsView.style.display = 'flex';
      renderLogs();
      if (autoScrollEnabled && logsContainer) {
        logsContainer.scrollTop = logsContainer.scrollHeight;
      }
    }
  }

  tabBtnTelemetry?.addEventListener('click', () => switchTab('telemetry'));
  tabBtnLogs?.addEventListener('click', () => switchTab('logs'));

  btnToggleAutoScroll?.addEventListener('click', () => {
    autoScrollEnabled = !autoScrollEnabled;
    btnToggleAutoScroll?.classList.toggle('active', autoScrollEnabled);
    if (autoscrollLabel) {
      autoscrollLabel.innerText = autoScrollEnabled ? 'Scroll: ON' : 'Scroll: OFF';
    }
    if (autoScrollEnabled && logsContainer) {
      logsContainer.scrollTop = logsContainer.scrollHeight;
    }
  });

  btnClearLogs?.addEventListener('click', async () => {
    try {
      await fetch('/api/logs/clear', { method: 'POST' });
      localLogs = [];
      lastLogsSignature = '';
      renderLogs();
    } catch (err) {
      console.error('Error clearing logs:', err);
    }
  });

  filterChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      filterChips.forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      activeLogFilter = chip.getAttribute('data-filter') || 'all';
      renderLogs();
    });
  });

  function updateLogsFromTelemetry(incomingLogs) {
    if (!incomingLogs || incomingLogs.length === 0) return;

    // Fast signature check to avoid DOM thrashing when logs haven't changed
    const sig = incomingLogs.map(l => l.id).join(',');
    if (sig === lastLogsSignature) return;
    lastLogsSignature = sig;

    // Merge incoming logs into localLogs without duplicates
    const existingIds = new Set(localLogs.map(l => l.id));
    let hasNew = false;
    for (const log of incomingLogs) {
      if (!existingIds.has(log.id)) {
        localLogs.push(log);
        existingIds.add(log.id);
        hasNew = true;
      }
    }

    if (localLogs.length > 250) {
      localLogs = localLogs.slice(localLogs.length - 250);
    }

    if (logsBadgeCount) logsBadgeCount.innerText = localLogs.length;
    if (logsTotalCount) logsTotalCount.innerText = `${localLogs.length} events`;

    if (activeTab === 'logs' && hasNew) {
      renderLogs();
    }
  }

  function renderLogs() {
    if (!logsList) return;

    const filtered = activeLogFilter === 'all'
      ? localLogs
      : localLogs.filter(l => (l.category || '').toLowerCase() === activeLogFilter.toLowerCase());

    if (logsBadgeCount) logsBadgeCount.innerText = localLogs.length;
    if (logsTotalCount) logsTotalCount.innerText = `${localLogs.length} events`;

    if (filtered.length === 0) {
      logsList.innerHTML = `
        <div class="logs-empty-state">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="color: var(--mono-500);"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span>No logs recorded for category: <strong>${activeLogFilter.toUpperCase()}</strong></span>
        </div>
      `;
      return;
    }

    const html = filtered.map(log => {
      const cat = (log.category || 'system').toLowerCase();
      const badge = log.badge || 'INFO';
      let badgeClass = 'badge-system';
      if (badge === 'REJECT' || badge === 'IMPOSSIBLE') badgeClass = 'badge-reject';
      else if (cat === 'vla') badgeClass = 'badge-vla';
      else if (cat === 'motion') badgeClass = 'badge-motion';
      else if (cat === 'contact') badgeClass = 'badge-contact';
      else if (cat === 'stability') badgeClass = 'badge-stability';

      return `
        <div class="log-entry cat-${cat}" data-id="${log.id}">
          <div class="log-entry-header">
            <span class="log-time">${log.timestamp || ''}</span>
            <span class="log-badge ${badgeClass}">${badge}</span>
          </div>
          <div class="log-what">
            <span class="log-what-tag">WHAT:</span> ${escapeHtml(log.what || '')}
          </div>
          <div class="log-how">
            <span class="log-how-tag">HOW:</span> ${escapeHtml(log.how || '')}
          </div>
        </div>
      `;
    }).join('');

    logsList.innerHTML = html;

    if (autoScrollEnabled && logsContainer) {
      logsContainer.scrollTop = logsContainer.scrollHeight;
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Initial fetch of logs from REST endpoint on load
  fetch('/api/logs')
    .then(r => r.json())
    .then(data => {
      if (data && data.logs) {
        updateLogsFromTelemetry(data.logs);
      }
    })
    .catch(() => {});

  let lastSemanticHtml = '';
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
    if (html !== lastSemanticHtml) {
      lastSemanticHtml = html;
      tbody.innerHTML = html;
    }
  }

  // 4. Command Input Handling
  const cmdInput = document.getElementById('command-input');
  const sendBtn = document.getElementById('send-command-btn');

  async function submitCommand(commandText) {
    const text = (commandText || cmdInput.value || '').trim();
    if (!text) return;

    // Retain and display command in text box so user can reference what was queried
    cmdInput.value = text;

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
      if (res.is_possible === false || res.status === 'IMPOSSIBLE') {
        const tBox = document.getElementById('vla-thought-box');
        const vBadge = document.getElementById('vla-status-badge');
        if (tBox) {
          tBox.innerText = res.reason || res.thought || 'Instruction cannot be physically executed.';
          tBox.classList.add('rejected');
          tBox.classList.remove('gesture');
        }
        if (vBadge) {
          vBadge.innerText = 'REJECTED';
          vBadge.className = 'thought-status-badge badge-rejected';
        }
      }
      if (res.api_warning) {
        console.warn('[Google VLA Warning]', res.api_warning);
      }
      // Do not remove command from text box - keep it visible for user reference
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
      document.querySelectorAll('.prompt-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        cmdInput.value = prompt;
        cmdInput.focus();
        submitCommand(prompt);
      }
    });
  });

  // Remove chip highlight when user manually types or edits query in text box
  cmdInput.addEventListener('input', () => {
    document.querySelectorAll('.prompt-chip').forEach((c) => c.classList.remove('active'));
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

  // Viewport Size Toggle (Full -> Standard -> Expanded -> Full)
  const sizeToggleBtn = document.getElementById('btn-toggle-viewport-size');
  const sizeLabel = document.getElementById('viewport-size-label');
  const viewportBox = document.getElementById('viewport-container');
  const sizeModes = ['fill', 'standard', 'expanded'];
  let currentSizeModeIdx = 0; // Starts at full fill

  if (sizeToggleBtn && viewportBox) {
    sizeToggleBtn.addEventListener('click', () => {
      currentSizeModeIdx = (currentSizeModeIdx + 1) % sizeModes.length;
      const nextMode = sizeModes[currentSizeModeIdx];

      viewportBox.classList.remove('viewport-fill', 'viewport-compact', 'viewport-standard', 'viewport-expanded');
      viewportBox.classList.add(`viewport-${nextMode}`);

      if (sizeLabel) {
        sizeLabel.innerText = nextMode === 'fill' ? 'Full' : (nextMode.charAt(0).toUpperCase() + nextMode.slice(1));
      }

      // Allow DOM reflow then recalculate Three.js projection and render buffer
      setTimeout(() => {
        simCanvas.onWindowResize();
      }, 50);
    });
  }

  // Performance Mode Toggle (Smooth 60 FPS <-> Retina HD)
  const perfToggleBtn = document.getElementById('btn-toggle-perf-mode');
  const perfLabel = document.getElementById('perf-mode-label');
  if (perfToggleBtn && perfLabel) {
    const curMode = simCanvas.perfMode || 'smooth';
    perfLabel.innerText = curMode === 'smooth' ? '⚡ 60 FPS' : '✦ HD';

    perfToggleBtn.addEventListener('click', () => {
      const newMode = simCanvas.togglePerfMode();
      perfLabel.innerText = newMode === 'smooth' ? '⚡ 60 FPS' : '✦ HD';
      console.log(`[Graphics] Performance Mode set to: ${newMode}`);
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
