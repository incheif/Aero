/**
 * Main Web Dashboard Controller & WebSocket Client
 * AERO: Autonomous Manipulation Studio
 */

document.addEventListener('DOMContentLoaded', () => {
  // 0. Session Isolation Management (Private Robot Workcell per Visitor)
  function getSessionId() {
    const urlParams = new URLSearchParams(window.location.search);
    let sid = urlParams.get('session_id') || sessionStorage.getItem('AERO_SESSION_ID');
    if (!sid) {
      sid = 'sess_' + Math.random().toString(36).substring(2, 8) + Date.now().toString(36).substring(4);
      sessionStorage.setItem('AERO_SESSION_ID', sid);
    }
    return sid;
  }
  const sessionId = getSessionId();

  // Populate Header Session Badge
  const sessionPillEl = document.getElementById('header-session-id');
  if (sessionPillEl) {
    sessionPillEl.innerText = sessionId.substring(0, 11);
    sessionPillEl.title = `Private Session: ${sessionId}`;
  }

  // Session-aware fetch helper
  function sessionFetch(url, options = {}) {
    options.headers = options.headers || {};
    if (!options.headers['X-Session-ID']) {
      options.headers['X-Session-ID'] = sessionId;
    }
    return fetch(url, options);
  }

  // 1. Initialize 3D Viewport
  const canvasEl = document.getElementById('canvas-3d');
  const simCanvas = new window.SimCanvas3D(canvasEl);
  window.simCanvas = simCanvas;

  // Immediate initial telemetry sync to ensure blocks & robot state mount instantly before WebSocket
  sessionFetch('/api/telemetry')
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((data) => {
      if (data && data.type === 'telemetry') {
        handleTelemetry(data);
      }
    })
    .catch((err) => {
      console.log('[Telemetry] Initial REST sync fallback:', err.message);
    });

  // 1B. Adaptive Workspace & Appearance Manager
  class WorkspaceManager {
    constructor(canvas) {
      this.simCanvas = canvas;
      this.layout = document.getElementById('app-layout');
      this.modalBackdrop = document.getElementById('modal-workspace-backdrop');
      this.btnSettings = document.getElementById('btn-workspace-settings');
      this.labelSettings = document.getElementById('btn-workspace-label');
      this.btnCloseModal = document.getElementById('btn-close-workspace-modal');
      this.btnDoneModal = document.getElementById('btn-modal-close-done');
      this.btnResetDefaults = document.getElementById('btn-modal-reset-defaults');
      this.btnExitPresentation = document.getElementById('btn-exit-presentation');
      this.presTopHud = document.getElementById('presentation-top-hud');
      this.presBottomBar = document.getElementById('presentation-bottom-bar');
      this.presStatusBadge = document.getElementById('pres-status-badge');
      this.presThoughtText = document.getElementById('pres-thought-text');
      this.btnToggleTheme = document.getElementById('btn-toggle-theme');

      // Defaults
      this.mode = 'developer';
      this.theme = 'graphite';
      this.panels = { sidebar: true, dock: true, pip: true };
      this.overlays = { axes: false, trails: true, frustums: false, boxes: false, grid: true };

      this.init();
    }

    init() {
      // 1. Restore Theme
      let savedTheme = localStorage.getItem('AERO_THEME');
      if (savedTheme === 'dark') savedTheme = 'graphite';
      if (!savedTheme || !['graphite', 'midnight', 'light', 'high-contrast'].includes(savedTheme)) {
        savedTheme = document.documentElement.getAttribute('data-theme') || 'graphite';
      }
      this.setTheme(savedTheme, false);

      // 2. Restore Mode
      const savedMode = localStorage.getItem('AERO_WORKSPACE_MODE');
      if (savedMode && ['engineering', 'developer', 'minimal', 'presentation'].includes(savedMode)) {
        this.mode = savedMode;
      }
      this.setMode(this.mode, false);

      // 3. Restore Panel Visibility
      try {
        const savedPanels = JSON.parse(localStorage.getItem('AERO_WORKSPACE_PANELS'));
        if (savedPanels && typeof savedPanels === 'object') {
          this.panels = { ...this.panels, ...savedPanels };
        }
      } catch (e) { }
      this.applyPanels(false);

      // 4. Restore 3D Overlays
      try {
        const savedOverlays = JSON.parse(localStorage.getItem('AERO_WORKSPACE_OVERLAYS'));
        if (savedOverlays && typeof savedOverlays === 'object') {
          this.overlays = { ...this.overlays, ...savedOverlays };
        }
      } catch (e) { }
      this.applyOverlays(false);

      // 5. Attach Listeners
      this.attachListeners();
    }

    setMode(mode, save = true) {
      this.mode = mode;
      if (this.layout) {
        this.layout.classList.remove('mode-engineering', 'mode-developer', 'mode-minimal', 'mode-presentation');
        this.layout.classList.add(`mode-${mode}`);
      }

      // Update button label in header
      if (this.labelSettings) {
        const labels = {
          engineering: 'Engineering',
          developer: 'Developer',
          minimal: 'Minimal',
          presentation: 'Presentation'
        };
        this.labelSettings.textContent = labels[mode] || 'Workspace';
      }

      // Update active card in modal
      document.querySelectorAll('.ws-mode-card').forEach(card => {
        const cardMode = card.getAttribute('data-mode');
        const isActive = cardMode === mode;
        card.classList.toggle('active', isActive);
        const pill = card.querySelector('.ws-mode-pill');
        if (pill) {
          pill.classList.toggle('ws-pill-active', isActive);
        }
      });

      // Special mode configurations
      if (mode === 'presentation') {
        if (this.presTopHud) this.presTopHud.style.display = 'flex';
        if (this.presBottomBar) this.presBottomBar.style.display = 'flex';
        if (this.btnExitPresentation) this.btnExitPresentation.style.display = 'inline-flex';
      } else {
        if (this.presTopHud) this.presTopHud.style.display = 'none';
        if (this.presBottomBar) this.presBottomBar.style.display = 'none';
        if (this.btnExitPresentation) this.btnExitPresentation.style.display = 'none';
      }

      // Trigger 3D canvas resize
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
        if (this.simCanvas && typeof this.simCanvas.onWindowResize === 'function') {
          this.simCanvas.onWindowResize();
        }
      }, 50);

      if (save) {
        try {
          localStorage.setItem('AERO_WORKSPACE_MODE', mode);
        } catch (e) { }
      }
    }

    setTheme(theme, save = true) {
      this.theme = theme;
      document.documentElement.setAttribute('data-theme', theme);

      // Update active theme swatch card
      document.querySelectorAll('.ws-theme-card').forEach(card => {
        card.classList.toggle('active', card.getAttribute('data-theme') === theme);
      });

      // Update Three.js scene lighting & background
      if (this.simCanvas && typeof this.simCanvas.setTheme === 'function') {
        this.simCanvas.setTheme(theme);
      }

      if (save) {
        try {
          localStorage.setItem('AERO_THEME', theme);
        } catch (e) { }
      }
    }

    applyPanels(save = true) {
      if (this.layout) {
        this.layout.classList.toggle('panel-sidebar-hidden', !this.panels.sidebar);
        this.layout.classList.toggle('panel-dock-hidden', !this.panels.dock);
        this.layout.classList.toggle('panel-pip-hidden', !this.panels.pip);
      }

      // Sync checkboxes
      const cbSidebar = document.getElementById('toggle-panel-sidebar');
      const cbDock = document.getElementById('toggle-panel-dock');
      const cbPip = document.getElementById('toggle-panel-pip');
      if (cbSidebar) cbSidebar.checked = !!this.panels.sidebar;
      if (cbDock) cbDock.checked = !!this.panels.dock;
      if (cbPip) cbPip.checked = !!this.panels.pip;

      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
        if (this.simCanvas && typeof this.simCanvas.onWindowResize === 'function') {
          this.simCanvas.onWindowResize();
        }
      }, 50);

      if (save) {
        try {
          localStorage.setItem('AERO_WORKSPACE_PANELS', JSON.stringify(this.panels));
        } catch (e) { }
      }
    }

    applyOverlays(save = true) {
      if (!this.simCanvas) return;

      if (typeof this.simCanvas.toggleAxes === 'function') {
        this.simCanvas.toggleAxes(this.overlays.axes);
      }
      if (typeof this.simCanvas.toggleTrails === 'function') {
        this.simCanvas.toggleTrails(this.overlays.trails);
      }
      if (typeof this.simCanvas.toggleFrustums === 'function') {
        this.simCanvas.toggleFrustums(this.overlays.frustums);
      }
      if (typeof this.simCanvas.togglePerceptionBoxes === 'function') {
        this.simCanvas.togglePerceptionBoxes(this.overlays.boxes);
      }
      if (typeof this.simCanvas.toggleGrid === 'function') {
        this.simCanvas.toggleGrid(this.overlays.grid);
      }

      // Sync checkboxes
      const cbAxes = document.getElementById('toggle-overlay-axes');
      const cbTrails = document.getElementById('toggle-overlay-trails');
      const cbFrustums = document.getElementById('toggle-overlay-frustums');
      const cbBoxes = document.getElementById('toggle-overlay-boxes');
      const cbGrid = document.getElementById('toggle-overlay-grid');
      if (cbAxes) cbAxes.checked = !!this.overlays.axes;
      if (cbTrails) cbTrails.checked = !!this.overlays.trails;
      if (cbFrustums) cbFrustums.checked = !!this.overlays.frustums;
      if (cbBoxes) cbBoxes.checked = !!this.overlays.boxes;
      if (cbGrid) cbGrid.checked = !!this.overlays.grid;

      if (save) {
        try {
          localStorage.setItem('AERO_WORKSPACE_OVERLAYS', JSON.stringify(this.overlays));
        } catch (e) { }
      }
    }

    resetDefaults() {
      this.mode = 'developer';
      this.theme = 'graphite';
      this.panels = { sidebar: true, dock: true, pip: true };
      this.overlays = { axes: false, trails: true, frustums: false, boxes: false, grid: true };

      this.setMode('developer', true);
      this.setTheme('graphite', true);
      this.applyPanels(true);
      this.applyOverlays(true);
    }

    openModal() {
      if (this.modalBackdrop) {
        this.modalBackdrop.style.display = 'flex';
      }
    }

    closeModal() {
      if (this.modalBackdrop) {
        this.modalBackdrop.style.display = 'none';
      }
    }

    attachListeners() {
      // Open / Close modal
      this.btnSettings?.addEventListener('click', () => this.openModal());
      this.btnCloseModal?.addEventListener('click', () => this.closeModal());
      this.btnDoneModal?.addEventListener('click', () => this.closeModal());
      this.modalBackdrop?.addEventListener('click', (e) => {
        if (e.target === this.modalBackdrop) this.closeModal();
      });

      // Mode cards
      document.querySelectorAll('.ws-mode-card').forEach(card => {
        card.addEventListener('click', () => {
          const mode = card.getAttribute('data-mode');
          if (mode) this.setMode(mode, true);
        });
      });

      // Theme cards
      document.querySelectorAll('.ws-theme-card').forEach(card => {
        card.addEventListener('click', () => {
          const theme = card.getAttribute('data-theme');
          if (theme) this.setTheme(theme, true);
        });
      });

      // Header Theme Quick-Toggle
      this.btnToggleTheme?.addEventListener('click', () => {
        const themeCycle = ['graphite', 'midnight', 'light', 'high-contrast'];
        const currentIndex = themeCycle.indexOf(this.theme);
        const nextTheme = themeCycle[(currentIndex + 1) % themeCycle.length];
        this.setTheme(nextTheme, true);
      });

      // Panel toggles
      document.getElementById('toggle-panel-sidebar')?.addEventListener('change', (e) => {
        this.panels.sidebar = e.target.checked;
        this.applyPanels(true);
      });
      document.getElementById('toggle-panel-dock')?.addEventListener('change', (e) => {
        this.panels.dock = e.target.checked;
        this.applyPanels(true);
      });
      document.getElementById('toggle-panel-pip')?.addEventListener('change', (e) => {
        this.panels.pip = e.target.checked;
        this.applyPanels(true);
      });

      // Overlay toggles
      document.getElementById('toggle-overlay-axes')?.addEventListener('change', (e) => {
        this.overlays.axes = e.target.checked;
        this.applyOverlays(true);
      });
      document.getElementById('toggle-overlay-trails')?.addEventListener('change', (e) => {
        this.overlays.trails = e.target.checked;
        this.applyOverlays(true);
      });
      document.getElementById('toggle-overlay-frustums')?.addEventListener('change', (e) => {
        this.overlays.frustums = e.target.checked;
        this.applyOverlays(true);
      });
      document.getElementById('toggle-overlay-boxes')?.addEventListener('change', (e) => {
        this.overlays.boxes = e.target.checked;
        this.applyOverlays(true);
      });
      document.getElementById('toggle-overlay-grid')?.addEventListener('change', (e) => {
        this.overlays.grid = e.target.checked;
        this.applyOverlays(true);
      });

      // Reset Defaults
      this.btnResetDefaults?.addEventListener('click', () => {
        this.resetDefaults();
      });

      // Presentation Exit Button
      this.btnExitPresentation?.addEventListener('click', () => {
        this.setMode('developer', true);
      });

      // Global keyboard shortcuts
      document.addEventListener('keydown', (e) => {
        // Alt+W: Toggle Workspace Modal
        if (e.altKey && (e.key === 'w' || e.key === 'W')) {
          e.preventDefault();
          if (this.modalBackdrop?.style.display === 'flex') {
            this.closeModal();
          } else {
            this.openModal();
          }
        }
        // Alt+T: Cycle Themes
        if (e.altKey && (e.key === 't' || e.key === 'T')) {
          e.preventDefault();
          const themeCycle = ['graphite', 'midnight', 'light', 'high-contrast'];
          const currentIndex = themeCycle.indexOf(this.theme);
          const nextTheme = themeCycle[(currentIndex + 1) % themeCycle.length];
          this.setTheme(nextTheme, true);
        }
        // Escape: Exit presentation mode or close modal
        if (e.key === 'Escape') {
          if (this.modalBackdrop?.style.display === 'flex') {
            this.closeModal();
          } else if (this.mode === 'presentation') {
            this.setMode('developer', true);
          }
        }
      });
    }

    updateTelemetryHUD(telemetry) {
      if (this.mode !== 'presentation') return;
      if (!telemetry) return;

      if (this.presStatusBadge) {
        const state = telemetry.state || (telemetry.dialogue && telemetry.dialogue.state) || 'READY';
        this.presStatusBadge.textContent = state;
        if (state === 'EXECUTING') {
          this.presStatusBadge.style.color = '#10b981';
          this.presStatusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
        } else if (state.includes('PAUSED')) {
          this.presStatusBadge.style.color = '#f59e0b';
          this.presStatusBadge.style.background = 'rgba(245, 158, 11, 0.15)';
        } else if (state === 'FAILED' || state === 'REJECTED') {
          this.presStatusBadge.style.color = '#ef4444';
          this.presStatusBadge.style.background = 'rgba(239, 68, 68, 0.15)';
        } else {
          this.presStatusBadge.style.color = '#10b981';
          this.presStatusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
        }
      }

      if (this.presThoughtText) {
        const thought = (telemetry.dialogue && telemetry.dialogue.latest_thought) ||
          (telemetry.dialogue && telemetry.dialogue.active_subgoal_desc) ||
          telemetry.thought ||
          'Standing by for instruction';
        this.presThoughtText.textContent = thought;
      }
    }
  }

  const workspaceMgr = new WorkspaceManager(simCanvas);
  window.workspaceMgr = workspaceMgr;

  // 2. WebSocket Telemetry Client
  let ws = null;
  let isConnected = false;
  let isSuspended = false;
  let isEvicted = false;

  function connectWebSocket() {
    if (isSuspended || isEvicted) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws?session_id=${encodeURIComponent(sessionId)}`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      isConnected = true;
      updateConnectionStatus(true);
      console.log('[WebSocket] Connected to simulation server.');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'session_evicted') {
          handleSessionEvicted(data);
          return;
        }
        if (data.type === 'session_suspended' || data.is_suspended) {
          handleSessionSuspended(data);
          return;
        }
        if (data.type === 'telemetry') {
          handleTelemetry(data);
        } else if (data.logs && Array.isArray(data.logs)) {
          updateLogsFromTelemetry(data.logs);
        }
      } catch (err) {
        console.error('Error parsing telemetry JSON:', err);
      }
    };

    ws.onclose = (event) => {
      isConnected = false;
      if (event && event.code === 4001) {
        isEvicted = true;
      }
      if (isEvicted) {
        updateConnectionStatus(false, 'SESSION CLOSED (MAX 5)');
        return;
      }
      if (!isSuspended) {
        updateConnectionStatus(false);
        console.warn('[WebSocket] Disconnected. Reconnecting in 1.5s...');
        setTimeout(connectWebSocket, 1500);
      }
    };

    ws.onerror = (err) => {
      console.error('[WebSocket] Error:', err);
      ws.close();
    };
  }

  function handleSessionEvicted(data) {
    isEvicted = true;
    isConnected = false;
    updateConnectionStatus(false, 'SESSION CLOSED (MAX 5)');
    showEvictedBanner(data && data.reason ? data.reason : 'Session closed: server reached maximum limit of 5 concurrent sessions.');
  }

  function showEvictedBanner(reason) {
    let banner = document.getElementById('session-evicted-banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'session-evicted-banner';
      banner.className = 'session-suspended-banner';
      banner.innerHTML = `
        <div class="suspended-banner-content">
          <div class="suspended-banner-icon" style="color: #ef4444;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="15" y1="9" x2="9" y2="15"></line>
              <line x1="9" y1="9" x2="15" y2="15"></line>
            </svg>
          </div>
          <div class="suspended-banner-text">
            <strong>Session Closed (Max 5 Limit)</strong>
            <span>${escapeHtml(reason || 'Session closed because the server reached the maximum limit of 5 concurrent sessions.')}</span>
          </div>
          <button id="btn-reconnect-evicted" class="btn-resume-session" style="background: #ef4444; color: #fff;">Reconnect</button>
        </div>
      `;
      document.body.appendChild(banner);
      document.getElementById('btn-reconnect-evicted')?.addEventListener('click', reconnectEvictedSession);
    }
    banner.style.display = 'flex';
  }

  function reconnectEvictedSession() {
    isEvicted = false;
    const banner = document.getElementById('session-evicted-banner');
    if (banner) banner.style.display = 'none';
    connectWebSocket();
  }

  function handleSessionSuspended(data) {
    isSuspended = true;
    isConnected = false;
    updateConnectionStatus(false, 'SUSPENDED (25 MIN)');
    showSuspensionBanner(data && data.reason ? data.reason : 'Session suspended after 25 minutes to conserve cloud bandwidth.');
    if (data && data.logs && Array.isArray(data.logs)) {
      updateLogsFromTelemetry(data.logs);
    }
  }

  async function resumeSession() {
    isSuspended = false;
    hideSuspensionBanner();
    try {
      await sessionFetch('/api/session/resume', { method: 'POST' });
    } catch (_) { }
    if (!isConnected) {
      connectWebSocket();
    }
  }

  function showSuspensionBanner(reason) {
    let banner = document.getElementById('session-suspended-banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'session-suspended-banner';
      banner.className = 'session-suspended-banner';
      banner.innerHTML = `
        <div class="suspended-banner-content">
          <div class="suspended-banner-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
          </div>
          <div class="suspended-banner-text">
            <strong>Session Suspended (25-Min Cap)</strong>
            <span>${escapeHtml(reason || 'Session paused to conserve cloud resources & prevent bandwidth overuse.')}</span>
          </div>
          <button id="btn-resume-session" class="btn-resume-session">Resume Session</button>
        </div>
      `;
      document.body.appendChild(banner);
      document.getElementById('btn-resume-session')?.addEventListener('click', resumeSession);
    }
    banner.style.display = 'flex';
  }

  function hideSuspensionBanner() {
    const banner = document.getElementById('session-suspended-banner');
    if (banner) {
      banner.style.display = 'none';
    }
  }

  connectWebSocket();

  // 3. UI State Handlers
  function updateConnectionStatus(connected, customText) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (dot) {
      if (connected) {
        dot.style.backgroundColor = '#ffffff';
        dot.style.boxShadow = '0 0 8px rgba(255, 255, 255, 0.8)';
      } else if (isSuspended) {
        dot.style.backgroundColor = '#f59e0b';
        dot.style.boxShadow = '0 0 8px rgba(245, 158, 11, 0.5)';
      } else {
        dot.style.backgroundColor = '#52525b';
        dot.style.boxShadow = 'none';
      }
    }
    if (text) {
      if (customText) {
        text.innerText = customText;
      } else {
        text.innerText = connected ? 'ONLINE / 60 FPS' : 'RECONNECTING...';
      }
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

    // Update Presentation HUD if active
    if (window.workspaceMgr) {
      window.workspaceMgr.updateTelemetryHUD(data);
    }

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
      const dockFb = document.getElementById('dock-feedback-text');
      if (dockFb && data.vla.thought) {
        dockFb.innerText = data.vla.thought;
      }
      const dockState = document.getElementById('chat-state-badge');
      if (dockState && data.vla.status) {
        dockState.innerText = data.vla.status;
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
        modelPill.innerText = `AERO (${data.vla.model})`;
      }

      // Update VLA Mode Badge & Explainer (Autonomous Zero-Dependency Runtime)
      const modeBadge = document.getElementById('vla-mode-badge');
      const modeTagStatus = document.getElementById('mode-tag-status');
      const modeInfoText = document.getElementById('mode-info-text');

      if (modeBadge) {
        modeBadge.innerText = 'AUTONOMOUS VLA';
        modeBadge.style.color = '#10b981';
        modeBadge.style.backgroundColor = 'rgba(16, 185, 129, 0.12)';
        modeBadge.style.borderColor = 'rgba(16, 185, 129, 0.35)';
        modeBadge.title = 'Running Embedded Autonomous VLA Engine';
        if (modeTagStatus) {
          modeTagStatus.innerText = 'ZERO-DEPENDENCY RUNTIME';
          modeTagStatus.className = 'mode-tag-pill';
        }
        if (modeInfoText) {
          modeInfoText.innerText = 'Autonomous multimodal policy engine: 3D spatial grounding, collision-free trajectory synthesis, and physics hold verification.';
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
      if (j.wristRoll !== undefined) {
        setVal('joint-roll', `${(j.wristRoll * 57.2958).toFixed(1)}° (${j.wristRoll.toFixed(2)})`);
      }
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
        const curTicks = Math.min(stab.hold_ticks || 0, stab.required_ticks || 18);
        const holdText = `${curTicks} / ${stab.required_ticks || 18}`;
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

    // 6B-2. Feetech STS3215 Actuator Dynamics Telemetry
    if (data.actuators) {
      const act = data.actuators;
      const pWatts = act.total_power_watts !== undefined ? `${act.total_power_watts.toFixed(1)} W` : '-- W';
      const pkTorque = act.peak_torque !== undefined ? `${act.peak_torque.toFixed(2)} Nm` : '-- Nm';
      setVal('actuator-power-display', pWatts);
      setVal('actuator-peak-torque', pkTorque);

      if (act.loads_pct) {
        const loads = Object.values(act.loads_pct);
        const avgLoad = loads.length ? (loads.reduce((a, b) => a + b, 0) / loads.length).toFixed(1) : '0.0';
        setVal('actuator-avg-load', `${avgLoad}%`);
      }

      if (act.temperatures_c) {
        const temps = Object.values(act.temperatures_c);
        const maxT = temps.length ? Math.max(...temps).toFixed(1) : '24.5';
        setVal('actuator-max-temp', `${maxT}°C`);
      }

      setVal('dock-actuators-power', `${pWatts} · ${pkTorque}`);
    }

    // 6C. Update Computer Vision & Camera HUD PiP Telemetry
    if (data.perception) {
      const p = data.perception;
      const targetLabel = document.getElementById('hud-target-label');
      const pipStatus = document.getElementById('hud-pip-status');
      const metaCalib = document.getElementById('meta-calib-status');
      const metaPhase = document.getElementById('meta-exec-phase');
      const confBadge = document.getElementById('perception-conf-badge');

      if (metaCalib) metaCalib.innerText = p.calibrated ? 'CAMERA → BASE [CALIBRATED]' : 'UNCALIBRATED';
      if (metaPhase) metaPhase.innerText = (data.vla && data.vla.status === 'EXECUTING') ? 'EXECUTE / OBSERVE' : (p.execution_phase || 'OBSERVE / IDLE');
      if (confBadge) confBadge.innerText = `${Math.round((p.confidence_avg || 0.96) * 100)}% CONF`;

      if (p.tracks) {
        updatePerceptionTracks(p.tracks);
      }

      if (targetLabel) {
        if (p.target_block_id) {
          const cleanName = p.target_block_id.replace('block_', '').toUpperCase();
          targetLabel.innerText = `TARGET: ${cleanName}`;
          targetLabel.style.color = '#38bdf8';
        } else {
          targetLabel.innerText = 'TARGET: TRACKING';
          targetLabel.style.color = '#a1a1aa';
        }
      }
      if (pipStatus) {
        if (data.vla && data.vla.status === 'EXECUTING') {
          pipStatus.innerText = 'TRACKING LOCK';
          pipStatus.style.color = '#38bdf8';
          pipStatus.style.borderColor = 'rgba(56, 189, 248, 0.4)';
          pipStatus.style.backgroundColor = 'rgba(56, 189, 248, 0.15)';
        } else {
          pipStatus.innerText = 'CV ACQUIRED';
          pipStatus.style.color = '#6ee7b7';
          pipStatus.style.borderColor = 'rgba(16, 185, 129, 0.35)';
          pipStatus.style.backgroundColor = 'rgba(16, 185, 129, 0.15)';
        }
      }
    }

    if (data.tcp && data.tcp.position) {
      const tcp = data.tcp.position;
      setVal('meta-tcp-est', `[${tcp[0].toFixed(2)}, ${tcp[1].toFixed(2)}, ${tcp[2].toFixed(2)}]`);
    }

    if (data.semantic_map && data.semantic_map.relations) {
      const relChips = document.getElementById('relations-chips');
      const relCount = document.getElementById('relations-count-tag');
      const rels = data.semantic_map.relations;
      if (relCount) relCount.innerText = `${rels.length} relations`;
      if (relChips) {
        if (rels.length > 0) {
          relChips.innerHTML = rels.map(r => `
            <span class="rel-chip"><strong>${r.subject}</strong> ${r.relation} <em>${r.object}</em></span>
          `).join('');
        } else {
          relChips.innerHTML = '<span class="rel-chip">No stacked relations</span>';
        }
      }
    }

    // 6D. Update Frontier Mapping & Occupancy Navigation Telemetry
    if (data.frontier) {
      const f = data.frontier;
      const metaFrontierPct = document.getElementById('meta-frontier-pct');
      const metaFrontierWpt = document.getElementById('meta-frontier-wpt');
      const metaGridStats = document.getElementById('meta-grid-stats');
      if (metaFrontierPct) metaFrontierPct.innerText = `${f.exploration_pct || 25}% EXPLORED`;
      if (metaFrontierWpt) metaFrontierWpt.innerText = f.is_exploring ? (f.current_waypoint || 'SCANNING') : 'READY';
      if (metaGridStats && f.occupancy_grid) {
        let occ = 0;
        let free = 0;
        f.occupancy_grid.forEach(row => row.forEach(c => {
          if (c === 2) occ++;
          else if (c === 1) free++;
        }));
        metaGridStats.innerText = `${free} FREE / ${occ} OCCUPIED`;
      }
    }

    // 7. Update LLM VLA Cost Tracking Metrics
    if (data.cost) {
      updateCostMetrics(data.cost);
    }

    // 8. Update Pause Button Icon
    const pauseBtn = document.getElementById('btn-pause');
    if (pauseBtn) {
      pauseBtn.innerText = data.is_paused ? 'Resume' : 'Pause';
    }

    // 9. Update Execution Logs
    if (data.logs && Array.isArray(data.logs)) {
      updateLogsFromTelemetry(data.logs);
    }

    // 10. Update Conversational Dialogue & Human-in-the-Loop
    if (data.dialogue) {
      updateDialogueUI(data.dialogue);
    }

    // 11. Process Narrative Story Logs
    if (data.narrative_logs && Array.isArray(data.narrative_logs) && data.narrative_logs.length > 0) {
      appendNarrativeLogs(data.narrative_logs);
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
    tabBtnTelemetry?.classList.toggle('active', tabName === 'telemetry');
    tabBtnLogs?.classList.toggle('active', tabName === 'logs');

    if (telemetryView) telemetryView.style.display = tabName === 'telemetry' ? 'flex' : 'none';
    if (logsView) logsView.style.display = tabName === 'logs' ? 'flex' : 'none';

    if (tabName === 'logs') {
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
      await sessionFetch('/api/logs/clear', { method: 'POST' });
      localLogs = [];
      lastLogsSignature = '';
      renderLogs();
    } catch (err) {
      console.error('Error clearing logs:', err);
    }
  });

  document.getElementById('btn-open-logs-tab')?.addEventListener('click', () => {
    window.open(`/logs?session_id=${encodeURIComponent(sessionId)}`, '_blank');
  });

  // 4b. Agent Execution Thought Stream Size Controls (S / M / L / XL)
  const THOUGHT_SIZES = ['sm', 'md', 'lg', 'xl'];
  const THOUGHT_SIZE_LABELS = { sm: 'S', md: 'M', lg: 'L', xl: 'XL' };
  let currentThoughtSize = localStorage.getItem('AERO_THOUGHT_SIZE') || 'md';
  const thoughtBoxEl = document.getElementById('vla-thought-box');
  const thoughtSizeLabelEl = document.getElementById('thought-size-label');
  const btnThoughtSmaller = document.getElementById('btn-thought-smaller');
  const btnThoughtLarger = document.getElementById('btn-thought-larger');

  function applyThoughtSize(size) {
    if (!thoughtBoxEl) return;
    THOUGHT_SIZES.forEach(s => thoughtBoxEl.classList.remove(`size-${s}`));
    thoughtBoxEl.classList.add(`size-${size}`);
    currentThoughtSize = size;
    localStorage.setItem('AERO_THOUGHT_SIZE', size);
    if (thoughtSizeLabelEl) {
      thoughtSizeLabelEl.innerText = THOUGHT_SIZE_LABELS[size] || 'M';
    }
  }

  btnThoughtSmaller?.addEventListener('click', (e) => {
    e.stopPropagation();
    const idx = THOUGHT_SIZES.indexOf(currentThoughtSize);
    if (idx > 0) applyThoughtSize(THOUGHT_SIZES[idx - 1]);
  });

  btnThoughtLarger?.addEventListener('click', (e) => {
    e.stopPropagation();
    const idx = THOUGHT_SIZES.indexOf(currentThoughtSize);
    if (idx < THOUGHT_SIZES.length - 1) applyThoughtSize(THOUGHT_SIZES[idx + 1]);
  });

  applyThoughtSize(currentThoughtSize);

  // --------------------------------------------------------------------------
  // Dynamic Section Sizing Controls (Dock, Sidebar, & Card Collapse)
  // --------------------------------------------------------------------------
  const dockEl = document.getElementById('command-bar-container');
  const dockResizer = document.getElementById('dock-resizer');

  // Restore saved dock height if previously dragged
  const savedDockHeight = localStorage.getItem('AERO_DOCK_HEIGHT');
  if (savedDockHeight && dockEl) {
    const h = parseInt(savedDockHeight, 10);
    if (!isNaN(h) && h >= 68 && h <= 500) {
      dockEl.style.height = `${h}px`;
      dockEl.style.maxHeight = `${h}px`;
    }
  }

  // Improved Smooth Draggable Dock Resizer (RAF-debounced, clean bounds)
  if (dockResizer && dockEl) {
    let isDraggingDock = false;
    let startY = 0;
    let startHeight = 0;
    let animFrameId = null;

    dockResizer.addEventListener('mousedown', (e) => {
      e.preventDefault();
      isDraggingDock = true;
      startY = e.clientY;
      startHeight = dockEl.getBoundingClientRect().height;
      dockResizer.classList.add('active');
      document.body.classList.add('resizing-active');
      document.body.style.cursor = 'ns-resize';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDraggingDock) return;
      const delta = startY - e.clientY;
      const newHeight = Math.max(68, Math.min(startHeight + delta, 440));
      if (animFrameId) cancelAnimationFrame(animFrameId);
      animFrameId = requestAnimationFrame(() => {
        dockEl.classList.remove('size-compact', 'size-normal', 'size-expanded');
        dockEl.style.maxHeight = `${newHeight}px`;
        dockEl.style.height = `${newHeight}px`;
      });
    });

    window.addEventListener('mouseup', () => {
      if (isDraggingDock) {
        isDraggingDock = false;
        dockResizer.classList.remove('active');
        document.body.classList.remove('resizing-active');
        document.body.style.cursor = '';
        const finalHeight = dockEl.getBoundingClientRect().height;
        localStorage.setItem('AERO_DOCK_HEIGHT', finalHeight);
        window.dispatchEvent(new Event('resize'));
      }
    });
  }

  // Sidebar Width Controls
  const layoutContainer = document.getElementById('app-layout');
  const sidebarResizer = document.getElementById('sidebar-resizer');

  const savedSidebarWidth = localStorage.getItem('AERO_SIDEBAR_WIDTH');
  if (savedSidebarWidth && layoutContainer) {
    const w = parseInt(savedSidebarWidth, 10);
    if (!isNaN(w) && w >= 300 && w <= 700) {
      layoutContainer.style.setProperty('--sidebar-width', `${w}px`);
    }
  }

  // Improved Smooth Draggable Sidebar Resizer (RAF-debounced, clean bounds)
  if (sidebarResizer && layoutContainer) {
    let isDraggingSidebar = false;
    let startX = 0;
    let startWidth = 0;
    let animFrameId = null;

    sidebarResizer.addEventListener('mousedown', (e) => {
      e.preventDefault();
      isDraggingSidebar = true;
      startX = e.clientX;
      const sidebar = document.getElementById('telemetry-panel');
      startWidth = sidebar ? sidebar.getBoundingClientRect().width : 440;
      sidebarResizer.classList.add('active');
      document.body.classList.add('resizing-active');
      document.body.style.cursor = 'ew-resize';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDraggingSidebar) return;
      const delta = startX - e.clientX;
      const newWidth = Math.max(300, Math.min(startWidth + delta, 640));
      if (animFrameId) cancelAnimationFrame(animFrameId);
      animFrameId = requestAnimationFrame(() => {
        layoutContainer.style.setProperty('--sidebar-width', `${newWidth}px`);
        btnSidebarNarrow?.classList.remove('active');
        btnSidebarNormal?.classList.remove('active');
        btnSidebarWide?.classList.remove('active');
      });
    });

    window.addEventListener('mouseup', () => {
      if (isDraggingSidebar) {
        isDraggingSidebar = false;
        sidebarResizer.classList.remove('active');
        document.body.classList.remove('resizing-active');
        document.body.style.cursor = '';
        const sidebar = document.getElementById('telemetry-panel');
        if (sidebar) {
          localStorage.setItem('AERO_SIDEBAR_WIDTH', sidebar.getBoundingClientRect().width);
        }
        window.dispatchEvent(new Event('resize'));
      }
    });
  }

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
      if (badge === 'FAILURE' || badge === 'FAILED') badgeClass = 'badge-failure';
      else if (badge === 'REJECT' || badge === 'IMPOSSIBLE') badgeClass = 'badge-reject';
      else if (badge === 'EXECUTION STARTED' || badge === 'STARTED') badgeClass = 'badge-started';
      else if (cat === 'perception') badgeClass = 'badge-perception';
      else if (cat === 'vla') badgeClass = 'badge-vla';
      else if (cat === 'motion') badgeClass = 'badge-motion';
      else if (cat === 'contact') badgeClass = 'badge-contact';
      else if (cat === 'stability') badgeClass = 'badge-stability';

      let catClass = `cat-${cat}`;
      if (badge === 'FAILURE' || badge === 'FAILED') catClass = 'cat-failure';
      else if (badge === 'EXECUTION STARTED' || badge === 'STARTED') catClass = 'cat-started';

      return `
        <div class="log-entry ${catClass}" data-id="${log.id}">
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

  // --------------------------------------------------------------------------
  // Conversational Human-in-the-Loop & Dialogue Supervisor
  // --------------------------------------------------------------------------
  let lastRenderedChatMsgId = '';
  let lastStepsSignature = '';

  function appendNarrativeLogs(narratives) {
    if (!narratives || narratives.length === 0) return;
    const existingIds = new Set(localLogs.map(l => l.id));
    let hasNew = false;
    for (const n of narratives) {
      if (!existingIds.has(n.id)) {
        localLogs.push({
          id: n.id,
          timestamp: n.timestamp,
          category: n.category,
          badge: (n.category || 'event').toUpperCase(),
          what: n.title,
          how: n.explanation,
        });
        existingIds.add(n.id);
        hasNew = true;
      }
    }
    if (hasNew && activeTab === 'logs') {
      renderLogs();
    }
  }

  function updateDialogueUI(d) {
    if (!d) return;

    // 1. State badge & Dot
    const badge = document.getElementById('chat-state-badge');
    const dot = document.getElementById('chat-status-dot');
    const badgeDot = document.getElementById('chat-badge-dot');

    if (badge) {
      badge.innerText = d.state || 'IDLE';
      if (d.state === 'PAUSED_BY_USER' || d.state === 'PAUSED_BY_ANOMALY') {
        badge.style.color = '#f59e0b';
        badge.style.background = 'rgba(245, 158, 11, 0.15)';
        badge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        if (dot) dot.style.background = '#f59e0b';
        if (badgeDot) badgeDot.style.display = 'inline-block';
      } else if (d.state === 'EXECUTING') {
        badge.style.color = '#10b981';
        badge.style.background = 'rgba(16, 185, 129, 0.15)';
        badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        if (dot) dot.style.background = '#10b981';
        if (badgeDot) badgeDot.style.display = 'none';
      } else if (d.state === 'FAILED' || d.state === 'REJECTED') {
        badge.style.color = '#ef4444';
        badge.style.background = 'rgba(239, 68, 68, 0.15)';
        badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        if (dot) dot.style.background = '#ef4444';
        if (badgeDot) badgeDot.style.display = 'none';
        const dockCard = document.getElementById('dock-agent-feedback');
        if (dockCard) {
          dockCard.classList.remove('feedback-planning');
          dockCard.classList.add('feedback-failed');
        }
      } else {
        badge.style.color = '';
        badge.style.background = '';
        badge.style.borderColor = '';
        if (dot) dot.style.background = '#a1a1aa';
        if (badgeDot) badgeDot.style.display = 'none';
        const dockCard = document.getElementById('dock-agent-feedback');
        if (dockCard) {
          dockCard.classList.remove('feedback-failed');
        }
      }
    }

    // 2. Update Single-Line Dock Agent Feedback
    const feedbackText = document.getElementById('dock-feedback-text');
    if (feedbackText && d) {
      if (d.active_subgoal_desc && d.active_subgoal_desc !== 'Idle') {
        feedbackText.innerText = d.active_subgoal_desc;
      } else if (d.latest_thought) {
        feedbackText.innerText = d.latest_thought;
      } else if (d.chat_history && d.chat_history.length > 0) {
        const lastMsg = d.chat_history[d.chat_history.length - 1];
        if (lastMsg && lastMsg.text) {
          feedbackText.innerText = lastMsg.text;
        }
      }
    }
  }

  function renderChatHistory(msgs) {
    const stream = document.getElementById('chat-messages-stream');
    if (!stream) return;
    stream.innerHTML = msgs.map(m => {
      const isUser = m.sender === 'user';
      const avatar = isUser ? `
          <div class="chat-avatar-icon user-avatar-icon" style="display: flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 6px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; flex-shrink: 0;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
          </div>` : `
          <div class="chat-avatar-icon robot-avatar-icon" style="display: flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 6px; background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.12); color: #38bdf8; flex-shrink: 0;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8.01" y2="16"></line><line x1="16" y1="16" x2="16.01" y2="16"></line></svg>
          </div>`;
      const author = isUser ? 'You (Operator)' : 'AERO Robot Arm';
      const bg = isUser ? 'rgba(56, 189, 248, 0.1)' : 'rgba(255, 255, 255, 0.05)';
      const border = isUser ? 'rgba(56, 189, 248, 0.25)' : 'var(--border-subtle, rgba(255,255,255,0.08))';
      const align = isUser ? 'flex-direction: row-reverse;' : '';
      return `
        <div class="chat-bubble" style="display: flex; gap: 8px; align-items: flex-start; ${align}">
          ${avatar}
          <div style="background: ${bg}; border: 1px solid ${border}; border-radius: 10px; padding: 8px 11px; font-size: 12px; line-height: 1.45; max-width: 85%;">
            <div style="font-size: 10.5px; font-weight: 600; color: #a1a1aa; margin-bottom: 3px; display: flex; justify-content: space-between; gap: 8px;">
              <span>${author}</span>
              <span style="opacity: 0.6; font-size: 10px;">${m.timestamp || ''}</span>
            </div>
            <div style="white-space: pre-wrap;">${escapeHtml(m.text || '')}</div>
          </div>
        </div>
      `;
    }).join('');
    stream.scrollTop = stream.scrollHeight;
  }

  async function sendGuidanceMessage(text) {
    if (!text) return;
    try {
      const res = await sessionFetch('/api/vla/guidance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ guidance: text })
      });
      const data = await res.json();
      console.log('[Guidance Sent]:', data);
    } catch (err) {
      console.error('[Guidance Error]:', err);
    }
  }

  async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';

    try {
      const res = await sessionFetch('/api/vla/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
      });
      const data = await res.json();
      console.log('[Chat Sent]:', data);
    } catch (err) {
      console.error('[Chat Error]:', err);
    }
  }

  // Wire Chat and Pause & Guide Listeners
  const btnChatSend = document.getElementById('btn-chat-send');
  const chatInput = document.getElementById('chat-input');
  btnChatSend?.addEventListener('click', sendChatMessage);
  chatInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendChatMessage();
    }
  });

  const btnPauseGuide = document.getElementById('btn-pause-guide');
  btnPauseGuide?.addEventListener('click', async () => {
    try {
      const res = await sessionFetch('/api/vla/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'Operator clicked Pause & Guide', session_id: sessionId })
      });
      const data = await res.json();
      console.log('[Pause & Guide]:', data);
      cmdInput?.focus();
    } catch (err) {
      console.error('[Pause & Guide Error]:', err);
    }
  });

  const btnChatReset = document.getElementById('btn-chat-reset');
  btnChatReset?.addEventListener('click', async () => {
    if (cmdInput) {
      cmdInput.value = '';
      cmdInput.focus();
    }
    const fb = document.getElementById('dock-feedback-text');
    if (fb) {
      fb.innerText = 'Awaiting instruction... Enter a natural language command below or select a challenge chip.';
    }
    const badge = document.getElementById('chat-state-badge');
    if (badge) {
      badge.innerText = 'IDLE';
      badge.style.color = '';
      badge.style.background = '';
      badge.style.borderColor = '';
    }
    try {
      await sessionFetch('/api/vla/chat/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (err) {
      console.error('[Command Reset Error]:', err);
    }
  });

  // Initial fetch of logs from REST endpoint on load
  function fetchSessionLogs() {
    sessionFetch('/api/logs')
      .then(r => r.json())
      .then(data => {
        if (data && data.logs && Array.isArray(data.logs)) {
          updateLogsFromTelemetry(data.logs);
        }
      })
      .catch(() => { });
  }
  fetchSessionLogs();
  setInterval(fetchSessionLogs, 2500);

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

  let lastTracksHtml = '';
  function updatePerceptionTracks(tracks) {
    const summaryEl = document.getElementById('perception-tracks-summary');
    const tbody = document.getElementById('perception-tracks-body');
    if (!summaryEl && !tbody) return;

    if (!tracks || tracks.length === 0) {
      if (summaryEl) summaryEl.innerHTML = '<span class="rel-chip">No objects detected</span>';
      if (tbody) tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-dim); padding: 8px;">No active tracks</td></tr>';
      return;
    }

    if (summaryEl) {
      const summaryHtml = tracks.map((t) => {
        const cName = t.color_name ? (t.color_name.charAt(0).toUpperCase() + t.color_name.slice(1)) : (t.class_name || 'Block');
        const cDot = t.color_rgb ? `rgb(${t.color_rgb[0]}, ${t.color_rgb[1]}, ${t.color_rgb[2]})` : '#38bdf8';
        return `<span class="scene-object-chip"><span class="color-dot" style="background-color: ${cDot};"></span>${cName}</span>`;
      }).join('');
      if (summaryHtml !== lastTracksHtml) {
        lastTracksHtml = summaryHtml;
        summaryEl.innerHTML = summaryHtml;
      }
      return;
    }

    let html = '';
    tracks.forEach((t) => {
      const pos = t.position || [0, 0, 0];
      const cName = t.color_name ? (t.color_name.charAt(0).toUpperCase() + t.color_name.slice(1) + ' Cube') : (t.class_name || 'Block');
      const cDot = t.color_rgb ? `rgb(${t.color_rgb[0]}, ${t.color_rgb[1]}, ${t.color_rgb[2]})` : '#38bdf8';
      const conf = Math.round((t.confidence || 0.95) * 100);
      const isAct = t.state === 'ACTIVE';

      html += `
        <tr>
          <td><span class="track-id-badge">${t.track_id}</span></td>
          <td><span class="color-dot" style="background-color: ${cDot};"></span>${cName}</td>
          <td class="est-pos-val">[${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}, ${pos[2].toFixed(2)}] <small style="color: #71717a; font-size: 0.58rem;">(est)</small></td>
          <td>${conf}%</td>
          <td><span style="color: ${isAct ? '#10b981' : '#f59e0b'}; font-weight: 600; font-size: 0.60rem;">${t.state || 'ACTIVE'}</span></td>
        </tr>
      `;
    });

    if (html !== lastTracksHtml) {
      lastTracksHtml = html;
      tbody.innerHTML = html;
    }
  }

  // 4. Command Input Handling
  const cmdInput = document.getElementById('command-input');
  const sendBtn = document.getElementById('send-command-btn');

  async function submitCommand(commandText) {
    const text = (commandText || cmdInput.value || '').trim();
    if (!text) return;

    if (isSuspended) {
      resumeSession();
    }

    // Retain and display command in text box so user can reference what was queried
    cmdInput.value = text;

    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="logs-dot-pulse"></span> Reasoning...';

    // Immediate UI feedback so user immediately knows execution has started
    const dockFb = document.getElementById('dock-feedback-text');
    const dockContainer = document.getElementById('dock-agent-feedback');
    const stateBadge = document.getElementById('chat-state-badge');
    const chatDot = document.getElementById('chat-status-dot');

    if (dockFb) {
      dockFb.innerText = `Execution started: Synthesizing spatial trajectory for "${text}"...`;
    }
    if (dockContainer) {
      dockContainer.classList.add('feedback-planning');
    }
    if (stateBadge) {
      stateBadge.innerText = 'PLANNING';
      stateBadge.style.color = '#38bdf8';
      stateBadge.style.background = 'rgba(56, 189, 248, 0.15)';
      stateBadge.style.borderColor = 'rgba(56, 189, 248, 0.4)';
    }
    if (chatDot) {
      chatDot.style.background = '#38bdf8';
    }

    // Insert immediate "EXECUTION STARTED" log in Execution Story tab
    const startLog = {
      id: 'local_start_' + Date.now(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      category: 'vla',
      badge: 'STARTED',
      what: `Execution started: "${text}"`,
      how: 'Reasoning pipeline engaged · Capturing overhead RGB perception & solving 5-DOF IK waypoints. Motion initiating shortly.',
    };
    localLogs.push(startLog);
    if (activeTab === 'logs') {
      renderLogs();
    }

    const apiKey = localStorage.getItem('GEMINI_API_KEY') || '';
    const model = localStorage.getItem('GEMINI_MODEL') || 'gemini-2.5-flash';

    try {
      const badge = document.getElementById('chat-state-badge');
      const isPaused = badge && (badge.innerText.includes('PAUSE'));
      const endpoint = isPaused ? '/api/vla/chat' : '/api/vla/command';
      const payload = isPaused ? { message: text, session_id: sessionId } : {
        command: text,
        api_key: apiKey,
        model: model,
        session_id: sessionId,
      };

      const resp = await sessionFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
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

  // Wire Presentation Mode Command Input
  const presCmdInput = document.getElementById('pres-cmd-input');
  const presBtnSend = document.getElementById('pres-btn-send');
  if (presBtnSend && presCmdInput) {
    const handlePresSend = () => {
      const val = presCmdInput.value.trim();
      if (val) {
        submitCommand(val);
        presCmdInput.value = '';
      }
    };
    presBtnSend.addEventListener('click', handlePresSend);
    presCmdInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handlePresSend();
    });
  }

  // 5. Preset Action Chips: Populate input bar and wait for user to press Enter or Execute
  document.querySelectorAll('.prompt-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.prompt-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      const prompt = chip.getAttribute('data-prompt');
      if (prompt && cmdInput) {
        cmdInput.value = prompt;
        cmdInput.focus();
        cmdInput.setSelectionRange(prompt.length, prompt.length);
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

  const btnToggleFrustum = document.getElementById('btn-toggle-frustum');
  if (btnToggleFrustum) {
    btnToggleFrustum.addEventListener('click', () => {
      simCanvas.showFrustums = !simCanvas.showFrustums;
      btnToggleFrustum.classList.toggle('active', simCanvas.showFrustums);
    });
  }

  const btnToggleP3D = document.getElementById('btn-toggle-perception-3d');
  if (btnToggleP3D) {
    btnToggleP3D.addEventListener('click', () => {
      simCanvas.showPerceptionBoxes = !simCanvas.showPerceptionBoxes;
      btnToggleP3D.classList.toggle('active', simCanvas.showPerceptionBoxes);
    });
  }

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
      await sessionFetch('/api/sim/reset', { method: 'POST' });
      console.log('[System] Arm returned to rest pose and workspace reset.');
    } catch (err) {
      console.error('Error resting arm:', err);
    }
  }

  async function triggerRandomizeLayout() {
    try {
      await sessionFetch('/api/sim/randomize', { method: 'POST' });
      console.log('[System] Blocks randomized across tabletop.');
    } catch (err) {
      console.error('Error randomizing blocks:', err);
    }
  }

  async function triggerNudgeBlock() {
    const btn = document.getElementById('btn-nudge-block');
    if (btn) btn.disabled = true;
    try {
      const res = await sessionFetch('/api/sim/nudge', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'ok') {
        console.log(`[Disturbance] Nudged ${data.name} to`, data.new_position);
      }
    } catch (err) {
      console.error('Error nudging block:', err);
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  document.getElementById('btn-rest-header')?.addEventListener('click', triggerRestArm);
  document.getElementById('btn-rest-inline')?.addEventListener('click', triggerRestArm);
  document.getElementById('btn-rest-main')?.addEventListener('click', triggerRestArm);
  document.getElementById('btn-randomize-main')?.addEventListener('click', triggerRandomizeLayout);
  document.getElementById('btn-nudge-block')?.addEventListener('click', triggerNudgeBlock);
  document.getElementById('btn-reset')?.addEventListener('click', triggerRestArm);

  document.getElementById('btn-reset-cost')?.addEventListener('click', async () => {
    await sessionFetch('/api/vla/cost/reset', { method: 'POST' });
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
      cmdInput.focus();
      cmdInput.setSelectionRange(transcript.length, transcript.length);
      micBtn.style.color = '';
    };

    recognition.onerror = () => {
      micBtn.style.color = '';
    };

    recognition.onend = () => {
      micBtn.style.color = '';
    };
  }

  // 10. Settings Modal Handlers (Preserved for future extensions)
  const settingsModal = document.getElementById('settings-modal');
  const openSettingsBtn = document.getElementById('open-settings-btn');
  const saveSettingsBtn = document.getElementById('save-settings-btn');
  const closeSettingsBtn = document.getElementById('close-settings-btn');
  const apiKeyInput = document.getElementById('api-key-input');
  const modelSelect = document.getElementById('model-select');

  // Load saved settings if inputs exist
  if (apiKeyInput) apiKeyInput.value = localStorage.getItem('GEMINI_API_KEY') || '';
  if (modelSelect) modelSelect.value = localStorage.getItem('GEMINI_MODEL') || 'gemini-2.5-flash';

  openSettingsBtn?.addEventListener('click', () => settingsModal?.classList.add('active'));
  closeSettingsBtn?.addEventListener('click', () => settingsModal?.classList.remove('active'));

  saveSettingsBtn?.addEventListener('click', () => {
    if (apiKeyInput) localStorage.setItem('GEMINI_API_KEY', apiKeyInput.value.trim());
    if (modelSelect) localStorage.setItem('GEMINI_MODEL', modelSelect.value);
    settingsModal?.classList.remove('active');
  });

  // Test Key Handler
  const testKeyBtn = document.getElementById('test-key-btn');
  const keyFeedback = document.getElementById('key-test-feedback');

  testKeyBtn?.addEventListener('click', async () => {
    const testKey = apiKeyInput ? apiKeyInput.value.trim() : '';
    if (!testKey) {
      if (keyFeedback) {
        keyFeedback.style.color = '#a1a1aa';
        keyFeedback.innerText = 'Please enter an authorization key to test.';
      }
      return;
    }

    testKeyBtn.disabled = true;
    testKeyBtn.innerText = 'Pinging...';
    if (keyFeedback) {
      keyFeedback.style.color = '#a1a1aa';
      keyFeedback.innerText = 'Testing connection...';
    }

    try {
      const resp = await sessionFetch('/api/vla/test_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: testKey, model: modelSelect ? modelSelect.value : 'gemini-2.5-flash', session_id: sessionId }),
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
      testKeyBtn.innerText = 'Test Key';
    }
  });

  // 11. Sensor HUD Switcher: Live 30 FPS WebGL perspective cameras for OVERHEAD / WRIST / DEPTH,
  //     and backend 2D frontier map PNG polling for MAP mode.
  const vlaImg = document.getElementById('vla-pip-img');
  const vlaCanvas = document.getElementById('vla-pip-canvas');
  let currentFeedType = 'rgb';
  const btnFeedRgb = document.getElementById('btn-feed-rgb');
  const btnFeedWrist = document.getElementById('btn-feed-wrist');
  const btnFeedDepth = document.getElementById('btn-feed-depth');
  const btnFeedMap = document.getElementById('btn-feed-map');
  const sensorTypeTag = document.getElementById('hud-sensor-type-tag');

  const hudFeedConfigs = [
    { btn: btnFeedRgb, type: 'rgb', tag: 'OVERHEAD RGB' },
    { btn: btnFeedWrist, type: 'wrist', tag: 'WRIST (EYE-IN-HAND)' },
    { btn: btnFeedDepth, type: 'depth', tag: 'METRIC DEPTH' },
    { btn: btnFeedMap, type: 'map', tag: 'FRONTIER MAP' },
  ];

  const syncFeedVisibility = () => {
    const isMap = currentFeedType === 'map';
    if (simCanvas) simCanvas.hudFeedMode = currentFeedType;
    if (vlaCanvas) vlaCanvas.style.display = isMap ? 'none' : 'block';
    if (vlaImg) vlaImg.style.display = isMap ? 'block' : 'none';
  };

  const refreshMapImg = () => {
    if (!vlaImg || currentFeedType !== 'map') return;
    const nextImg = new Image();
    nextImg.onload = () => {
      if (currentFeedType === 'map') vlaImg.src = nextImg.src;
    };
    nextImg.src = `/api/vla/map_image?session_id=${encodeURIComponent(sessionId)}&annotated=true&t=${Date.now()}`;
  };

  hudFeedConfigs.forEach((cfg) => {
    if (!cfg.btn) return;
    cfg.btn.addEventListener('click', (e) => {
      e.stopPropagation();
      hudFeedConfigs.forEach((c) => c.btn?.classList.remove('active'));
      cfg.btn.classList.add('active');
      currentFeedType = cfg.type;
      if (sensorTypeTag) sensorTypeTag.innerText = cfg.tag;
      syncFeedVisibility();
      if (currentFeedType === 'map') {
        refreshMapImg();
      }
    });
  });

  syncFeedVisibility();
  setInterval(refreshMapImg, 500);

  // 11b. Main Big-Screen Multi-View Manager (1 = full view, 2 = 2-way split, 3 = 3-way split)
  const splitGrid = document.getElementById('split-screen-grid');
  const screenButtons = Array.from(document.querySelectorAll('.screen-btn[data-screen]'));
  const screenLayoutBadge = document.getElementById('screen-layout-badge');
  const SCREEN_LABELS = {
    studio: 'STUDIO 3D',
    rgb: 'OVERHEAD RGB',
    wrist: 'WRIST (EYE-IN-HAND)',
    depth: 'METRIC DEPTH',
  };
  let selectedMainScreens = ['studio'];

  const renderScreenLayout = () => {
    if (!splitGrid) return;
    // Filter out any stale 'map' selection if persisted
    selectedMainScreens = selectedMainScreens.filter(s => s !== 'map');
    if (selectedMainScreens.length === 0) selectedMainScreens = ['studio'];

    const count = selectedMainScreens.length;
    splitGrid.className = `split-${Math.min(4, Math.max(1, count))}`;

    screenButtons.forEach((btn) => {
      const s = btn.getAttribute('data-screen');
      btn.classList.toggle('active', selectedMainScreens.includes(s));
    });

    if (screenLayoutBadge) {
      if (count === 1) screenLayoutBadge.innerText = '1 VIEW';
      else if (count <= 3) screenLayoutBadge.innerText = `${count} SPLIT`;
      else screenLayoutBadge.innerText = `${count} GRID`;
    }

    splitGrid.innerHTML = '';
    const isDefaultStudio = count === 1 && selectedMainScreens[0] === 'studio';

    selectedMainScreens.forEach((screenType) => {
      const pane = document.createElement('div');
      pane.className = 'split-pane';
      pane.setAttribute('data-screen', screenType);

      if (!isDefaultStudio) {
        const chip = document.createElement('div');
        chip.className = 'split-pane-chip';

        const dot = document.createElement('span');
        dot.className = `split-pane-dot dot-${screenType}`;
        chip.appendChild(dot);

        const txt = document.createElement('span');
        txt.innerText = SCREEN_LABELS[screenType] || screenType.toUpperCase();
        chip.appendChild(txt);

        if (count > 1) {
          const closeBtn = document.createElement('button');
          closeBtn.className = 'split-pane-close';
          closeBtn.type = 'button';
          closeBtn.title = `Close ${SCREEN_LABELS[screenType] || screenType}`;
          closeBtn.innerText = '×';
          closeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleMainScreen(screenType);
          });
          chip.appendChild(closeBtn);
        }
        pane.appendChild(chip);
      }

      splitGrid.appendChild(pane);
    });

    if (simCanvas && typeof simCanvas.setMainScreens === 'function') {
      simCanvas.setMainScreens(selectedMainScreens, () => splitGrid.querySelectorAll('.split-pane'));
    }
  };

  const toggleMainScreen = (screenType) => {
    const idx = selectedMainScreens.indexOf(screenType);
    if (idx !== -1) {
      if (selectedMainScreens.length <= 1) return; // Always keep at least 1 feed on main screen
      selectedMainScreens.splice(idx, 1);
    } else {
      selectedMainScreens.push(screenType);
    }
    renderScreenLayout();
  };

  screenButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const s = btn.getAttribute('data-screen');
      if (s) toggleMainScreen(s);
    });
  });

  renderScreenLayout();

  // 12. Pre-fill command input if passed via URL ?cmd=... (wait for user to press Enter/Execute)
  const urlParams = new URLSearchParams(window.location.search);
  const initialCmd = urlParams.get('cmd');
  if (initialCmd && cmdInput) {
    cmdInput.value = initialCmd;
    cmdInput.focus();
    cmdInput.setSelectionRange(initialCmd.length, initialCmd.length);
  }

  // 13. Responsive Drawer & Viewport Controls (Mobile & Laptop Adaptation)
  const appLayout = document.getElementById('app-layout');
  const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
  const btnCloseSidebar = document.getElementById('btn-close-sidebar');
  const telemetryPanel = document.getElementById('telemetry-panel');
  const sidebarBackdrop = document.getElementById('sidebar-backdrop');
  const btnPipToggle = document.getElementById('btn-pip-toggle');
  const cameraHudPip = document.getElementById('camera-hud-pip');

  function openMobileDrawer() {
    telemetryPanel?.classList.add('drawer-open');
    sidebarBackdrop?.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeMobileDrawer() {
    telemetryPanel?.classList.remove('drawer-open');
    sidebarBackdrop?.classList.remove('active');
    document.body.style.overflow = '';
  }

  function toggleSidebar() {
    if (window.innerWidth <= 768) {
      if (telemetryPanel?.classList.contains('drawer-open')) {
        closeMobileDrawer();
      } else {
        openMobileDrawer();
      }
    } else {
      appLayout?.classList.toggle('sidebar-collapsed');
      btnToggleSidebar?.classList.toggle('active', !appLayout?.classList.contains('sidebar-collapsed'));
      // Notify Three.js renderer to instantly resize canvas to available space
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 50);
    }
  }

  btnToggleSidebar?.addEventListener('click', toggleSidebar);
  btnCloseSidebar?.addEventListener('click', closeMobileDrawer);
  sidebarBackdrop?.addEventListener('click', closeMobileDrawer);

  // Close drawer with Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeMobileDrawer();
    }
  });

  // Toggle Camera PiP minimization
  btnPipToggle?.addEventListener('click', (e) => {
    e.stopPropagation();
    cameraHudPip?.classList.toggle('minimized');
    const isMin = cameraHudPip?.classList.contains('minimized');
    btnPipToggle.innerHTML = isMin
      ? '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>'
      : '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"></line></svg>';
    if (!isMin) {
      refreshMapImg();
    }
  });

  // Clicking on minimized bottom-left pill expands it
  cameraHudPip?.addEventListener('click', (e) => {
    if (cameraHudPip.classList.contains('minimized') && !e.target.closest('button')) {
      cameraHudPip.classList.remove('minimized');
      if (btnPipToggle) {
        btnPipToggle.innerHTML = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"></line></svg>';
      }
      refreshMapImg();
    }
  });
});

