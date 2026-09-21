/**
 * code_diff.js
 * Python Controller Code Diff Renderer & Editor for AERO Dashboard.
 */

class CodeDiffRenderer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.currentCode = '';
    this.previousCode = '';
    this.viewMode = 'diff'; // 'diff' or 'full'
  }

  updateCode(current, previous) {
    this.currentCode = current || '';
    this.previousCode = previous || current || '';
    this.render();
  }

  toggleViewMode() {
    this.viewMode = this.viewMode === 'diff' ? 'full' : 'diff';
    this.render();
    return this.viewMode;
  }

  computeDiff(oldText, newText) {
    const oldLines = oldText.split('\n');
    const newLines = newText.split('\n');
    const diff = [];

    // Simple line comparison for clean display
    let i = 0, j = 0;
    while (i < oldLines.length || j < newLines.length) {
      if (i < oldLines.length && j < newLines.length && oldLines[i] === newLines[j]) {
        diff.push({ type: 'neutral', text: newLines[j], line: j + 1 });
        i++;
        j++;
      } else if (j < newLines.length && (i >= oldLines.length || !oldLines.includes(newLines[j]))) {
        diff.push({ type: 'add', text: `+ ${newLines[j]}`, line: j + 1 });
        j++;
      } else if (i < oldLines.length && (j >= newLines.length || !newLines.includes(oldLines[i]))) {
        diff.push({ type: 'del', text: `- ${oldLines[i]}`, line: i + 1 });
        i++;
      } else {
        diff.push({ type: 'add', text: `+ ${newLines[j]}`, line: j + 1 });
        j++;
        i++;
      }
    }
    return diff;
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = '';

    if (this.viewMode === 'full') {
      const lines = this.currentCode.split('\n');
      lines.forEach((line, idx) => {
        const span = document.createElement('div');
        span.className = 'diff-line diff-neutral';
        span.textContent = `${String(idx + 1).padStart(3, ' ')}  ${line}`;
        this.container.appendChild(span);
      });
      return;
    }

    // Diff mode
    const diff = this.computeDiff(this.previousCode, this.currentCode);
    diff.forEach((item) => {
      const span = document.createElement('div');
      span.className = `diff-line diff-${item.type}`;
      const lineNum = String(item.line || '').padStart(3, ' ');
      const prefix = item.type === 'neutral' ? '   ' : '';
      span.textContent = `${lineNum} ${prefix}${item.text}`;
      this.container.appendChild(span);
    });
  }
}

window.CodeDiffRenderer = CodeDiffRenderer;
