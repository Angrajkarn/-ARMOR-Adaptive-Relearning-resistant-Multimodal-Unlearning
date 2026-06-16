/* ============================================================
   APEX — Trade Journal & Review Protocol
   ============================================================ */

const Journal = {

  // ── Data Access ─────────────────────────────────────────────
  getAll()  { return JSON.parse(localStorage.getItem('apex_journal') || '[]'); },
  save(arr) { localStorage.setItem('apex_journal', JSON.stringify(arr)); },

  add(trade) {
    const arr = this.getAll();
    trade.id        = 'T' + Date.now();
    trade.createdAt = new Date().toISOString();
    trade.status    = trade.status || 'OPEN';
    arr.unshift(trade);
    this.save(arr);
    return trade;
  },

  update(id, updates) {
    const arr = this.getAll();
    const idx = arr.findIndex(t => t.id === id);
    if (idx !== -1) {
      arr[idx] = { ...arr[idx], ...updates, updatedAt: new Date().toISOString() };
      this.save(arr);
      return arr[idx];
    }
    return null;
  },

  remove(id) {
    const arr = this.getAll().filter(t => t.id !== id);
    this.save(arr);
  },

  getById(id) { return this.getAll().find(t => t.id === id) || null; },

  // ── Emotion Definitions ──────────────────────────────────────
  emotions: [
    { id: 'calm',        label: '😌 Calm',        type: 'good' },
    { id: 'confident',   label: '💪 Confident',    type: 'good' },
    { id: 'anxious',     label: '😰 Anxious',      type: 'warn' },
    { id: 'excited',     label: '🔥 Excited',      type: 'warn' },
    { id: 'fomo',        label: '⚡ FOMO',         type: 'bad' },
    { id: 'greedy',      label: '🤑 Greedy',       type: 'bad' },
    { id: 'fearful',     label: '😨 Fearful',      type: 'bad' },
    { id: 'revenge',     label: '😤 Revenge',      type: 'bad' },
    { id: 'overconfident', label: '😎 Overconfident', type: 'bad' },
    { id: 'bored',       label: '😑 Bored (Best)', type: 'good' },
  ],

  // ── Review Questions ─────────────────────────────────────────
  reviewQuestions: [
    'Was the setup valid per my rules? (Y/N)',
    'Was entry timing optimal? What could improve it?',
    'Did I follow my stop/target plan? If not, why?',
    'What was the actual vs expected outcome?',
    'Emotional state during the trade (calm / anxious / greedy / FOMO)',
  ],

  // ── Render Journal List ──────────────────────────────────────
  renderList(containerId, filter = 'ALL') {
    const container = document.getElementById(containerId);
    if (!container) return;

    let trades = this.getAll();
    if (filter !== 'ALL') trades = trades.filter(t => t.status === filter);

    if (!trades.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="es-icon">📓</div>
          <div class="es-title">No trades logged yet</div>
          <div class="es-text">Use the Trade Analyzer to generate and save Trade Briefs</div>
        </div>`;
      return;
    }

    container.innerHTML = trades.map(t => this.renderEntry(t)).join('');
  },

  renderEntry(t) {
    const dir      = t.direction === 'LONG' ? '▲' : '▼';
    const dirClass = t.direction === 'LONG' ? 'text-green' : 'text-red';
    const rml      = t.rMultiple != null ? `<span class="${t.rMultiple > 0 ? 'profit' : 'loss'}">${APEX.fmt.sign(t.rMultiple)}R</span>` : '<span>Open</span>';
    const grade    = t.grade || '';
    const gradeHtml = grade
      ? `<span class="je-grade ${grade}">${grade}</span>`
      : `<span class="badge badge-muted" style="font-size:10px">OPEN</span>`;

    return `
<div class="journal-entry" onclick="Journal.openReview('${t.id}')">
  <div class="je-header">
    <span class="je-ticker">${t.ticker?.toUpperCase() || '—'}</span>
    <span class="badge ${t.direction === 'LONG' ? 'badge-green' : 'badge-red'} btn-sm">${dir} ${t.direction}</span>
    ${gradeHtml}
    <span style="margin-left:auto;font-size:11px;color:var(--text-muted)">${APEX.fmt.date(t.createdAt)}</span>
  </div>
  <div class="je-meta">
    <span>Setup: ${t.setupType || '—'}</span>
    <span>Conviction: ${t.conviction?.total ?? '—'}/10</span>
    <span>Entry: ${APEX.fmt.price(t.entry)}</span>
    <span>SL: ${APEX.fmt.price(t.stopLoss)}</span>
    <span>Result: ${rml}</span>
    ${t.emotion ? `<span>Emotion: ${t.emotion}</span>` : ''}
  </div>
</div>`;
  },

  // ── Open Review Modal ────────────────────────────────────────
  openReview(id) {
    const trade = this.getById(id);
    if (!trade) return;

    const modal = document.getElementById('journal-modal');
    const body  = document.getElementById('journal-modal-body');
    if (!modal || !body) return;

    body.innerHTML = this.renderReviewForm(trade);
    modal.classList.remove('hidden');

    // Bind close grade & R buttons
    body.querySelectorAll('.grade-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        body.querySelectorAll('.grade-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        document.getElementById('jm-grade').value = btn.dataset.grade;
      });
    });

    body.querySelectorAll('.emotion-tag').forEach(tag => {
      tag.addEventListener('click', () => {
        body.querySelectorAll('.emotion-tag').forEach(t => t.classList.remove('selected'));
        tag.classList.add('selected', tag.dataset.type === 'warn' ? 'warn' : tag.dataset.type === 'bad' ? 'bad' : '');
        document.getElementById('jm-emotion').value = tag.dataset.id;
      });
    });

    // Pre-fill existing
    if (trade.grade) {
      body.querySelector(`[data-grade="${trade.grade}"]`)?.classList.add('selected');
    }
    if (trade.emotion) {
      body.querySelector(`[data-id="${trade.emotion}"]`)?.classList.add('selected');
    }
  },

  renderReviewForm(t) {
    return `
<div class="modal-header">
  <h3>📓 Trade Review — ${t.ticker?.toUpperCase()} <span class="badge ${t.direction === 'LONG' ? 'badge-green' : 'badge-red'}">${t.direction}</span></h3>
  <button class="btn btn-secondary btn-sm" onclick="Journal.closeModal()">✕ Close</button>
</div>

<div class="grid-2 gap-3 mb-3">
  <div class="stat-card">
    <div class="sc-label">Entry</div>
    <div class="sc-value" style="font-size:18px">${APEX.fmt.price(t.entry)}</div>
  </div>
  <div class="stat-card">
    <div class="sc-label">Stop Loss</div>
    <div class="sc-value" style="font-size:18px;color:var(--red)">${APEX.fmt.price(t.stopLoss)}</div>
  </div>
</div>

<input type="hidden" id="jm-trade-id" value="${t.id}">
<input type="hidden" id="jm-grade" value="${t.grade || ''}">
<input type="hidden" id="jm-emotion" value="${t.emotion || ''}">

<div class="form-group">
  <label class="form-label">Status</label>
  <select class="form-select" id="jm-status">
    <option value="OPEN"   ${t.status === 'OPEN'   ? 'selected' : ''}>Open</option>
    <option value="CLOSED" ${t.status === 'CLOSED' ? 'selected' : ''}>Closed</option>
    <option value="CANCELLED" ${t.status === 'CANCELLED' ? 'selected' : ''}>Cancelled</option>
  </select>
</div>

<div class="grid-2 gap-3">
  <div class="form-group">
    <label class="form-label">Actual Exit Price</label>
    <input class="form-input" id="jm-exit" type="number" step="0.01" value="${t.exitPrice || ''}" placeholder="0.00">
  </div>
  <div class="form-group">
    <label class="form-label">R-Multiple Result</label>
    <input class="form-input" id="jm-r" type="number" step="0.01" value="${t.rMultiple ?? ''}" placeholder="e.g. 2.5">
  </div>
</div>

<div class="form-group">
  <label class="form-label">Trade Grade</label>
  <div style="display:flex;gap:8px;">
    <button class="btn btn-success grade-btn ${t.grade === 'A' ? 'selected' : ''}" data-grade="A">A — Followed Plan</button>
    <button class="btn btn-secondary grade-btn ${t.grade === 'B' ? 'selected' : ''}" data-grade="B">B — Minor Deviation</button>
    <button class="btn btn-danger grade-btn ${t.grade === 'C' ? 'selected' : ''}" data-grade="C">C — Major Deviation</button>
  </div>
</div>

<div class="form-group">
  <label class="form-label">Emotional State</label>
  <div class="emotion-selector">
    ${Journal.emotions.map(e => `
      <div class="emotion-tag ${t.emotion === e.id ? 'selected' : ''}" data-id="${e.id}" data-type="${e.type}">${e.label}</div>
    `).join('')}
  </div>
</div>

<div class="form-group">
  <label class="form-label">Post-Trade Review Notes</label>
  <textarea class="form-input" id="jm-review" rows="5" placeholder="${Journal.reviewQuestions.map((q, i) => `${i + 1}. ${q}`).join('\n')}" style="resize:vertical;font-family:var(--font-sans);font-size:12px;">${t.reviewNotes || ''}</textarea>
</div>

<div style="display:flex;gap:8px;margin-top:8px;">
  <button class="btn btn-primary flex-1" onclick="Journal.saveReview()">💾 Save Review</button>
  <button class="btn btn-danger" onclick="Journal.confirmDelete('${t.id}')">🗑 Delete</button>
</div>`;
  },

  saveReview() {
    const id     = document.getElementById('jm-trade-id')?.value;
    const status = document.getElementById('jm-status')?.value;
    const exit   = parseFloat(document.getElementById('jm-exit')?.value) || null;
    const rMult  = parseFloat(document.getElementById('jm-r')?.value);
    const grade  = document.getElementById('jm-grade')?.value || null;
    const emotion= document.getElementById('jm-emotion')?.value || null;
    const review = document.getElementById('jm-review')?.value || '';

    this.update(id, {
      status,
      exitPrice:   exit,
      rMultiple:   isNaN(rMult) ? null : rMult,
      grade,
      emotion,
      reviewNotes: review,
      reviewedAt:  new Date().toISOString(),
    });

    this.closeModal();
    this.renderList('journal-list', document.getElementById('journal-filter')?.value || 'ALL');
    App.toast('Trade review saved ✓', 'success');
  },

  confirmDelete(id) {
    if (confirm('Delete this trade from your journal?')) {
      this.remove(id);
      this.closeModal();
      this.renderList('journal-list');
      App.toast('Trade deleted', 'info');
    }
  },

  closeModal() {
    document.getElementById('journal-modal')?.classList.add('hidden');
  },
};
