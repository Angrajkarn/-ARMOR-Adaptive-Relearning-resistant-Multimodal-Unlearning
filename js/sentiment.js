/* ============================================================
   APEX — Sentiment & Flow Analyzer Module
   ============================================================ */

const Sentiment = {

  state: {
    newsRating:    'NEUTRAL',
    optionsFlow:   { uoa: false, pcRatio: null, gammaSqueezeRisk: false, darkPool: false },
    shortInterest: null,
    retailFlag:    false,
    direction:     'LONG',
  },

  // ── Render Sentiment Panel ────────────────────────────────────
  render() {
    const container = document.getElementById('sentiment-panel');
    if (!container) return;
    // Rendered via HTML — just update dynamic parts
    this.updateConflictCheck();
    this.updateOptionsAnalysis();
  },

  updateConflictCheck() {
    const result = APEX.sentiment.conflictCheck(this.state.direction, this.state.newsRating);
    const el = document.getElementById('sentiment-conflict');
    if (!el) return;

    const configs = {
      'ALIGNED':  { cls: 'alert-success', icon: '✅', title: 'Aligned',  text: 'Technical direction and sentiment confirm each other. +1 conviction bonus eligible.' },
      'CONFLICT': { cls: 'alert-danger',  icon: '⚠️', title: 'Conflict', text: 'Technical direction conflicts with sentiment. Reduce position size by 50% or skip.' },
      'NEUTRAL':  { cls: 'alert-info',    icon: 'ℹ️', title: 'Neutral',  text: 'Sentiment is neutral — use as secondary confirmation only.' },
    };

    const cfg = configs[result];
    el.className = `alert ${cfg.cls}`;
    el.innerHTML = `
      <div class="alert-icon">${cfg.icon}</div>
      <div class="alert-content">
        <div class="alert-title">Sentiment ${cfg.title}</div>
        <div>${cfg.text}</div>
      </div>`;
  },

  updateOptionsAnalysis() {
    const pcEl = document.getElementById('pc-interpretation');
    if (pcEl && this.state.optionsFlow.pcRatio != null) {
      const interp = APEX.sentiment.pcRatio(this.state.optionsFlow.pcRatio);
      pcEl.className = `badge badge-${interp.color === 'red' ? 'red' : interp.color === 'amber' ? 'amber' : 'muted'}`;
      pcEl.textContent = interp.label;
    }

    const siEl = document.getElementById('si-interpretation');
    if (siEl && this.state.shortInterest != null) {
      const interp = APEX.sentiment.shortInterest(this.state.shortInterest);
      siEl.className = `badge badge-${interp.color === 'amber' ? 'amber' : 'muted'}`;
      siEl.textContent = interp.label;
    }
  },

  setNewsRating(rating) {
    this.state.newsRating = rating;
    document.querySelectorAll('.news-rating-btn').forEach(btn => {
      btn.classList.toggle('selected', btn.dataset.rating === rating);
    });
    this.updateConflictCheck();
  },

  setDirection(dir) {
    this.state.direction = dir;
    this.updateConflictCheck();
  },

  setPcRatio(val) {
    this.state.optionsFlow.pcRatio = val;
    this.updateOptionsAnalysis();
  },

  setShortInterest(val) {
    this.state.shortInterest = val;
    this.updateOptionsAnalysis();
  },

  // ── Generate Sentiment Score Summary ─────────────────────────
  scoreSummary() {
    const { newsRating, optionsFlow, shortInterest, retailFlag } = this.state;
    const level = APEX.sentiment.levels.find(l => l.id === newsRating);
    const score = level?.score ?? 0;

    const signals = [];

    if (newsRating === 'STRONG_BULL' || newsRating === 'STRONG_BEAR') {
      signals.push({ text: `News: ${level.label}`, weight: 'High' });
    }
    if (optionsFlow.uoa) signals.push({ text: 'Unusual Options Activity detected', weight: 'High' });
    if (optionsFlow.gammaSqueezeRisk) signals.push({ text: 'Gamma Squeeze potential', weight: 'Medium' });
    if (optionsFlow.darkPool) signals.push({ text: 'Dark pool block prints detected', weight: 'Medium' });
    if (shortInterest > 0.20) signals.push({ text: 'Short squeeze candidate (>20% float)', weight: 'High' });
    if (retailFlag) signals.push({ text: '⚠️ Elevated retail interest — caution signal', weight: 'Caution' });

    return { score, signals, level };
  },
};
