/* ============================================================
   APEX — Portfolio & Drawdown Manager
   ============================================================ */

const Portfolio = {

  // ── Settings ─────────────────────────────────────────────────
  getSettings() {
    return JSON.parse(localStorage.getItem('apex_settings') || JSON.stringify({
      accountSize:  50000,
      riskPctDefault: 0.02,
      vixHigh:      false,
      currentRegime: 'TRENDING',
      vixLevel:     18.5,
    }));
  },

  saveSettings(s) { localStorage.setItem('apex_settings', JSON.stringify(s)); },

  // ── P&L Snapshots ────────────────────────────────────────────
  getSnapshots() { return JSON.parse(localStorage.getItem('apex_snapshots') || '[]'); },

  addSnapshot(equity) {
    const snaps = this.getSnapshots();
    snaps.push({ equity, date: new Date().toISOString() });
    localStorage.setItem('apex_snapshots', JSON.stringify(snaps.slice(-365))); // keep 1yr
  },

  // ── Equity Curve Data ────────────────────────────────────────
  equityData() {
    const snaps = this.getSnapshots();
    if (!snaps.length) return [];
    return snaps.map(s => s.equity);
  },

  // ── Drawdown Calculation ─────────────────────────────────────
  maxDrawdown() {
    const data = this.equityData();
    if (data.length < 2) return 0;
    let peak = data[0], maxDD = 0;
    for (const v of data) {
      if (v > peak) peak = v;
      const dd = (v - peak) / peak;
      if (dd < maxDD) maxDD = dd;
    }
    return maxDD;
  },

  // ── Daily/Weekly/Monthly PnL from Journal ────────────────────
  periodPnl() {
    const trades  = Journal.getAll().filter(t => t.status === 'CLOSED');
    const settings = this.getSettings();
    const now     = new Date();

    const dayStart   = new Date(now); dayStart.setHours(0,0,0,0);
    const weekStart  = new Date(now); weekStart.setDate(now.getDate() - now.getDay());
    const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);

    const pnlFrom = (from) => trades
      .filter(t => new Date(t.updatedAt || t.createdAt) >= from)
      .reduce((s, t) => s + (t.rMultiple != null ? t.rMultiple * (settings.accountSize * settings.riskPctDefault) : 0), 0);

    return {
      daily:   pnlFrom(dayStart),
      weekly:  pnlFrom(weekStart),
      monthly: pnlFrom(monthStart),
    };
  },

  // ── Open Positions from Journal ───────────────────────────────
  openPositions() { return Journal.getAll().filter(t => t.status === 'OPEN'); },

  // ── Total Open Risk ───────────────────────────────────────────
  totalOpenRisk() {
    const settings = this.getSettings();
    const open     = this.openPositions();
    return open.reduce((s, t) => {
      const ps = APEX.positionSize(
        settings.accountSize,
        settings.riskPctDefault,
        t.entry   || 0,
        t.stopLoss || 0,
        settings.vixHigh
      );
      return s + ps.pctRisk;
    }, 0);
  },

  // ── Sector Exposure ───────────────────────────────────────────
  sectorExposure() {
    const open   = this.openPositions();
    const map    = {};
    open.forEach(t => {
      const sector = t.sector || 'Unknown';
      if (!map[sector]) map[sector] = 0;
      map[sector] += 1; // simplified: 1 position = 2% exposure
    });
    return Object.entries(map).map(([name, count]) => ({
      name,
      exposure: count * 2, // 2% per default position
    }));
  },

  // ── Render Open Positions Table ───────────────────────────────
  renderOpenPositions(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const open     = this.openPositions();
    const settings = this.getSettings();

    if (!open.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="es-icon">📊</div>
          <div class="es-title">No open positions</div>
          <div class="es-text">Add trades via the Trade Analyzer</div>
        </div>`;
      return;
    }

    container.innerHTML = `
<table class="data-table">
  <thead>
    <tr>
      <th>Ticker</th>
      <th>Direction</th>
      <th>Entry</th>
      <th>Stop Loss</th>
      <th>TP1</th>
      <th>Conviction</th>
      <th>Sector</th>
      <th>Risk $</th>
      <th>Action</th>
    </tr>
  </thead>
  <tbody>
    ${open.map(t => {
      const ps = APEX.positionSize(settings.accountSize, settings.riskPctDefault, t.entry || 0, t.stopLoss || 0, settings.vixHigh);
      const dirColor = t.direction === 'LONG' ? 'var(--green)' : 'var(--red)';
      return `
<tr>
  <td class="mono" style="font-weight:700">${t.ticker?.toUpperCase()}</td>
  <td><span class="badge ${t.direction === 'LONG' ? 'badge-green' : 'badge-red'}" style="font-size:10px">${t.direction === 'LONG' ? '▲' : '▼'} ${t.direction}</span></td>
  <td class="mono">${APEX.fmt.price(t.entry)}</td>
  <td class="mono" style="color:var(--red)">${APEX.fmt.price(t.stopLoss)}</td>
  <td class="mono" style="color:var(--green)">${APEX.fmt.price(t.tp1)}</td>
  <td><span class="badge ${t.conviction?.total >= 9 ? 'badge-cyan' : t.conviction?.total >= 7 ? 'badge-green' : 'badge-amber'}">${t.conviction?.total ?? '—'}/10</span></td>
  <td style="color:var(--text-secondary)">${t.sector || '—'}</td>
  <td class="mono" style="color:var(--red)">${APEX.fmt.dollar(ps.dollarRisk)}</td>
  <td>
    <button class="btn btn-secondary btn-sm" onclick="Journal.openReview('${t.id}')">Review</button>
  </td>
</tr>`;
    }).join('')}
  </tbody>
</table>`;
  },

  // ── Render Drawdown Gauges ────────────────────────────────────
  renderDrawdownGauges(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const settings = this.getSettings();
    const pnl      = this.periodPnl();
    const acc      = settings.accountSize;

    const gauges = [
      { label: 'Daily',   pnl: pnl.daily,   limit: APEX.RISK.DAILY_LOSS_LIMIT   * acc },
      { label: 'Weekly',  pnl: pnl.weekly,   limit: APEX.RISK.WEEKLY_LOSS_LIMIT  * acc },
      { label: 'Monthly', pnl: pnl.monthly,  limit: APEX.RISK.MONTHLY_LOSS_LIMIT * acc },
    ];

    container.innerHTML = gauges.map(g => {
      const usedPct = g.limit > 0 ? Math.min(Math.max(-g.pnl / g.limit, 0), 1) : 0;
      const color   = usedPct > 0.8 ? 'red' : usedPct > 0.5 ? 'amber' : 'green';
      const warn    = usedPct >= 1;

      return `
<div class="dg-row">
  <span class="dg-label">${g.label}</span>
  <div class="dg-bar">
    <div class="progress-bar">
      <div class="progress-fill ${color}" style="width:${(usedPct * 100).toFixed(1)}%"></div>
    </div>
  </div>
  <span class="dg-value ${color === 'red' ? 'text-red' : color === 'amber' ? 'text-amber' : 'text-green'}">
    ${warn ? '🛑' : ''} ${(usedPct * 100).toFixed(0)}%
  </span>
</div>`;
    }).join('');
  },
};
