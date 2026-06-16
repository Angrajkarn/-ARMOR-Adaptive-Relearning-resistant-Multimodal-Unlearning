/* ============================================================
   APEX — App Controller
   Routing, State, Panel Management, UI Bindings
   ============================================================ */

const App = {

  currentPanel: 'dashboard',
  analyzerStep: 0,
  analyzerScores: [0, 0, 0, 0, 0],
  currentTradeDraft: {},
  sentimentBonus: false,

  // Live data state
  tvWidget: null,
  tvInterval: 'D',
  liveTickerData: {},       // symbol -> {price, change, up}
  liveRefreshTimer: null,
  currentAnalyzerTicker: '',

  // ── Init ─────────────────────────────────────────────────────
  init() {
    this.bindNav();
    this.bindAnalyzer();
    this.bindPortfolio();
    this.bindSentiment();
    this.bindJournal();
    this.bindPerformance();
    this.bindSettings();
    this.startClock();
    this.loadDashboard();
    this.loadTickerTape();
    this.navigate('dashboard');
    this.updateMarketStatus();
    setInterval(() => this.updateMarketStatus(), 60_000);

    // Boot live data system
    MarketData.loadKeys();
    this.bindApiKeys();
    if (MarketData.keys.finnhub) {
      this.startLiveFeed();
    }

    // Seed demo data if first launch
    if (!localStorage.getItem('apex_seeded')) {
      this.seedDemoData();
      localStorage.setItem('apex_seeded', '1');
    }
  },

  // ── Navigation ───────────────────────────────────────────────
  bindNav() {
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', () => {
        this.navigate(item.dataset.panel);
      });
    });
  },

  navigate(panelId) {
    this.currentPanel = panelId;

    document.querySelectorAll('.nav-item').forEach(i => i.classList.toggle('active', i.dataset.panel === panelId));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === `panel-${panelId}`));

    const titles = {
      dashboard:   ['Command Center',  'Live Portfolio Overview'],
      analyzer:    ['Trade Analyzer',  '4-Step APEX Analysis Workflow'],
      portfolio:   ['Portfolio Risk',  'Position Sizing & Exposure Manager'],
      sentiment:   ['Sentiment & Flow','Options Flow, News & Social Signals'],
      journal:     ['Trade Journal',   'Post-Trade Review & Psychology Log'],
      performance: ['Performance',     'Analytics, Statistics & Review'],
    };

    const [title, sub] = titles[panelId] || ['APEX', ''];
    document.getElementById('topbar-title').textContent = title;
    document.getElementById('topbar-subtitle').textContent = sub;

    // Panel-specific loads
    if (panelId === 'dashboard')   this.loadDashboard();
    if (panelId === 'portfolio')   this.loadPortfolioPanel();
    if (panelId === 'journal')     this.loadJournalPanel();
    if (panelId === 'performance') this.loadPerformancePanel();
  },

  // ── Clock ─────────────────────────────────────────────────────
  startClock() {
    const update = () => {
      const el = document.getElementById('topbar-clock');
      if (el) el.textContent = APEX.fmt.time();
    };
    update();
    setInterval(update, 1000);
  },

  // ── Toast Notifications ───────────────────────────────────────
  toast(msg, type = 'info') {
    const types = {
      success: { bg: 'var(--green-glow)', border: 'rgba(0,230,118,0.4)', color: 'var(--green)', icon: '✅' },
      error:   { bg: 'var(--red-glow)',   border: 'rgba(255,61,113,0.4)',color: 'var(--red)',   icon: '❌' },
      warning: { bg: 'var(--amber-glow)', border: 'rgba(255,179,0,0.4)', color: 'var(--amber)', icon: '⚠️' },
      info:    { bg: 'var(--cyan-glow)',  border: 'rgba(0,212,255,0.3)', color: 'var(--cyan)',  icon: 'ℹ️' },
    };
    const cfg = types[type] || types.info;

    const el = document.createElement('div');
    el.style.cssText = `
      position:fixed; bottom:24px; right:24px; z-index:9000;
      background:var(--bg-card); border:1px solid ${cfg.border};
      border-radius:10px; padding:12px 18px;
      display:flex; align-items:center; gap:10px;
      font-size:13px; color:${cfg.color};
      box-shadow:0 4px 20px rgba(0,0,0,0.5);
      animation:slideUp 0.3s ease;
      max-width:320px;
    `;
    el.innerHTML = `<span>${cfg.icon}</span><span>${msg}</span>`;
    document.body.appendChild(el);
    setTimeout(() => el.style.opacity = '0', 2800);
    setTimeout(() => el.remove(), 3100);
  },

  // ── Dashboard ─────────────────────────────────────────────────
  loadDashboard() {
    const settings = Portfolio.getSettings();
    const pnl      = Portfolio.periodPnl();
    const stats    = APEX.stats(Journal.getAll()) || {};
    const openRisk = Portfolio.totalOpenRisk();

    // Account equity stat
    this.setEl('dash-account', APEX.fmt.dollar(settings.accountSize));

    // Daily PnL
    const dailyColor = pnl.daily >= 0 ? 'up' : 'down';
    this.setEl('dash-daily-pnl', `${pnl.daily >= 0 ? '+' : ''}${APEX.fmt.dollar(pnl.daily)}`);
    const dpEl = document.getElementById('dash-daily-pnl');
    if (dpEl) dpEl.className = `ts-value ${dailyColor}`;

    // Open risk
    this.setEl('dash-open-risk', APEX.fmt.pct(openRisk));

    // Win rate
    this.setEl('dash-winrate', stats.winRate != null ? `${(stats.winRate * 100).toFixed(0)}%` : '—');

    // Expectancy
    this.setEl('dash-expectancy', stats.expectancy != null ? `${stats.expectancy.toFixed(2)}R` : '—');

    // VIX warning
    const vixEl = document.getElementById('dash-vix-warning');
    if (vixEl) {
      if (settings.vixHigh) {
        vixEl.classList.remove('hidden');
      } else {
        vixEl.classList.add('hidden');
      }
    }

    // Regime
    const regime = APEX.regimes[settings.currentRegime] || APEX.regimes.TRENDING;
    this.setEl('sidebar-regime-value', regime.label);
    const dot = document.querySelector('.regime-dot');
    if (dot) {
      dot.className = `regime-dot ${regime.dot}`;
    }

    // Drawdown gauges
    Portfolio.renderDrawdownGauges('dash-drawdown-gauges');

    // Open positions
    Portfolio.renderOpenPositions('dash-positions');

    // Equity curve chart
    const equityData = Portfolio.equityData();
    const chartData  = equityData.length >= 2 ? equityData : this.demoEquityCurve(settings.accountSize);
    ApexCharts.equityCurve('equity-chart', chartData);

    // Macro events
    this.loadMacroEventsReal();
  },

  demoEquityCurve(base) {
    const data = [base];
    for (let i = 1; i < 30; i++) {
      const prev = data[i - 1];
      const change = (Math.random() - 0.42) * (base * 0.015);
      data.push(Math.max(prev + change, base * 0.8));
    }
    return data;
  },

  setEl(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  },

  // ── Ticker Tape ───────────────────────────────────────────────
  loadTickerTape() {
    const items = [
      { sym: 'SPX',  price: '5,342.18',  chg: '+0.48%', up: true  },
      { sym: 'QQQ',  price: '461.72',    chg: '+0.61%', up: true  },
      { sym: 'DXY',  price: '104.23',    chg: '-0.12%', up: false },
      { sym: 'VIX',  price: '18.45',     chg: '-2.34%', up: false },
      { sym: 'TNX',  price: '4.42%',     chg: '+0.03',  up: true  },
      { sym: 'GLD',  price: '2,347.50',  chg: '+0.22%', up: true  },
      { sym: 'BTC',  price: '67,842',    chg: '+1.87%', up: true  },
      { sym: 'NVDA', price: '891.43',    chg: '+2.15%', up: true  },
      { sym: 'AAPL', price: '189.61',    chg: '-0.32%', up: false },
      { sym: 'TSLA', price: '183.25',    chg: '+3.42%', up: true  },
      { sym: 'WTI',  price: '78.34',     chg: '-0.55%', up: false },
      { sym: 'EUR',  price: '1.0842',    chg: '+0.08%', up: true  },
    ];

    const html = items.map(i => `
<span class="ticker-item" data-live-sym="${i.sym}">
  <span class="ticker-symbol">${i.sym}</span>
  <span class="ticker-price">${i.price}</span>
  <span class="ticker-change ${i.up ? 'up' : 'down'}">${i.chg}</span>
</span>`).join('');

    const inner = document.getElementById('ticker-inner');
    if (inner) inner.innerHTML = html + html; // duplicate for seamless loop
  },

  // ── Trade Analyzer ────────────────────────────────────────────
  bindAnalyzer() {
    // Step navigation
    document.querySelectorAll('.step-tab').forEach((tab, idx) => {
      tab.addEventListener('click', () => this.goAnalyzerStep(idx));
    });

    // Score buttons
    document.querySelectorAll('.score-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const category = parseInt(btn.dataset.category);
        const value    = parseInt(btn.dataset.value);

        // Deselect siblings
        document.querySelectorAll(`.score-btn[data-category="${category}"]`).forEach(b => {
          b.classList.remove('selected');
        });
        btn.classList.add('selected');
        this.analyzerScores[category] = value;
        this.updateConvictionDisplay();
      });
    });

    // Sentiment bonus toggle
    const sbToggle = document.getElementById('sentiment-bonus-toggle');
    if (sbToggle) {
      sbToggle.addEventListener('click', () => {
        sbToggle.classList.toggle('on');
        this.sentimentBonus = sbToggle.classList.contains('on');
        this.updateConvictionDisplay();
      });
    }

    // Generate trade brief button
    const genBtn = document.getElementById('generate-brief-btn');
    if (genBtn) {
      genBtn.addEventListener('click', () => this.generateTradeBrief());
    }

    // Position size live calc
    ['az-account', 'az-risk-pct', 'az-entry', 'az-stoploss'].forEach(id => {
      document.getElementById(id)?.addEventListener('input', () => this.calcPositionSize());
    });

    // Direction buttons
    document.querySelectorAll('.direction-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.direction-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentTradeDraft.direction = btn.dataset.dir;
        Sentiment.setDirection(btn.dataset.dir);
        this.calcPositionSize();
      });
    });

    // Copy brief
    document.getElementById('copy-brief-btn')?.addEventListener('click', () => {
      const text = this.currentTradeDraft._plainText || '';
      navigator.clipboard.writeText(text).then(() => this.toast('Trade Brief copied!', 'success'));
    });

    // Save to journal
    document.getElementById('save-journal-btn')?.addEventListener('click', () => {
      if (this.currentTradeDraft.ticker) {
        Journal.add({ ...this.currentTradeDraft });
        this.toast('Trade saved to journal ✓', 'success');
      } else {
        this.toast('Generate a Trade Brief first', 'warning');
      }
    });

    // Load account from settings
    const settings = Portfolio.getSettings();
    const azAccount = document.getElementById('az-account');
    if (azAccount && !azAccount.value) azAccount.value = settings.accountSize;
    const azRisk = document.getElementById('az-risk-pct');
    if (azRisk && !azRisk.value) azRisk.value = (settings.riskPctDefault * 100).toFixed(1);

    // Live data triggers
    document.getElementById('auto-fetch-btn')?.addEventListener('click', () => this.fetchLiveData());
    document.getElementById('fetch-news-btn')?.addEventListener('click', () => this.fetchLiveNews());
  },

  goAnalyzerStep(step) {
    this.analyzerStep = step;
    document.querySelectorAll('.step-tab').forEach((t, i) => {
      t.classList.toggle('active', i === step);
      if (i < step) t.classList.add('done');
      else t.classList.remove('done');
    });
    document.querySelectorAll('.step-content').forEach((c, i) => {
      c.classList.toggle('active', i === step);
    });
  },

  updateConvictionDisplay() {
    const result = APEX.scoring.calculate(this.analyzerScores, this.sentimentBonus);

    // Score circle
    const numEl = document.getElementById('conviction-number');
    const circEl = document.getElementById('conviction-circle');
    if (numEl) numEl.textContent = result.total;
    if (circEl) circEl.className = `conviction-circle ${result.color}`;

    // Grade
    const gradeEl = document.getElementById('conviction-grade');
    if (gradeEl) {
      gradeEl.textContent = result.grade;
      gradeEl.className = 'badge ' + (result.pass ? 'badge-green' : 'badge-red');
    }

    // Progress bars per category
    result.breakdown.forEach(cat => {
      const barEl = document.getElementById(`score-bar-${cat.id}`);
      if (barEl) {
        const pct = cat.max > 0 ? (cat.score / cat.max) * 100 : 0;
        barEl.style.width = `${pct}%`;
        barEl.className = `progress-fill ${pct === 100 ? 'cyan' : pct >= 50 ? 'green' : pct > 0 ? 'amber' : 'red'}`;
      }
      const valEl = document.getElementById(`score-val-${cat.id}`);
      if (valEl) valEl.textContent = `${cat.score}/${cat.max}`;
    });

    this.currentTradeDraft.conviction = result;
    return result;
  },

  calcPositionSize() {
    const account = parseFloat(document.getElementById('az-account')?.value) || 50000;
    const riskPct = parseFloat(document.getElementById('az-risk-pct')?.value) / 100 || 0.02;
    const entry   = parseFloat(document.getElementById('az-entry')?.value) || 0;
    const sl      = parseFloat(document.getElementById('az-stoploss')?.value) || 0;
    const vix     = Portfolio.getSettings().vixHigh;

    if (!entry || !sl || entry === sl) return;

    const direction = this.currentTradeDraft.direction || 'LONG';
    const ps  = APEX.positionSize(account, riskPct, entry, sl, vix);
    const tps = APEX.tpLevels(entry, sl, direction);

    // Update UI
    this.setEl('ps-shares',   Math.round(ps.shares).toLocaleString());
    this.setEl('ps-dollar',   APEX.fmt.dollar(ps.dollarRisk));
    this.setEl('ps-pct',      APEX.fmt.pct(ps.pctRisk));
    this.setEl('ps-tp1',      APEX.fmt.price(tps.tp1));
    this.setEl('ps-tp2',      APEX.fmt.price(tps.tp2));
    this.setEl('ps-runner',   APEX.fmt.price(tps.runner));
    this.setEl('ps-rr1',      `1:${tps.tp1R.toFixed(1)}`);
    this.setEl('ps-rr2',      `1:${tps.tp2R.toFixed(1)}`);
    this.setEl('ps-rr3',      `1:${tps.tp3R.toFixed(1)}`);

    // Warn if risk > max
    const riskWarnEl = document.getElementById('ps-risk-warn');
    if (riskWarnEl) {
      riskWarnEl.classList.toggle('hidden', ps.pctRisk <= APEX.RISK.MAX_RISK_PCT);
    }

    // Store draft
    Object.assign(this.currentTradeDraft, {
      entry, stopLoss: sl,
      tp1: tps.tp1, tp2: tps.tp2, runner: tps.runner,
      rr1: tps.tp1R, rr2: tps.tp2R,
      shares: ps.shares, dollarRisk: ps.dollarRisk, pctRisk: ps.pctRisk,
    });
  },

  generateTradeBrief() {
    const ticker   = document.getElementById('az-ticker')?.value?.trim().toUpperCase() || 'TICKER';
    const setup    = document.getElementById('az-setup')?.value || 'Custom Setup';
    const tf       = document.getElementById('az-timeframe')?.value || 'Daily';
    const thesis1  = document.getElementById('az-thesis1')?.value?.trim() || '';
    const thesis2  = document.getElementById('az-thesis2')?.value?.trim() || '';
    const thesis3  = document.getElementById('az-thesis3')?.value?.trim() || '';
    const inval    = document.getElementById('az-invalidation')?.value?.trim() || 'Price reclaims key stop level';
    const notes    = document.getElementById('az-notes')?.value?.trim() || '';
    const sector   = document.getElementById('az-sector')?.value || 'Technology';

    // Trigger calc in case not done
    this.calcPositionSize();

    const conviction = this.currentTradeDraft.conviction || APEX.scoring.calculate(this.analyzerScores, this.sentimentBonus);

    if (!conviction.pass) {
      const briefEl = document.getElementById('trade-brief-output');
      if (briefEl) {
        briefEl.innerHTML = `
<div class="decision-banner no-trade">
  🚫 NO-TRADE — Score ${conviction.total}/10 (minimum 7 required)
  <div style="font-size:12px;font-weight:400;margin-top:8px;color:var(--text-secondary)">
    Improve technical setup quality before entering. Weak setups lose money.
  </div>
</div>`;
      }
      return;
    }

    const data = {
      ticker, date: new Date().toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric' }),
      setupType:  setup,
      conviction,
      direction:  this.currentTradeDraft.direction || 'LONG',
      timeframe:  tf,
      entry:      this.currentTradeDraft.entry    || 0,
      stopLoss:   this.currentTradeDraft.stopLoss || 0,
      tp1:        this.currentTradeDraft.tp1      || 0,
      tp2:        this.currentTradeDraft.tp2      || 0,
      runner:     this.currentTradeDraft.runner   || 0,
      rr1:        this.currentTradeDraft.rr1      || 1.5,
      rr2:        this.currentTradeDraft.rr2      || 2.5,
      shares:     this.currentTradeDraft.shares   || 0,
      dollarRisk: this.currentTradeDraft.dollarRisk || 0,
      pctRisk:    this.currentTradeDraft.pctRisk  || 0,
      thesis:     [thesis1, thesis2, thesis3].filter(Boolean),
      invalidation: inval, notes, sector,
    };

    // Store full draft
    Object.assign(this.currentTradeDraft, data);
    this.currentTradeDraft._plainText = APEX.tradeBriefText(data);

    const briefEl = document.getElementById('trade-brief-output');
    if (briefEl) {
      briefEl.innerHTML = `
        <div class="decision-banner pass">✅ PASS — Conviction ${conviction.total}/10 · ${conviction.grade}</div>
        ${APEX.generateTradeBrief(data)}`;
    }

    this.toast('Trade Brief generated!', 'success');
  },

  // ── Portfolio Panel ───────────────────────────────────────────
  loadPortfolioPanel() {
    const settings = Portfolio.getSettings();

    // Fill form
    this.setInputVal('port-account',  settings.accountSize);
    this.setInputVal('port-risk-pct', (settings.riskPctDefault * 100).toFixed(1));
    this.setInputVal('port-vix',      settings.vixLevel);
    this.setInputVal('port-regime',   settings.currentRegime);

    // VIX toggle
    const vixToggle = document.getElementById('vix-toggle');
    if (vixToggle) {
      vixToggle.classList.toggle('on', settings.vixHigh);
    }

    // Sector donut
    const sectors = Portfolio.sectorExposure();
    if (sectors.length) {
      ApexCharts.sectorDonut('sector-donut', sectors);
    }

    // Sector rows
    this.renderSectorRows(sectors);

    // Total open risk
    const totalRisk = Portfolio.totalOpenRisk();
    const riskBar = document.getElementById('total-risk-bar');
    if (riskBar) {
      const pct = Math.min(totalRisk / APEX.RISK.MAX_PORTFOLIO_RISK, 1) * 100;
      riskBar.querySelector('.progress-fill').style.width = `${pct}%`;
    }
    this.setEl('total-risk-pct', APEX.fmt.pct(totalRisk));
    this.setEl('total-risk-max', APEX.fmt.pct(APEX.RISK.MAX_PORTFOLIO_RISK));

    // Drawdown gauges
    Portfolio.renderDrawdownGauges('port-drawdown-gauges');
  },

  bindPortfolio() {
    // Save settings
    document.getElementById('save-settings-btn')?.addEventListener('click', () => {
      const settings = {
        accountSize:    parseFloat(document.getElementById('port-account')?.value) || 50000,
        riskPctDefault: parseFloat(document.getElementById('port-risk-pct')?.value) / 100 || 0.02,
        vixLevel:       parseFloat(document.getElementById('port-vix')?.value) || 18.5,
        currentRegime:  document.getElementById('port-regime')?.value || 'TRENDING',
        vixHigh: document.getElementById('vix-toggle')?.classList.contains('on') || false,
      };
      Portfolio.saveSettings(settings);
      this.toast('Settings saved ✓', 'success');
      this.loadDashboard();
    });

    // VIX toggle
    document.getElementById('vix-toggle')?.addEventListener('click', function() {
      this.classList.toggle('on');
    });

    // Snapshot equity
    document.getElementById('snapshot-btn')?.addEventListener('click', () => {
      const acc = parseFloat(document.getElementById('port-account')?.value) || 50000;
      Portfolio.addSnapshot(acc);
      this.toast('Equity snapshot saved ✓', 'success');
    });

    // Position size calc widget
    ['port-calc-account','port-calc-risk','port-calc-entry','port-calc-sl'].forEach(id => {
      document.getElementById(id)?.addEventListener('input', () => this.calcWidgetPositionSize());
    });
  },

  calcWidgetPositionSize() {
    const account = parseFloat(document.getElementById('port-calc-account')?.value) || 50000;
    const riskPct = parseFloat(document.getElementById('port-calc-risk')?.value) / 100 || 0.02;
    const entry   = parseFloat(document.getElementById('port-calc-entry')?.value) || 0;
    const sl      = parseFloat(document.getElementById('port-calc-sl')?.value) || 0;
    const vix     = Portfolio.getSettings().vixHigh;

    if (!entry || !sl || entry === sl) return;
    const ps = APEX.positionSize(account, riskPct, entry, sl, vix);
    this.setEl('calc-shares',  Math.round(ps.shares).toLocaleString());
    this.setEl('calc-dollar',  APEX.fmt.dollar(ps.dollarRisk));
    this.setEl('calc-pct',     APEX.fmt.pct(ps.pctRisk));
  },

  renderSectorRows(sectors) {
    const container = document.getElementById('sector-rows');
    if (!container) return;
    if (!sectors.length) { container.innerHTML = '<div class="text-muted text-sm">No open positions</div>'; return; }
    const maxExposure = APEX.RISK.MAX_SECTOR_EXPOSURE * 100; // 6%
    container.innerHTML = sectors.map(s => {
      const pct  = Math.min(s.exposure / maxExposure, 1) * 100;
      const over = s.exposure > maxExposure;
      return `
<div class="sector-row">
  <span class="sector-name">${s.name}</span>
  <div class="sector-bar">
    <div class="progress-bar">
      <div class="progress-fill ${over ? 'red' : 'cyan'}" style="width:${pct}%"></div>
    </div>
  </div>
  <span class="sector-pct ${over ? 'text-red' : 'text-cyan'}">${s.exposure.toFixed(1)}%</span>
</div>`;
    }).join('');
  },

  // ── Sentiment Panel ───────────────────────────────────────────
  bindSentiment() {
    // News rating buttons
    document.querySelectorAll('.news-rating-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.news-rating-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        Sentiment.setNewsRating(btn.dataset.rating);
      });
    });

    // Direction
    document.querySelectorAll('.sent-dir-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sent-dir-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        Sentiment.setDirection(btn.dataset.dir);
      });
    });

    // P/C ratio input
    document.getElementById('sent-pc-ratio')?.addEventListener('input', e => {
      Sentiment.setPcRatio(parseFloat(e.target.value) || null);
    });

    // Short interest
    document.getElementById('sent-si')?.addEventListener('input', e => {
      Sentiment.setShortInterest(parseFloat(e.target.value) / 100 || null);
    });

    // Flags
    ['sent-uoa', 'sent-gamma', 'sent-darkpool', 'sent-retail'].forEach(id => {
      document.getElementById(id)?.addEventListener('click', function() {
        this.classList.toggle('on');
        const key = { 'sent-uoa': 'uoa', 'sent-gamma': 'gammaSqueezeRisk', 'sent-darkpool': 'darkPool' }[id];
        if (key) Sentiment.state.optionsFlow[key] = this.classList.contains('on');
        if (id === 'sent-retail') Sentiment.state.retailFlag = this.classList.contains('on');
      });
    });
  },

  // ── Journal Panel ──────────────────────────────────────────────
  loadJournalPanel() {
    Journal.renderList('journal-list', document.getElementById('journal-filter')?.value || 'ALL');
  },

  bindJournal() {
    document.getElementById('journal-filter')?.addEventListener('change', e => {
      Journal.renderList('journal-list', e.target.value);
    });

    document.getElementById('journal-modal-close')?.addEventListener('click', () => {
      Journal.closeModal();
    });

    document.getElementById('export-journal-btn')?.addEventListener('click', () => {
      this.exportJournal();
    });
  },

  exportJournal() {
    const data = Journal.getAll();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = `apex-journal-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    this.toast('Journal exported ✓', 'success');
  },

  // ── Performance Panel ────────────────────────────────────────
  loadPerformancePanel() {
    const trades = Journal.getAll();
    const stats  = APEX.stats(trades);

    if (!stats) {
      this.setEl('perf-winrate',    '—');
      this.setEl('perf-expectancy', '—');
      this.setEl('perf-avg-win',    '—');
      this.setEl('perf-avg-loss',   '—');
      this.setEl('perf-total',      '—');
      this.setEl('perf-total-r',    '—');
      return;
    }

    this.setEl('perf-winrate',    `${(stats.winRate * 100).toFixed(1)}%`);
    this.setEl('perf-expectancy', `${stats.expectancy.toFixed(2)}R`);
    this.setEl('perf-avg-win',    `+${stats.avgWin.toFixed(2)}R`);
    this.setEl('perf-avg-loss',   `-${stats.avgLoss.toFixed(2)}R`);
    this.setEl('perf-total',      `${stats.total} trades`);
    this.setEl('perf-total-r',    `${stats.totalR > 0 ? '+' : ''}${stats.totalR.toFixed(2)}R`);

    // Best / worst
    if (stats.bestTrade) {
      this.setEl('perf-best',  `${stats.bestTrade.ticker?.toUpperCase()} +${stats.bestTrade.rMultiple.toFixed(2)}R`);
    }
    if (stats.worstTrade) {
      this.setEl('perf-worst', `${stats.worstTrade.ticker?.toUpperCase()} ${stats.worstTrade.rMultiple.toFixed(2)}R`);
    }

    // Win rate gauge
    ApexCharts.winRateGauge('winrate-chart', stats.winRate);

    // R-Multiple distribution
    ApexCharts.rMultipleDist('rmult-chart', trades);

    // Equity curve
    const equityData = Portfolio.equityData();
    const base = Portfolio.getSettings().accountSize;
    ApexCharts.equityCurve('perf-equity-chart', equityData.length >= 2 ? equityData : this.demoEquityCurve(base));
  },

  bindPerformance() {
    // Weekly review export
    document.getElementById('weekly-review-btn')?.addEventListener('click', () => {
      this.generateWeeklyReview();
    });
  },

  generateWeeklyReview() {
    const trades = Journal.getAll().filter(t => {
      const weekAgo = new Date(); weekAgo.setDate(weekAgo.getDate() - 7);
      return new Date(t.createdAt) >= weekAgo;
    });
    const stats = APEX.stats(trades.filter(t => t.status === 'CLOSED'));

    const text = `APEX WEEKLY PERFORMANCE REVIEW — ${new Date().toLocaleDateString()}
${'═'.repeat(50)}

TRADES THIS WEEK: ${trades.length}
CLOSED: ${trades.filter(t => t.status === 'CLOSED').length}

${stats ? `WIN RATE:   ${(stats.winRate * 100).toFixed(1)}%
AVG WIN:    +${stats.avgWin.toFixed(2)}R
AVG LOSS:   -${stats.avgLoss.toFixed(2)}R
EXPECTANCY: ${stats.expectancy.toFixed(2)}R
TOTAL R:    ${stats.totalR > 0 ? '+' : ''}${stats.totalR.toFixed(2)}R

BEST TRADE:  ${stats.bestTrade?.ticker} +${stats.bestTrade?.rMultiple.toFixed(2)}R
WORST TRADE: ${stats.worstTrade?.ticker} ${stats.worstTrade?.rMultiple.toFixed(2)}R` : 'No closed trades this week.'}

GRADES:
A (Followed Plan): ${trades.filter(t => t.grade === 'A').length}
B (Minor Deviation): ${trades.filter(t => t.grade === 'B').length}
C (Major Deviation): ${trades.filter(t => t.grade === 'C').length}
`;

    const blob = new Blob([text], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = `apex-weekly-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    this.toast('Weekly review exported ✓', 'success');
  },

  // ── Settings Panel ─────────────────────────────────────────────
  bindSettings() {
    // Reset button
    document.getElementById('reset-all-btn')?.addEventListener('click', () => {
      if (confirm('⚠️ This will erase ALL journal entries and settings. Are you sure?')) {
        localStorage.clear();
        location.reload();
      }
    });
  },

  // ── Market Status ──────────────────────────────────────────
  updateMarketStatus() {
    const status = MarketData.marketStatusLabel();
    const el = document.getElementById('market-status-label');
    if (el) { el.textContent = status.label; el.style.color = status.color; }
  },

  // ── API Key Management ─────────────────────────────────────
  bindApiKeys() {
    // Populate saved keys (masked) into inputs
    const kfh = document.getElementById('key-finnhub');
    const kav = document.getElementById('key-av');
    if (kfh && MarketData.keys.finnhub) kfh.value = MarketData.keys.finnhub;
    if (kav && MarketData.keys.alphaVantage) kav.value = MarketData.keys.alphaVantage;

    if (MarketData.keys.finnhub) {
      this.setEl('key-status', '✅ Keys loaded');
      document.getElementById('key-status').style.color = 'var(--green)';
    }

    document.getElementById('save-keys-btn')?.addEventListener('click', () => {
      const fh = document.getElementById('key-finnhub')?.value?.trim();
      const av = document.getElementById('key-av')?.value?.trim();
      if (!fh) { this.toast('Enter Finnhub API key', 'error'); return; }
      MarketData.saveKeys(fh, av || '');
      this.setEl('key-status', '✅ Saved — Connecting...');
      document.getElementById('key-status').style.color = 'var(--green)';
      this.startLiveFeed();
      this.toast('API keys saved ✓ Connecting WebSocket...', 'success');
    });
  },

  // ── Live Feed: WebSocket + Ticker Tape ──────────────────────
  startLiveFeed() {
    const TICKER_SYMBOLS = ['AAPL','MSFT','NVDA','TSLA','AMZN','GOOGL','META','SPY','QQQ','BTC-USD','ETH-USD'];

    MarketData.connectWebSocket(TICKER_SYMBOLS, (symbol, price, volume) => {
      // Update ticker tape with live price
      if (!this.liveTickerData[symbol]) this.liveTickerData[symbol] = {};
      const prev = this.liveTickerData[symbol].price;
      this.liveTickerData[symbol].price = price;
      this.liveTickerData[symbol].up = prev == null ? true : price >= prev;
      this.liveTickerData[symbol].live = true;
      this.updateTickerItem(symbol, price);

      // Update open positions P&L if symbol matches
      this.updatePositionLivePrice(symbol, price);
    });

    // Also fetch VIX and SPX live quotes via REST as fallback
    this.fetchLiveTickerQuotes();
  },

  async fetchLiveTickerQuotes() {
    if (!MarketData.keys.finnhub) return;
    const symbols = ['^GSPC','^VIX','SPY','QQQ','NVDA','AAPL','TSLA','BTC-USD','GLD'];
    try {
      const quotes = await MarketData.getMultipleQuotes(symbols);
      symbols.forEach(sym => {
        const q = quotes[sym];
        if (q && q.c) {
          this.liveTickerData[sym] = MarketData.fmtQuote(q);
        }
      });
      this.refreshTickerTapeUI();

      // Update VIX in settings if live
      const vix = quotes['^VIX'];
      if (vix?.c) {
        const portVix = document.getElementById('port-vix');
        if (portVix) portVix.value = vix.c.toFixed(2);
        // Auto set high-VIX mode
        const isHigh = vix.c > 25;
        const toggle = document.getElementById('vix-toggle');
        if (toggle) toggle.classList.toggle('on', isHigh);
      }
    } catch(e) { console.warn('[APEX] Ticker quotes failed:', e.message); }
  },

  updateTickerItem(symbol, price) {
    const el = document.querySelector(`[data-live-sym="${symbol}"]`);
    if (!el) return;
    const d = this.liveTickerData[symbol];
    el.querySelector('.ticker-price').textContent = typeof price === 'number' ? price.toFixed(price > 100 ? 2 : 4) : price;
    const changeEl = el.querySelector('.ticker-change');
    if (d.change) {
      changeEl.textContent = d.change;
      changeEl.className = `ticker-change ${d.up ? 'up' : 'down'}`;
    }
    // Flash animation
    el.style.background = d.up ? 'rgba(0,230,118,0.08)' : 'rgba(255,61,113,0.08)';
    setTimeout(() => el.style.background = '', 400);
  },

  refreshTickerTapeUI() {
    this.loadTickerTape();
  },

  updatePositionLivePrice(symbol, price) {
    // Flash live P&L on open positions
    const container = document.getElementById('dash-positions');
    if (!container) return;
    const rows = container.querySelectorAll(`[data-position-ticker="${symbol.replace('-USD','').toUpperCase()}"]`);
    rows.forEach(row => {
      const entry = parseFloat(row.dataset.entry);
      if (!entry) return;
      const pnlEl = row.querySelector('.pos-live-pnl');
      if (pnlEl) {
        const pct = ((price - entry) / entry * 100).toFixed(2);
        pnlEl.textContent = `${pct >= 0 ? '+' : ''}${pct}%`;
        pnlEl.style.color = pct >= 0 ? 'var(--green)' : 'var(--red)';
      }
    });
  },

  // ── TradingView Widget ──────────────────────────────────────
  initTradingView(symbol, interval = 'D') {
    const container = document.getElementById('tradingview-chart');
    if (!container || typeof TradingView === 'undefined') return;

    container.innerHTML = '';

    const sym = symbol.includes(':') ? symbol :
      symbol.includes('-') ? `CRYPTO:${symbol.replace('-USD','USDT')}` :
      `NASDAQ:${symbol}`;

    try {
      this.tvWidget = new TradingView.widget({
        autosize: true,
        symbol: sym,
        interval: interval,
        timezone: 'America/New_York',
        theme: 'dark',
        style: '1',
        locale: 'en',
        toolbar_bg: '#0d1421',
        enable_publishing: false,
        withdateranges: true,
        allow_symbol_change: false,
        save_image: false,
        container_id: 'tradingview-chart',
        studies: [
          'RSI@tv-basicstudies',
          'MACD@tv-basicstudies',
          'BB@tv-basicstudies',
        ],
        overrides: {
          'paneProperties.background': '#0a0f1a',
          'paneProperties.backgroundType': 'solid',
          'mainSeriesProperties.candleStyle.upColor': '#00e676',
          'mainSeriesProperties.candleStyle.downColor': '#ff3d71',
          'mainSeriesProperties.candleStyle.borderUpColor': '#00e676',
          'mainSeriesProperties.candleStyle.borderDownColor': '#ff3d71',
          'mainSeriesProperties.candleStyle.wickUpColor': '#00e676',
          'mainSeriesProperties.candleStyle.wickDownColor': '#ff3d71',
        },
      });

      // Update badge
      const badge = document.getElementById('tv-symbol-badge');
      if (badge) badge.textContent = symbol.toUpperCase();

    } catch(e) { console.warn('[TV]', e); }
  },

  setTVInterval(interval) {
    this.tvInterval = interval;
    if (this.currentAnalyzerTicker) {
      this.initTradingView(this.currentAnalyzerTicker, interval);
    }
  },

  // ── Auto-Fetch Live Data for Analyzer ───────────────────────
  async fetchLiveData() {
    const ticker = document.getElementById('az-ticker')?.value?.trim().toUpperCase();
    if (!ticker) { this.toast('Enter a ticker symbol first', 'warning'); return; }

    const { finnhub, alphaVantage } = MarketData.hasKeys();
    if (!finnhub && !alphaVantage) {
      this.toast('Add API keys in Portfolio → API Keys section first', 'error');
      this.navigate('portfolio');
      return;
    }

    this.currentAnalyzerTicker = ticker;
    const spinner  = document.getElementById('fetch-spinner');
    const fetchBtn = document.getElementById('auto-fetch-btn');
    const statusEl = document.getElementById('fetch-status');
    const quoteStrip = document.getElementById('live-quote-strip');

    if (spinner)  { spinner.classList.remove('hidden'); }
    if (fetchBtn) { fetchBtn.disabled = true; fetchBtn.textContent = '⏳ Fetching...'; }

    const log = (msg, ok = true) => {
      if (statusEl) {
        statusEl.classList.remove('hidden');
        statusEl.innerHTML = `<div style="font-size:11px;color:${ok ? 'var(--green)' : 'var(--amber)'};">${msg}</div>`;
      }
    };

    const tasks = [];
    const results = {};

    // 1. Finnhub Quote
    if (finnhub) {
      tasks.push(
        MarketData.getQuote(ticker)
          .then(q => {
            results.quote = q;
            log(`✅ Quote: $${q.c?.toFixed(2)} (${q.dp >= 0 ? '+' : ''}${q.dp?.toFixed(2)}%)`);

            // Update live quote strip
            if (quoteStrip) quoteStrip.classList.remove('hidden');
            this.setEl('lq-price',  `$${q.c?.toFixed(2) ?? '—'}`);
            const chEl = document.getElementById('lq-change');
            if (chEl) {
              chEl.textContent = `${q.dp >= 0 ? '+' : ''}${q.dp?.toFixed(2)}%`;
              chEl.style.color = q.dp >= 0 ? 'var(--green)' : 'var(--red)';
            }
            this.setEl('lq-high', `$${q.h?.toFixed(2) ?? '—'}`);
            this.setEl('lq-low',  `$${q.l?.toFixed(2) ?? '—'}`);

            // Pre-fill entry price with current price
            const entryEl = document.getElementById('az-entry');
            if (entryEl && !entryEl.value) entryEl.value = q.c?.toFixed(2);
          })
          .catch(e => log(`⚠️ Quote: ${e.message}`, false))
      );
    }

    // 2. TradingView chart (no API key needed)
    this.goAnalyzerStep(1); // Switch to step 2 to show chart
    setTimeout(() => this.initTradingView(ticker, this.tvInterval), 300);

    // 3. RSI
    if (alphaVantage) {
      tasks.push(
        MarketData.getRSI(ticker)
          .then(r => {
            results.rsi = r;
            const rsiEl = document.getElementById('az-rsi');
            if (rsiEl) rsiEl.value = r.value.toFixed(1);
            const interp = MarketData.interpretRSI(r.value);
            const sigEl = document.getElementById('az-rsi-signal');
            if (sigEl) this.selectOption(sigEl, interp.selectVal);
            log(`✅ RSI(14): ${r.value.toFixed(1)}`);
          })
          .catch(e => log(`⚠️ RSI: ${e.message}`, false))
      );
    }

    // 4. MACD (after slight delay for rate limiting)
    if (alphaVantage) {
      tasks.push(
        (async () => {
          await MarketData._delay(800);
          return MarketData.getMACD(ticker);
        })()
          .then(m => {
            results.macd = m;
            const interp = MarketData.interpretMACD(m);
            const macdEl = document.getElementById('az-macd');
            if (macdEl) this.selectOption(macdEl, interp.selectVal);
            log(`✅ MACD: ${m.macd.toFixed(3)} / Hist: ${m.hist.toFixed(3)}`);
          })
          .catch(e => log(`⚠️ MACD: ${e.message}`, false))
      );
    }

    // 5. ATR
    if (alphaVantage) {
      tasks.push(
        (async () => {
          await MarketData._delay(1600);
          return MarketData.getATR(ticker);
        })()
          .then(a => {
            results.atr = a;
            const atrEl = document.getElementById('az-atr');
            if (atrEl) atrEl.value = a.value.toFixed(2);
            log(`✅ ATR(14): ${a.value.toFixed(2)}`);
          })
          .catch(e => log(`⚠️ ATR: ${e.message}`, false))
      );
    }

    // 6. OBV
    if (alphaVantage) {
      tasks.push(
        (async () => {
          await MarketData._delay(2400);
          return MarketData.getOBV(ticker);
        })()
          .then(o => {
            const interp = MarketData.interpretOBV(o);
            const obvEl = document.getElementById('az-obv');
            if (obvEl) this.selectOption(obvEl, interp.selectVal);
            log(`✅ OBV: ${o.trend}`);
          })
          .catch(e => log(`⚠️ OBV: ${e.message}`, false))
      );
    }

    // Run all in parallel where possible
    await Promise.allSettled(tasks);

    // Final
    if (spinner)  { spinner.classList.add('hidden'); }
    if (fetchBtn) { fetchBtn.disabled = false; fetchBtn.textContent = '✅ Re-Fetch'; }

    // Add live badge to status
    if (statusEl) {
      statusEl.innerHTML += `
        <div style="margin-top:6px;display:flex;align-items:center;gap:8px;">
          <span class="live-data-badge">LIVE DATA</span>
          <span style="font-size:10px;color:var(--text-muted);">${ticker} — ${new Date().toLocaleTimeString()}</span>
        </div>`;
    }

    this.toast(`Live data loaded for ${ticker}`, 'success');
    this.calcPositionSize();
  },

  // Helper: select option in a <select> by text value
  selectOption(selectEl, text) {
    const opts = selectEl.options;
    for (let i = 0; i < opts.length; i++) {
      if (opts[i].text.toLowerCase().includes(text.toLowerCase())) {
        selectEl.selectedIndex = i;
        return;
      }
    }
  },

  // ── Fetch Live News for Sentiment Panel ─────────────────────
  async fetchLiveNews() {
    const ticker = this.currentAnalyzerTicker || document.getElementById('az-ticker')?.value?.trim().toUpperCase();
    if (!ticker) { this.toast('Analyze a ticker first in Trade Analyzer', 'warning'); return; }
    if (!MarketData.keys.finnhub) { this.toast('Finnhub API key required', 'error'); return; }

    const btn = document.getElementById('fetch-news-btn');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Loading...'; }

    try {
      const [news, sentiment] = await Promise.allSettled([
        MarketData.getCompanyNews(ticker),
        MarketData.getNewsSentiment(ticker),
      ]);

      // Render news feed
      const feed = document.getElementById('live-news-feed');
      if (feed && news.value?.length) {
        feed.innerHTML = news.value.map(item => {
          const date = new Date(item.datetime * 1000).toLocaleDateString('en-US', { month:'short', day:'numeric' });
          return `
<div class="news-item" onclick="window.open('${item.url}','_blank')">
  <div class="news-item-headline">${item.headline}</div>
  <div class="news-item-meta">
    <span class="news-item-source">${item.source}</span>
    <span>${date}</span>
  </div>
</div>`;
        }).join('');
      } else if (feed) {
        feed.innerHTML = '<div style="font-size:11px;color:var(--text-muted);padding:8px 0;">No news found for this ticker in the last 7 days.</div>';
      }

      // Render sentiment score bar
      const sentBar = document.getElementById('finnhub-sentiment-bar');
      if (sentBar && sentiment.value?.sentiment) {
        sentBar.classList.remove('hidden');
        const bull = (sentiment.value.sentiment.bullishPercent * 100).toFixed(0);
        const bear = (sentiment.value.sentiment.bearishPercent * 100).toFixed(0);
        const buzz = sentiment.value.buzz?.buzz?.toFixed(2) ?? '—';
        this.setEl('fs-bull', `${bull}%`);
        this.setEl('fs-bear', `${bear}%`);
        this.setEl('fs-buzz', buzz);
        const bar = document.getElementById('fs-bull-bar');
        if (bar) bar.style.width = `${bull}%`;

        // Auto-select sentiment rating
        const rating = MarketData.interpretNewsSentiment(sentiment.value);
        document.querySelectorAll('.news-rating-btn').forEach(b => {
          b.classList.toggle('selected', b.dataset.rating === rating);
        });
        Sentiment.setNewsRating(rating);
      }

      this.toast(`News loaded for ${ticker}`, 'success');
    } catch(e) {
      this.toast(`News fetch error: ${e.message}`, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = '🔄 Fetch Live News'; }
    }
  },

  // ── Live Macro Events from Finnhub ──────────────────────────
  async loadMacroEventsReal() {
    if (!MarketData.keys.finnhub) return;
    try {
      const [eco, earnings] = await Promise.allSettled([
        MarketData.getEconomicCalendar(),
        MarketData.getUpcomingEarnings(14),
      ]);

      const events = [];
      const now = Date.now();

      // Economic events
      (eco.value || []).slice(0, 5).forEach(e => {
        const ms = new Date(e.time || e.date || 0).getTime();
        const daysOut = Math.max(0, Math.round((ms - now) / 86_400_000));
        events.push({ name: e.event || e.country, daysOut, impact: e.impact === 'high' ? 'HIGH' : 'MED' });
      });

      // Earnings (for currently analyzed ticker)
      (earnings.value || []).slice(0, 3).forEach(e => {
        const ms = new Date(e.date).getTime();
        const daysOut = Math.max(0, Math.round((ms - now) / 86_400_000));
        events.push({ name: `${e.symbol} Earnings`, daysOut, impact: 'HIGH' });
      });

      if (!events.length) return;
      events.sort((a, b) => a.daysOut - b.daysOut);

      const container = document.getElementById('dash-events');
      if (!container) return;
      container.innerHTML = events.slice(0, 5).map(ev => {
        const color = ev.impact === 'HIGH' ? 'var(--red)' : ev.daysOut <= 2 ? 'var(--amber)' : 'var(--text-secondary)';
        return `
<div class="event-card">
  <div class="event-countdown" style="color:${color}">${ev.daysOut}d</div>
  <div>
    <div class="event-name">${ev.name}</div>
    <div style="color:${color};font-size:10px">${ev.impact} IMPACT</div>
  </div>
  <span class="badge ${ev.impact === 'HIGH' ? 'badge-red' : 'badge-amber'}" style="margin-left:auto">${ev.impact}</span>
</div>`;
      }).join('');
    } catch(e) { console.warn('[APEX] Macro events error:', e.message); }
  },

  // ── Demo Data ─────────────────────────────────────────────────
  seedDemoData() {
    const base = 50000;
    // Seed equity snapshots
    const dates = Array.from({ length: 20 }, (_, i) => {
      const d = new Date(); d.setDate(d.getDate() - (20 - i));
      return d.toISOString();
    });

    let equity = base;
    dates.forEach(date => {
      equity += (Math.random() - 0.38) * 900;
      Portfolio.addSnapshot(Math.max(equity, base * 0.85));
    });

    // Seed a few demo trades
    Journal.add({
      ticker: 'NVDA', direction: 'LONG', setupType: 'Bull Flag Breakout',
      entry: 820, stopLoss: 800, tp1: 843, tp2: 865, runner: 900,
      sector: 'Technology',
      conviction: { total: 8 }, rMultiple: 2.1, status: 'CLOSED', grade: 'A',
      emotion: 'calm', reviewNotes: '1. Yes valid\n2. Entry was optimal\n3. Yes followed plan\n4. Hit TP2\n5. Calm throughout',
    });

    Journal.add({
      ticker: 'AAPL', direction: 'LONG', setupType: 'EMA Crossover',
      entry: 185, stopLoss: 181, tp1: 191, tp2: 195, runner: 203,
      sector: 'Technology',
      conviction: { total: 7 }, rMultiple: -1, status: 'CLOSED', grade: 'B',
      emotion: 'anxious', reviewNotes: '1. Yes\n2. Entry slightly early\n3. Moved stop too early\n4. Hit stop loss\n5. Anxious after news came out',
    });

    Journal.add({
      ticker: 'SPY', direction: 'SHORT', setupType: 'Head & Shoulders',
      entry: 531, stopLoss: 536, tp1: 523.5, tp2: 518.5, runner: 510,
      sector: 'Indices',
      conviction: { total: 9 }, rMultiple: 3.2, status: 'CLOSED', grade: 'A',
      emotion: 'calm',
    });

    Journal.add({
      ticker: 'TSLA', direction: 'LONG', setupType: 'Demand Zone Bounce',
      entry: 178, stopLoss: 172, tp1: 187, tp2: 193, runner: 205,
      sector: 'Consumer Disc.', conviction: { total: 7 }, status: 'OPEN',
    });
  },

  // ── Helpers ───────────────────────────────────────────────────
  setInputVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
  },
};

// ── Boot ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => App.init());
