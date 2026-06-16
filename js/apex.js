/* ============================================================
   APEX — Core Engine
   Conviction Scoring, Position Sizing, Risk Rules
   ============================================================ */

const APEX = {

  // ── Risk Constants ──────────────────────────────────────────
  RISK: {
    DEFAULT_RISK_PCT:    0.02,   // 2% default
    MAX_RISK_PCT:        0.03,   // 3% absolute max
    MAX_SECTOR_EXPOSURE: 0.06,   // 6% per sector
    MAX_PORTFOLIO_RISK:  0.10,   // 10% total
    VIX_THRESHOLD:       25,     // reduce size by 50%
    MIN_RR:              2.0,    // min 1:2 R:R to enter
    CONVICTION_THRESHOLD: 7,     // min score to PASS
    DAILY_LOSS_LIMIT:    0.03,   // 3%
    WEEKLY_LOSS_LIMIT:   0.05,   // 5%
    MONTHLY_LOSS_LIMIT:  0.10,   // 10%
    CONSECUTIVE_LOSS_PAUSE: 3,   // pause after 3 losses

    // TP levels
    TP1_R: 1.5,  TP1_SIZE: 0.45,  // 45% at 1.5R
    TP2_R: 2.5,  TP2_SIZE: 0.30,  // 30% at 2.5R
    TP3_R: 4.0,                    // runner at 4R+

    // Stop ATR buffer
    MIN_ATR_BUFFER: 0.5,
    BREAKEVEN_ACTIVATION: 1.5,   // move stop to BE at 1.5R
  },

  // ── Conviction Scoring Engine ───────────────────────────────
  scoring: {
    categories: [
      { id: 'trend',   label: 'Trend Alignment',      max: 2 },
      { id: 'sr',      label: 'S/R Quality',          max: 2 },
      { id: 'pattern', label: 'Pattern Clarity',      max: 2 },
      { id: 'volume',  label: 'Volume Confirmation',  max: 2 },
      { id: 'rr',      label: 'Risk/Reward Ratio',    max: 2 },
    ],

    calculate(scores, sentimentBonus = false) {
      let total = scores.reduce((s, v) => s + v, 0);
      if (sentimentBonus) total = Math.min(10, total + 1);

      return {
        total,
        breakdown: this.categories.map((cat, i) => ({
          ...cat,
          score: scores[i] ?? 0
        })),
        pass:       total >= APEX.RISK.CONVICTION_THRESHOLD,
        sentimentBonus,
        grade:      this.grade(total),
        color:      this.color(total),
      };
    },

    grade(score) {
      if (score >= 9)  return 'ELITE';
      if (score >= 7)  return 'PASS';
      if (score >= 5)  return 'MARGINAL';
      return 'NO-TRADE';
    },

    color(score) {
      if (score >= 9)  return 'score-elite';
      if (score >= 7)  return 'score-high';
      if (score >= 5)  return 'score-mid';
      return 'score-low';
    }
  },

  // ── Position Sizing ─────────────────────────────────────────
  positionSize(accountSize, riskPct, entry, stopLoss, vixHigh = false) {
    let effectiveRisk = Math.min(riskPct, this.RISK.MAX_RISK_PCT);
    if (vixHigh) effectiveRisk *= 0.5;

    const dollarRisk    = accountSize * effectiveRisk;
    const pointRisk     = Math.abs(entry - stopLoss);
    const shares        = pointRisk > 0 ? Math.floor(dollarRisk / pointRisk) : 0;
    const actualDollar  = shares * pointRisk;
    const actualPct     = accountSize > 0 ? actualDollar / accountSize : 0;

    return {
      shares,
      dollarRisk:  actualDollar,
      pctRisk:     actualPct,
      effectiveRisk,
      pointRisk,
      vixReduced:  vixHigh,
    };
  },

  // ── Take Profit Calculator ───────────────────────────────────
  tpLevels(entry, stopLoss, direction = 'LONG') {
    const risk = Math.abs(entry - stopLoss);
    const dir  = direction === 'LONG' ? 1 : -1;

    return {
      tp1:    entry + dir * risk * this.RISK.TP1_R,
      tp2:    entry + dir * risk * this.RISK.TP2_R,
      runner: entry + dir * risk * this.RISK.TP3_R,
      tp1R:   this.RISK.TP1_R,
      tp2R:   this.RISK.TP2_R,
      tp3R:   this.RISK.TP3_R,
      risk,
    };
  },

  // ── R:R Calculator ──────────────────────────────────────────
  rrRatio(entry, stopLoss, target, direction = 'LONG') {
    const risk   = Math.abs(entry - stopLoss);
    const reward = direction === 'LONG'
      ? target - entry
      : entry - target;
    return risk > 0 ? reward / risk : 0;
  },

  // ── ATR-based Stop ──────────────────────────────────────────
  atrStop(entry, atr, direction = 'LONG', multiplier = 1.5) {
    const buffer = atr * multiplier;
    return direction === 'LONG'
      ? entry - buffer
      : entry + buffer;
  },

  // ── Drawdown Check ──────────────────────────────────────────
  drawdownStatus(dailyPnl, weeklyPnl, monthlyPnl, accountSize) {
    const d = dailyPnl   / accountSize;
    const w = weeklyPnl  / accountSize;
    const m = monthlyPnl / accountSize;

    return {
      daily:   { pct: d, limit: this.RISK.DAILY_LOSS_LIMIT,   halt: d <= -this.RISK.DAILY_LOSS_LIMIT },
      weekly:  { pct: w, limit: this.RISK.WEEKLY_LOSS_LIMIT,  halt: w <= -this.RISK.WEEKLY_LOSS_LIMIT },
      monthly: { pct: m, limit: this.RISK.MONTHLY_LOSS_LIMIT, halt: m <= -this.RISK.MONTHLY_LOSS_LIMIT },
      any:     d <= -this.RISK.DAILY_LOSS_LIMIT || w <= -this.RISK.WEEKLY_LOSS_LIMIT || m <= -this.RISK.MONTHLY_LOSS_LIMIT,
    };
  },

  // ── Market Regime ───────────────────────────────────────────
  regimes: {
    TRENDING:   { label: 'Trending',   class: 'trending',   dot: '' },
    RANGING:    { label: 'Ranging',    class: 'ranging',    dot: 'ranging' },
    VOLATILE:   { label: 'Volatile',   class: 'volatile',   dot: 'volatile' },
    LOW_VOL:    { label: 'Low-Vol',    class: 'low-vol',    dot: 'low-vol' },
  },

  // ── Sentiment Engine ────────────────────────────────────────
  sentiment: {
    levels: [
      { id: 'STRONG_BULL', label: 'Strong Bullish', short: 'STRONG BULL', score: 2,  badgeClass: 'badge-strong-bull' },
      { id: 'BULL',        label: 'Bullish',        short: 'BULL',        score: 1,  badgeClass: 'badge-bull' },
      { id: 'NEUTRAL',     label: 'Neutral',        short: 'NEUTRAL',     score: 0,  badgeClass: 'badge-neutral-s' },
      { id: 'BEAR',        label: 'Bearish',        short: 'BEAR',        score: -1, badgeClass: 'badge-bear' },
      { id: 'STRONG_BEAR', label: 'Strong Bearish', short: 'STRONG BEAR', score: -2, badgeClass: 'badge-strong-bear' },
    ],

    // If tech is bullish but sentiment is strongly bearish → warn
    conflictCheck(techDirection, sentimentId) {
      const bullish = ['STRONG_BULL','BULL'];
      const bearish  = ['STRONG_BEAR','BEAR'];
      if (techDirection === 'LONG'  && bearish.includes(sentimentId))  return 'CONFLICT';
      if (techDirection === 'SHORT' && bullish.includes(sentimentId))  return 'CONFLICT';
      if (
        (techDirection === 'LONG'  && bullish.includes(sentimentId)) ||
        (techDirection === 'SHORT' && bearish.includes(sentimentId))
      ) return 'ALIGNED';
      return 'NEUTRAL';
    },

    // P/C ratio interpretation
    pcRatio(ratio) {
      if (ratio > 1.2) return { label: 'Bearish Pressure', color: 'red' };
      if (ratio < 0.7) return { label: 'Bullish Extreme (Contrarian)', color: 'amber' };
      return { label: 'Neutral', color: 'secondary' };
    },

    // Short interest interpretation
    shortInterest(pct) {
      if (pct > 0.20) return { label: 'Short Squeeze Candidate', color: 'amber', flag: true };
      if (pct > 0.10) return { label: 'Elevated Short Interest', color: 'amber', flag: false };
      return { label: 'Normal', color: 'secondary', flag: false };
    },
  },

  // ── Performance Statistics ──────────────────────────────────
  stats(trades) {
    if (!trades.length) return null;
    const closed = trades.filter(t => t.status === 'CLOSED' && t.rMultiple != null);
    if (!closed.length) return null;

    const winners   = closed.filter(t => t.rMultiple > 0);
    const losers    = closed.filter(t => t.rMultiple <= 0);
    const winRate   = winners.length / closed.length;
    const avgWin    = winners.length ? winners.reduce((s,t) => s + t.rMultiple, 0) / winners.length : 0;
    const avgLoss   = losers.length  ? Math.abs(losers.reduce((s,t) => s + t.rMultiple, 0) / losers.length) : 0;
    const expectancy = (winRate * avgWin) - ((1 - winRate) * avgLoss);
    const totalR    = closed.reduce((s,t) => s + (t.rMultiple || 0), 0);

    return {
      total:    closed.length,
      winners:  winners.length,
      losers:   losers.length,
      winRate,
      avgWin,
      avgLoss,
      expectancy,
      totalR,
      bestTrade:  closed.reduce((b,t) => t.rMultiple > (b?.rMultiple ?? -Infinity) ? t : b, null),
      worstTrade: closed.reduce((w,t) => t.rMultiple < (w?.rMultiple ??  Infinity) ? t : w, null),
    };
  },

  // ── Formatting Helpers ──────────────────────────────────────
  fmt: {
    price:  v => v == null ? '—' : `$${Number(v).toFixed(2)}`,
    pct:    v => v == null ? '—' : `${(v * 100).toFixed(2)}%`,
    r:      v => v == null ? '—' : `${Number(v).toFixed(2)}R`,
    dollar: v => v == null ? '—' : `$${Number(v).toLocaleString('en-US', { minimumFractionDigits: 0 })}`,
    shares: v => v == null ? '—' : `${Math.round(v).toLocaleString()} shares`,
    sign:   v => v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2),
    date:   d => new Date(d).toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric' }),
    time:   () => {
      const n = new Date();
      return n.toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit', second:'2-digit', hour12: false });
    },
  },

  // ── Pattern Names ───────────────────────────────────────────
  patterns: [
    'Bull Flag','Bear Flag','Bull Pennant','Bear Pennant',
    'Ascending Triangle','Descending Triangle','Symmetrical Triangle',
    'Cup & Handle','Inverse Cup & Handle',
    'Head & Shoulders','Inverse H&S',
    'Double Top','Double Bottom',
    'Rising Wedge','Falling Wedge',
    'Bullish Engulfing','Bearish Engulfing',
    'Morning Star','Evening Star',
    'Pin Bar Long','Pin Bar Short',
    'Inside Bar Breakout','VWAP Reclaim',
    'Breakout Retest','Demand Zone Bounce',
    'Supply Zone Rejection','EMA Crossover',
    'Custom / Other',
  ],

  // ── Sector List ─────────────────────────────────────────────
  sectors: [
    'Technology','Healthcare','Financials','Energy','Industrials',
    'Consumer Disc.','Consumer Staples','Materials','Real Estate',
    'Utilities','Communication','Crypto','Forex','Commodities',
  ],

  // ── Macro Events ────────────────────────────────────────────
  macroEvents: [
    { name: 'FOMC Meeting',     impact: 'HIGH',   desc: 'Federal Reserve interest rate decision' },
    { name: 'CPI Release',      impact: 'HIGH',   desc: 'Consumer Price Index inflation data' },
    { name: 'PCE Inflation',    impact: 'HIGH',   desc: 'Fed\'s preferred inflation gauge' },
    { name: 'NFP / Jobs',       impact: 'HIGH',   desc: 'Non-Farm Payroll employment data' },
    { name: 'FOMC Minutes',     impact: 'MEDIUM', desc: 'Fed meeting minutes release' },
    { name: 'GDP Release',      impact: 'MEDIUM', desc: 'Gross Domestic Product data' },
    { name: 'PPI Data',         impact: 'MEDIUM', desc: 'Producer Price Index' },
    { name: 'Retail Sales',     impact: 'MEDIUM', desc: 'Consumer spending tracker' },
    { name: 'Earnings — Q',     impact: 'HIGH',   desc: 'Major earnings release' },
    { name: 'FDA Decision',     impact: 'HIGH',   desc: 'Biotech/pharma catalyst' },
    { name: 'Custom Event',     impact: 'CUSTOM', desc: '' },
  ],

  // ── EMA Alignment ───────────────────────────────────────────
  emaAlignments: [
    { id: 'full_bull', label: '9 > 21 > 50 > 200 (Full Bull)',  score: 2, direction: 'LONG' },
    { id: 'bull',      label: 'Price > 50 > 200 (Bullish)',     score: 1, direction: 'LONG' },
    { id: 'neutral',   label: 'Mixed / Consolidating',          score: 0, direction: 'NEUTRAL' },
    { id: 'bear',      label: 'Price < 50 < 200 (Bearish)',     score: 1, direction: 'SHORT' },
    { id: 'full_bear', label: '9 < 21 < 50 < 200 (Full Bear)', score: 2, direction: 'SHORT' },
  ],

  // ── Generate Trade Brief (HTML) ─────────────────────────────
  generateTradeBrief(data) {
    const {
      ticker, date, setupType, conviction, direction,
      timeframe, entry, stopLoss, tp1, tp2, runner,
      shares, dollarRisk, pctRisk,
      thesis, invalidation, notes,
      rr1, rr2
    } = data;

    const dirClass = direction === 'LONG' ? 'long' : 'short';
    const dirIcon  = direction === 'LONG' ? '▲' : '▼';
    const rating   = APEX.scoring.grade(conviction.total);
    const passClass= conviction.pass ? 'badge-green' : 'badge-red';

    return `
<div class="trade-brief">
  <div class="trade-brief-header">
    <span class="trade-brief-title">⚡ TRADE BRIEF — ${ticker.toUpperCase()} ${date}</span>
    <span class="badge ${passClass}">${rating}</span>
  </div>
  <div class="trade-brief-body">
    <div class="tb-row"><span class="tb-key">SETUP TYPE</span><span class="tb-val">${setupType}</span></div>
    <div class="tb-row"><span class="tb-key">CONVICTION</span><span class="tb-val">${conviction.total}/10 — ${conviction.breakdown.map(b=>`${b.label}: ${b.score}/${b.max}`).join(', ')}</span></div>
    <div class="tb-row"><span class="tb-key">DIRECTION</span><span class="tb-val ${dirClass}">${dirIcon} ${direction}</span></div>
    <div class="tb-row"><span class="tb-key">TIMEFRAME</span><span class="tb-val">${timeframe}</span></div>

    <div class="divider"></div>

    <div class="tb-row"><span class="tb-key">ENTRY</span><span class="tb-val price">${APEX.fmt.price(entry)}</span></div>
    <div class="tb-row"><span class="tb-key">STOP LOSS</span><span class="tb-val" style="color:var(--red)">${APEX.fmt.price(stopLoss)} (${APEX.fmt.pct(Math.abs(stopLoss - entry) / entry)} away)</span></div>
    <div class="tb-row"><span class="tb-key">TP1</span><span class="tb-val" style="color:var(--green)">${APEX.fmt.price(tp1)} | R:R = 1:${Number(rr1).toFixed(1)}</span></div>
    <div class="tb-row"><span class="tb-key">TP2</span><span class="tb-val" style="color:var(--green)">${APEX.fmt.price(tp2)} | R:R = 1:${Number(rr2).toFixed(1)}</span></div>
    <div class="tb-row"><span class="tb-key">RUNNER TP</span><span class="tb-val" style="color:var(--cyan)">${APEX.fmt.price(runner)} | Trailing stop after 1.5R</span></div>

    <div class="divider"></div>

    <div class="tb-row"><span class="tb-key">POSITION SIZE</span><span class="tb-val">${APEX.fmt.shares(shares)}</span></div>
    <div class="tb-row"><span class="tb-key">ACCOUNT RISK</span><span class="tb-val">${APEX.fmt.pct(pctRisk)} = ${APEX.fmt.dollar(dollarRisk)}</span></div>

    <div class="trade-brief-section">
      <h4>THESIS</h4>
      ${thesis.map(t => `<div class="thesis-point">${t}</div>`).join('')}
    </div>

    <div class="trade-brief-section">
      <h4>NOTES</h4>
      <div style="font-size:11px;color:var(--text-secondary);">${notes || 'No additional notes.'}</div>
    </div>
  </div>
  <div class="trade-brief-footer">
    <div class="invalidation-text">⚠ INVALIDATION: ${invalidation}</div>
  </div>
</div>`;
  },
};

// ── Generate Trade Brief (plain text for clipboard) ──────────
APEX.tradeBriefText = function(data) {
  const { ticker, date, setupType, conviction, direction, timeframe,
    entry, stopLoss, tp1, tp2, runner, shares, dollarRisk, pctRisk,
    thesis, invalidation, notes, rr1, rr2 } = data;

  return `━━━━━━━━━━━━━━━━━━━━━━━━
TRADE BRIEF — ${ticker.toUpperCase()} ${date}
━━━━━━━━━━━━━━━━━━━━━━━━
SETUP TYPE: ${setupType}
CONVICTION: ${conviction.total}/10 — ${APEX.scoring.grade(conviction.total)}
DIRECTION:  ${direction}
TIMEFRAME:  ${timeframe}

ENTRY:      ${APEX.fmt.price(entry)}
STOP LOSS:  ${APEX.fmt.price(stopLoss)} (${APEX.fmt.pct(Math.abs(stopLoss - entry) / entry)} away)
TP1:        ${APEX.fmt.price(tp1)} | R:R = 1:${Number(rr1).toFixed(1)}
TP2:        ${APEX.fmt.price(tp2)} | R:R = 1:${Number(rr2).toFixed(1)}
RUNNER TP:  ${APEX.fmt.price(runner)} | Trailing stop after 1.5R

POSITION SIZE:  ${APEX.fmt.shares(shares)}
ACCOUNT RISK:   ${APEX.fmt.pct(pctRisk)} = ${APEX.fmt.dollar(dollarRisk)}

THESIS:
${thesis.map(t => `• ${t}`).join('\n')}

INVALIDATION: ${invalidation}
NOTES: ${notes || 'None.'}
━━━━━━━━━━━━━━━━━━━━━━━━`;
};
