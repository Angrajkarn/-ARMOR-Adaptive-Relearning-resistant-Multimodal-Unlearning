/* ============================================================
   APEX — Real-Time Market Data Service
   Sources: Finnhub (WebSocket + REST) + Alpha Vantage (Technicals)
   ============================================================ */

const MarketData = {

  // ── Config ───────────────────────────────────────────────────
  FINNHUB_BASE: 'https://finnhub.io/api/v1',
  AV_BASE:      'https://www.alphavantage.co/query',
  CACHE_TTL:    60_000,   // 1 min default
  LONG_TTL:     300_000,  // 5 min for news/sentiment

  keys: { finnhub: '', alphaVantage: '' },
  cache: {},
  ws: null,
  wsCallbacks: {},
  refreshTimer: null,
  marketStatusEl: null,

  // ── API Key Management ───────────────────────────────────────
  loadKeys() {
    const saved = JSON.parse(localStorage.getItem('apex_api_keys') || '{}');
    this.keys = { finnhub: '', alphaVantage: '', ...saved };
    return this.keys;
  },

  saveKeys(finnhub, alphaVantage) {
    this.keys = { finnhub: finnhub.trim(), alphaVantage: alphaVantage.trim() };
    localStorage.setItem('apex_api_keys', JSON.stringify(this.keys));
  },

  hasKeys() {
    return { finnhub: !!this.keys.finnhub, alphaVantage: !!this.keys.alphaVantage };
  },

  // ── Cache ────────────────────────────────────────────────────
  fromCache(key, ttl = this.CACHE_TTL) {
    const e = this.cache[key];
    return e && Date.now() - e.ts < ttl ? e.data : null;
  },

  toCache(key, data) {
    this.cache[key] = { data, ts: Date.now() };
    return data;
  },

  clearCache() { this.cache = {}; },

  // ── Generic Fetch with Error Handling ───────────────────────
  async apiFetch(url, label = 'API') {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${label} error: HTTP ${res.status}`);
    const data = await res.json();
    if (data['Error Message'] || data['Note']) {
      throw new Error(data['Error Message'] || data['Note'] || 'Rate limit hit');
    }
    return data;
  },

  // ══════════════════════════════════════════════════════════════
  // FINNHUB — Real-time Quotes
  // ══════════════════════════════════════════════════════════════

  async getQuote(symbol) {
    if (!this.keys.finnhub) throw new Error('No Finnhub API key');
    const cached = this.fromCache(`q_${symbol}`, 15_000); // 15s for quotes
    if (cached) return cached;
    const data = await this.apiFetch(
      `${this.FINNHUB_BASE}/quote?symbol=${encodeURIComponent(symbol)}&token=${this.keys.finnhub}`,
      `Quote(${symbol})`
    );
    // {c: current, d: change, dp: %change, h: high, l: low, o: open, pc: prevClose}
    return this.toCache(`q_${symbol}`, data);
  },

  async getMultipleQuotes(symbols) {
    const results = {};
    await Promise.allSettled(
      symbols.map(async sym => {
        try { results[sym] = await this.getQuote(sym); }
        catch(e) { results[sym] = null; }
      })
    );
    return results;
  },

  // ── Finnhub Company Profile ──────────────────────────────────
  async getCompanyProfile(symbol) {
    if (!this.keys.finnhub) return null;
    const cached = this.fromCache(`profile_${symbol}`, 3_600_000);
    if (cached) return cached;
    try {
      const data = await this.apiFetch(
        `${this.FINNHUB_BASE}/stock/profile2?symbol=${symbol}&token=${this.keys.finnhub}`
      );
      return this.toCache(`profile_${symbol}`, data);
    } catch { return null; }
  },

  // ── Finnhub Company News ─────────────────────────────────────
  async getCompanyNews(symbol) {
    if (!this.keys.finnhub) throw new Error('No Finnhub API key');
    const cached = this.fromCache(`news_${symbol}`, this.LONG_TTL);
    if (cached) return cached;
    const to   = new Date().toISOString().split('T')[0];
    const from = new Date(Date.now() - 7 * 86_400_000).toISOString().split('T')[0];
    const data = await this.apiFetch(
      `${this.FINNHUB_BASE}/company-news?symbol=${symbol}&from=${from}&to=${to}&token=${this.keys.finnhub}`
    );
    const news = (Array.isArray(data) ? data : []).slice(0, 12);
    return this.toCache(`news_${symbol}`, news);
  },

  // ── Finnhub News Sentiment ───────────────────────────────────
  async getNewsSentiment(symbol) {
    if (!this.keys.finnhub) throw new Error('No Finnhub API key');
    const cached = this.fromCache(`sent_${symbol}`, this.LONG_TTL);
    if (cached) return cached;
    const data = await this.apiFetch(
      `${this.FINNHUB_BASE}/news-sentiment?symbol=${symbol}&token=${this.keys.finnhub}`
    );
    return this.toCache(`sent_${symbol}`, data);
  },

  // ── Finnhub Earnings Calendar ────────────────────────────────
  async getUpcomingEarnings(daysAhead = 30) {
    if (!this.keys.finnhub) return [];
    const cached = this.fromCache('earnings_cal', 3_600_000);
    if (cached) return cached;
    const from = new Date().toISOString().split('T')[0];
    const to   = new Date(Date.now() + daysAhead * 86_400_000).toISOString().split('T')[0];
    try {
      const data = await this.apiFetch(
        `${this.FINNHUB_BASE}/calendar/earnings?from=${from}&to=${to}&token=${this.keys.finnhub}`
      );
      const list = (data.earningsCalendar || []).slice(0, 15);
      return this.toCache('earnings_cal', list);
    } catch { return []; }
  },

  // ── Finnhub Economic Calendar ────────────────────────────────
  async getEconomicCalendar() {
    if (!this.keys.finnhub) return [];
    const cached = this.fromCache('eco_cal', 3_600_000);
    if (cached) return cached;
    try {
      const data = await this.apiFetch(
        `${this.FINNHUB_BASE}/calendar/economic?token=${this.keys.finnhub}`
      );
      const events = (data.economicCalendar || [])
        .filter(e => e.impact === 'high' || e.impact === 'medium')
        .slice(0, 10);
      return this.toCache('eco_cal', events);
    } catch { return []; }
  },

  // ── Finnhub Stock Metrics ─────────────────────────────────────
  async getMetrics(symbol) {
    if (!this.keys.finnhub) return null;
    const cached = this.fromCache(`metrics_${symbol}`, 3_600_000);
    if (cached) return cached;
    try {
      const data = await this.apiFetch(
        `${this.FINNHUB_BASE}/stock/metric?symbol=${symbol}&metric=all&token=${this.keys.finnhub}`
      );
      return this.toCache(`metrics_${symbol}`, data.metric || {});
    } catch { return null; }
  },

  // ══════════════════════════════════════════════════════════════
  // ALPHA VANTAGE — Technical Indicators
  // ══════════════════════════════════════════════════════════════

  async avFetch(params) {
    if (!this.keys.alphaVantage) throw new Error('No Alpha Vantage API key');
    const url = `${this.AV_BASE}?${new URLSearchParams({ ...params, apikey: this.keys.alphaVantage })}`;
    return this.apiFetch(url, params.function);
  },

  // ── RSI ───────────────────────────────────────────────────────
  async getRSI(symbol, period = 14, interval = 'daily') {
    const cKey = `rsi_${symbol}_${interval}`;
    const cached = this.fromCache(cKey);
    if (cached) return cached;
    const data = await this.avFetch({ function: 'RSI', symbol, interval, time_period: period, series_type: 'close' });
    const vals  = data['Technical Analysis: RSI'];
    if (!vals) throw new Error('RSI data empty');
    const [[date, v]] = Object.entries(vals);
    return this.toCache(cKey, { date, value: parseFloat(v.RSI) });
  },

  // ── MACD ──────────────────────────────────────────────────────
  async getMACD(symbol, interval = 'daily') {
    const cKey = `macd_${symbol}_${interval}`;
    const cached = this.fromCache(cKey);
    if (cached) return cached;
    const data = await this.avFetch({ function: 'MACD', symbol, interval, series_type: 'close' });
    const vals  = data['Technical Analysis: MACD'];
    if (!vals) throw new Error('MACD data empty');
    const [[date, v]] = Object.entries(vals);
    return this.toCache(cKey, {
      date,
      macd:   parseFloat(v.MACD),
      signal: parseFloat(v.MACD_Signal),
      hist:   parseFloat(v.MACD_Hist),
    });
  },

  // ── ATR ───────────────────────────────────────────────────────
  async getATR(symbol, period = 14, interval = 'daily') {
    const cKey = `atr_${symbol}_${interval}`;
    const cached = this.fromCache(cKey);
    if (cached) return cached;
    const data = await this.avFetch({ function: 'ATR', symbol, interval, time_period: period });
    const vals  = data['Technical Analysis: ATR'];
    if (!vals) throw new Error('ATR data empty');
    const [[date, v]] = Object.entries(vals);
    return this.toCache(cKey, { date, value: parseFloat(v.ATR) });
  },

  // ── Single EMA ────────────────────────────────────────────────
  async getEMA(symbol, period, interval = 'daily') {
    const cKey = `ema${period}_${symbol}_${interval}`;
    const cached = this.fromCache(cKey);
    if (cached) return cached;
    const data = await this.avFetch({ function: 'EMA', symbol, interval, time_period: period, series_type: 'close' });
    const vals  = data['Technical Analysis: EMA'];
    if (!vals) throw new Error('EMA data empty');
    const [[, v]] = Object.entries(vals);
    return this.toCache(cKey, parseFloat(v.EMA));
  },

  // ── All 4 EMAs (sequential to avoid rate limits) ─────────────
  async getAllEMAs(symbol, interval = 'daily') {
    const cKey = `emas_${symbol}_${interval}`;
    const cached = this.fromCache(cKey);
    if (cached) return cached;
    const periods = [9, 21, 50, 200];
    const result  = {};
    for (const p of periods) {
      try {
        result[p] = await this.getEMA(symbol, p, interval);
        await this._delay(600); // Alpha Vantage rate limit buffer
      } catch(e) { console.warn(`EMA${p} failed:`, e.message); }
    }
    return this.toCache(cKey, result);
  },

  // ── OBV ───────────────────────────────────────────────────────
  async getOBV(symbol, interval = 'daily') {
    const cKey = `obv_${symbol}_${interval}`;
    const cached = this.fromCache(cKey);
    if (cached) return cached;
    const data = await this.avFetch({ function: 'OBV', symbol, interval });
    const vals  = data['Technical Analysis: OBV'];
    if (!vals) throw new Error('OBV data empty');
    const entries = Object.entries(vals).slice(0, 5);
    const latest  = parseFloat(entries[0][1].OBV);
    const prev    = parseFloat(entries[1][1].OBV);
    const prev2   = parseFloat(entries[2][1].OBV);
    let trend;
    if (latest > prev && prev > prev2) trend = 'rising';
    else if (latest < prev && prev < prev2) trend = 'falling';
    else trend = 'flat';
    return this.toCache(cKey, { value: latest, prev, trend });
  },

  // ── Daily OHLCV ───────────────────────────────────────────────
  async getDailyOHLCV(symbol) {
    const cKey = `ohlcv_${symbol}`;
    const cached = this.fromCache(cKey, this.LONG_TTL);
    if (cached) return cached;
    const data = await this.avFetch({ function: 'TIME_SERIES_DAILY', symbol, outputsize: 'compact' });
    const series = data['Time Series (Daily)'];
    if (!series) throw new Error('OHLCV data empty');
    const result = Object.entries(series)
      .slice(0, 60)
      .reverse()
      .map(([date, v]) => ({
        date,
        open:   parseFloat(v['1. open']),
        high:   parseFloat(v['2. high']),
        low:    parseFloat(v['3. low']),
        close:  parseFloat(v['4. close']),
        volume: parseInt(v['5. volume']),
      }));
    return this.toCache(cKey, result);
  },

  // ══════════════════════════════════════════════════════════════
  // SIGNAL INTERPRETERS
  // ══════════════════════════════════════════════════════════════

  interpretRSI(rsiVal) {
    if (rsiVal <= 30)       return { label: 'Oversold (<30)',             selectVal: 'Oversold (<30)',             bullish: true,  score: 1 };
    if (rsiVal >= 70)       return { label: 'Overbought (>70)',           selectVal: 'Overbought (>70)',           bullish: false, score: 0 };
    if (rsiVal >= 50)       return { label: 'Trending up (mid-range)',    selectVal: 'Neutral / Trending',         bullish: true,  score: 1 };
    if (rsiVal >= 40)       return { label: 'Neutral / Trending',         selectVal: 'Neutral / Trending',         bullish: null,  score: 0 };
    return                           { label: 'Trending down (mid-range)', selectVal: 'Neutral / Trending',         bullish: false, score: 0 };
  },

  interpretMACD(macdData) {
    const { macd, signal, hist } = macdData;
    if (macd > signal && hist > 0)  return { label: 'Bullish crossover (signal line)',  selectVal: 'Bullish crossover (signal line)',  bullish: true };
    if (macd > signal && hist <= 0) return { label: 'Histogram momentum building ↑',  selectVal: 'Histogram momentum building ↑',  bullish: true };
    if (macd < signal && hist < 0)  return { label: 'Bearish crossover (signal line)', selectVal: 'Bearish crossover (signal line)', bullish: false };
    if (macd < signal && hist >= 0) return { label: 'Histogram momentum building ↓', selectVal: 'Histogram momentum building ↓', bullish: false };
    return                                   { label: 'Neutral / Flat',                  selectVal: 'Neutral / Flat',                  bullish: null };
  },

  interpretEMAs(emas, currentPrice) {
    const { 9: e9, 21: e21, 50: e50, 200: e200 } = emas;
    if (!e50 || !e200) return { selectVal: 'Mixed / Consolidating', bullish: null };

    if (e9 && e21 && e9 > e21 && e21 > e50 && e50 > e200)
      return { selectVal: '9 > 21 > 50 > 200 — Full Bull Stack', bullish: true };
    if (currentPrice > e50 && e50 > e200)
      return { selectVal: 'Price > 50 > 200 — Bullish Bias', bullish: true };
    if (e9 && e21 && e9 < e21 && e21 < e50 && e50 < e200)
      return { selectVal: '9 < 21 < 50 < 200 — Full Bear Stack', bullish: false };
    if (currentPrice < e50 && e50 < e200)
      return { selectVal: 'Price < 50 < 200 — Bearish Bias', bullish: false };
    return { selectVal: 'Mixed / Consolidating', bullish: null };
  },

  interpretOBV(obvData) {
    if (obvData.trend === 'rising') return { selectVal: 'Rising (bullish accumulation)', bullish: true };
    if (obvData.trend === 'falling') return { selectVal: 'Falling (distribution)', bullish: false };
    return { selectVal: 'Flat (no conviction)', bullish: null };
  },

  interpretNewsSentiment(sentData) {
    const bull = sentData?.sentiment?.bullishPercent ?? 0.5;
    const bear = sentData?.sentiment?.bearishPercent ?? 0.5;
    if (bull > 0.70) return 'STRONG_BULL';
    if (bull > 0.55) return 'BULL';
    if (bear > 0.70) return 'STRONG_BEAR';
    if (bear > 0.55) return 'BEAR';
    return 'NEUTRAL';
  },

  // ══════════════════════════════════════════════════════════════
  // WEBSOCKET — Real-Time Price Streaming
  // ══════════════════════════════════════════════════════════════

  wsReconnectTimer: null,
  wsSymbols: [],

  connectWebSocket(symbols, onTrade) {
    if (!this.keys.finnhub) return;
    this.wsSymbols = symbols;
    this._openWS(symbols, onTrade);
  },

  _openWS(symbols, onTrade) {
    if (this.ws) {
      try { this.ws.close(); } catch {}
    }

    this.ws = new WebSocket(`wss://ws.finnhub.io?token=${this.keys.finnhub}`);

    this.ws.onopen = () => {
      console.log('[APEX] WebSocket connected');
      symbols.forEach(sym => {
        this.ws.send(JSON.stringify({ type: 'subscribe', symbol: sym }));
      });
      this._updateWSStatus(true);
    };

    this.ws.onmessage = e => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'trade' && Array.isArray(msg.data)) {
          msg.data.forEach(t => {
            if (onTrade) onTrade(t.s, t.p, t.v, t.t);
          });
        }
      } catch {}
    };

    this.ws.onerror = () => this._updateWSStatus(false);
    this.ws.onclose = () => {
      this._updateWSStatus(false);
      // Auto-reconnect after 5s
      clearTimeout(this.wsReconnectTimer);
      this.wsReconnectTimer = setTimeout(() => this._openWS(symbols, onTrade), 5000);
    };
  },

  disconnectWebSocket() {
    clearTimeout(this.wsReconnectTimer);
    if (this.ws) { try { this.ws.close(); } catch {} this.ws = null; }
    this._updateWSStatus(false);
  },

  _updateWSStatus(connected) {
    const dots = document.querySelectorAll('.ws-status-dot');
    dots.forEach(d => {
      d.style.background = connected ? 'var(--green)' : 'var(--red)';
      d.style.boxShadow  = connected ? '0 0 8px var(--green)' : '0 0 8px var(--red)';
    });
    const labels = document.querySelectorAll('.ws-status-label');
    labels.forEach(l => {
      l.textContent = connected ? 'Live' : 'Disconnected';
      l.style.color = connected ? 'var(--green)' : 'var(--red)';
    });
  },

  // ══════════════════════════════════════════════════════════════
  // MARKET STATUS
  // ══════════════════════════════════════════════════════════════

  isMarketOpen() {
    const et    = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const day   = et.getDay();
    const mins  = et.getHours() * 60 + et.getMinutes();
    if (day === 0 || day === 6) return false;
    return mins >= 570 && mins < 960; // 9:30 AM – 4:00 PM ET
  },

  isPreMarket() {
    const et   = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const day  = et.getDay();
    const mins = et.getHours() * 60 + et.getMinutes();
    if (day === 0 || day === 6) return false;
    return mins >= 240 && mins < 570; // 4:00 AM – 9:30 AM ET
  },

  marketStatusLabel() {
    if (this.isMarketOpen())   return { label: 'MARKET OPEN',    color: 'var(--green)' };
    if (this.isPreMarket())    return { label: 'PRE-MARKET',     color: 'var(--amber)' };
    return                            { label: 'MARKET CLOSED',  color: 'var(--red)' };
  },

  // ── Helpers ───────────────────────────────────────────────────
  _delay(ms) { return new Promise(r => setTimeout(r, ms)); },

  // ── Format quote change ───────────────────────────────────────
  fmtQuote(q) {
    if (!q || q.c == null) return { price: '—', change: '—', up: null };
    return {
      price:  q.c.toFixed(2),
      change: `${q.dp >= 0 ? '+' : ''}${q.dp?.toFixed(2) ?? '0.00'}%`,
      up:     q.dp >= 0,
      raw:    q,
    };
  },
};
