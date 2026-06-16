/* ============================================================
   APEX — Auto-Config
   Pre-seeds API keys into localStorage so live data starts
   immediately on every page load.
   ============================================================ */

(function() {
  const KEYS = {
    finnhub:      'd8l4ll9r01qut1f8vll0',
    alphaVantage: 'D9BG0YYHV0AHOPCA',
  };

  // Save to localStorage (same format as MarketData.saveKeys uses)
  const existing = JSON.parse(localStorage.getItem('apex_api_keys') || '{}');

  // Only overwrite if not already saved or empty
  if (!existing.finnhub || !existing.alphaVantage) {
    localStorage.setItem('apex_api_keys', JSON.stringify(KEYS));
    console.log('[APEX] API keys auto-configured ✓');
  }
})();
