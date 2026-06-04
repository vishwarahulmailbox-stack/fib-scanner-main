"""
╔══════════════════════════════════════════════════════════════╗
║     FIBONACCI RETRACEMENT SCANNER — SMC Strategy            ║
║     61%–79% Entry Zone + FVG Confluence                      ║
║     Daily Timeframe | yfinance | HTML Report Output          ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
  1. pip install yfinance pandas numpy
  2. Put your CSV file (column name: "Symbol") in same folder
  3. python fib_scanner.py
  4. Open: fib_report.html in browser

CSV FORMAT:
  Symbol
  RELIANCE.NS
  TCS.NS
  INFY.NS
  ...
  (Use .NS suffix for NSE stocks, .BO for BSE)

For educational purposes only. Not financial advice.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG — Change these if needed
# ─────────────────────────────────────────────
CSV_FILE       = "nse_stocks.csv"       # Your CSV filename
SYMBOL_COLUMN  = "Symbol"           # Column name in CSV
OUTPUT_FILE    = "index.html"       # Output HTML report
PERIOD         = "6mo"              # Data period: 6mo, 1y, etc.
INTERVAL       = "1d"               # Timeframe: 1d (Daily)
SWING_WINDOW   = 10                 # Candles for swing detection
UPTREND_LOOKBACK = 40               # Candles to check uptrend
MIN_RR         = 2.0                # Minimum Risk:Reward ratio
SL_BUFFER_PCT  = 0.003              # 0.3% buffer below Swing Low
# ─────────────────────────────────────────────


def find_swing_points(df, window=SWING_WINDOW):
    """
    Find the most recent valid Swing Low → Swing High sequence.
    Swing High = recent peak. Swing Low = lowest point before that peak.
    Returns: (swing_low, swing_high, sl_idx, sh_idx)
    """
    highs = df['High'].values
    lows  = df['Low'].values
    n     = len(highs)

    # Look at last 60 candles for swing high
    lookback = min(60, n)
    recent_highs = highs[-lookback:]
    sh_rel_idx   = np.argmax(recent_highs)
    sh_idx       = n - lookback + sh_rel_idx
    sh_price     = highs[sh_idx]

    # Swing low = lowest point in window BEFORE swing high
    search_start = max(0, sh_idx - 60)
    sl_rel_idx   = np.argmin(lows[search_start:sh_idx])
    sl_idx       = search_start + sl_rel_idx
    sl_price     = lows[sl_idx]

    return sl_price, sh_price, sl_idx, sh_idx


def check_uptrend(df, lookback=UPTREND_LOOKBACK):
    """
    Uptrend = Higher Highs + Higher Lows + price above 20MA.
    Returns (bool, reason_string)
    """
    if len(df) < lookback:
        return False, "Not enough data"

    closes = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values

    ma20          = np.mean(closes[-20:])
    current_price = closes[-1]
    above_ma      = current_price > ma20

    mid = lookback // 2
    fh_high = np.max(highs[-lookback : -mid])
    sh_high = np.max(highs[-mid:])
    fh_low  = np.min(lows[-lookback : -mid])
    sh_low  = np.min(lows[-mid:])

    hh = sh_high > fh_high
    hl = sh_low  > fh_low

    reasons = []
    if not hh:      reasons.append("No HH")
    if not hl:      reasons.append("No HL")
    if not above_ma: reasons.append("Below MA20")

    if hh and hl and above_ma:
        return True, "HH + HL + Above MA20"
    return False, " | ".join(reasons)


def find_fvg_in_zone(df, zone_low, zone_high, from_idx=0):
    """
    Scan for Bullish Fair Value Gap that overlaps with Fib zone.
    FVG = 3-candle pattern where gap exists between candle[i-1].High and candle[i+1].Low
    Returns: (fvg_found: bool, fvg_mid_price: float or None)
    """
    sub = df.iloc[from_idx:].reset_index(drop=True)
    best_fvg = None

    for i in range(1, len(sub) - 1):
        gap_low  = sub.iloc[i - 1]['High']
        gap_high = sub.iloc[i + 1]['Low']

        if gap_high > gap_low:                      # Valid bullish gap
            gap_mid = (gap_low + gap_high) / 2
            # Check overlap with fibonacci zone
            overlap = not (gap_high < zone_low or gap_low > zone_high)
            if overlap:
                best_fvg = gap_mid                  # Take most recent

    return (best_fvg is not None), best_fvg


def check_bullish_pa(df):
    """
    Check last candle for bullish confirmation pattern:
    Hammer | Bullish Engulfing | Pin Bar
    Returns: (is_valid: bool, pattern_name: str)
    """
    if len(df) < 2:
        return False, "Not enough candles"

    c  = df.iloc[-1]   # Current candle
    p  = df.iloc[-2]   # Previous candle

    body        = c['Close'] - c['Open']
    total_range = c['High'] - c['Low']

    if total_range < 1e-8:
        return False, "Doji / No Range"

    upper_wick  = c['High']  - max(c['Open'], c['Close'])
    lower_wick  = min(c['Open'], c['Close']) - c['Low']

    is_bullish_close = c['Close'] > c['Open']

    # 1. Hammer: small body at top, lower wick >= 2x body
    is_hammer = (
        is_bullish_close and
        abs(body) > 0 and
        lower_wick >= 2 * abs(body) and
        upper_wick < abs(body)
    )

    # 2. Bullish Engulfing
    prev_body = p['Open'] - p['Close']   # Bearish previous
    is_engulfing = (
        is_bullish_close and
        p['Close'] < p['Open'] and       # Previous was bearish
        c['Close'] > p['Open'] and
        c['Open']  < p['Close']
    )

    # 3. Pin Bar: lower wick > 60% of total range, bullish close
    is_pin_bar = (
        is_bullish_close and
        (lower_wick / total_range) > 0.60
    )

    if is_hammer:    return True, "🔨 Hammer"
    if is_engulfing: return True, "🕯️ Bullish Engulfing"
    if is_pin_bar:   return True, "📌 Pin Bar"
    return False, "No Pattern"


def calculate_rr(entry, sl, tp1):
    """Calculate Risk:Reward ratio"""
    risk   = entry - sl
    reward = tp1 - entry
    if risk <= 0:
        return 0
    return round(reward / risk, 2)


def scan_stock(symbol):
    """
    Full pipeline for one stock.
    Returns dict with all scan results, or None if skip.
    """
    try:
        df = yf.download(symbol, period=PERIOD, interval=INTERVAL,
                         progress=False, auto_adjust=True)

        if df is None or len(df) < 60:
            return {"symbol": symbol, "status": "skip", "reason": "Insufficient data"}

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna().reset_index()

        # ── STEP 1: Uptrend check ─────────────────────────────
        uptrend, uptrend_reason = check_uptrend(df)
        if not uptrend:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"No uptrend ({uptrend_reason})"}

        # ── STEP 2: Swing points ──────────────────────────────
        sl_price, sh_price, sl_idx, sh_idx = find_swing_points(df)
        price_range = sh_price - sl_price

        if price_range <= 0:
            return {"symbol": symbol, "status": "fail", "reason": "Invalid swing range"}

        # ── STEP 3: Fibonacci zone ────────────────────────────
        fib_61   = sh_price - 0.61  * price_range
        fib_705  = sh_price - 0.705 * price_range
        fib_79   = sh_price - 0.79  * price_range
        fib_n27  = sh_price + 0.27  * price_range   # TP2 extension

        current_price = float(df['Close'].iloc[-1])
        in_zone       = (fib_79 <= current_price <= fib_61)

        if not in_zone:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"Price {current_price:.1f} not in zone ({fib_79:.1f}–{fib_61:.1f})"}

        # ── STEP 4: FVG check (bonus) ─────────────────────────
        fvg_found, fvg_level = find_fvg_in_zone(df, fib_79, fib_61, from_idx=sl_idx)

        # ── STEP 5: Bullish Price Action ──────────────────────
        pa_valid, pa_pattern = check_bullish_pa(df)
        if not pa_valid:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"No bullish PA (last pattern: {pa_pattern})"}

        # ── STEP 6: R:R calculation ───────────────────────────
        entry_price = current_price
        stop_loss   = sl_price * (1 - SL_BUFFER_PCT)
        tp1         = sh_price
        tp2         = fib_n27
        rr          = calculate_rr(entry_price, stop_loss, tp1)

        if rr < MIN_RR:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"Low R:R = {rr} (min {MIN_RR})"}

        # ── PASSED! ───────────────────────────────────────────
        last_candle = df.iloc[-1]
        score = (
            3 +
            (2 if fvg_found else 0) +
            (1 if pa_pattern == "🕯️ Bullish Engulfing" else 0) +
            (1 if rr >= 3 else 0)
        )

        return {
            "symbol":        symbol,
            "status":        "pass",
            "current_price": round(current_price, 2),
            "swing_low":     round(sl_price, 2),
            "swing_high":    round(sh_price, 2),
            "fib_61":        round(fib_61, 2),
            "fib_705":       round(fib_705, 2),
            "fib_79":        round(fib_79, 2),
            "entry":         round(entry_price, 2),
            "stop_loss":     round(stop_loss, 2),
            "tp1":           round(tp1, 2),
            "tp2":           round(tp2, 2),
            "rr":            rr,
            "fvg_found":     fvg_found,
            "fvg_level":     round(fvg_level, 2) if fvg_level else None,
            "pa_pattern":    pa_pattern,
            "uptrend_reason": uptrend_reason,
            "score":         score,
            "volume":        int(last_candle.get('Volume', 0)),
            "reason":        "All conditions met ✓",
        }

    except Exception as e:
        return {"symbol": symbol, "status": "error", "reason": str(e)}


# ─────────────────────────────────────────────────────────────
#  HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_html(passed, failed, skipped, scan_time, total):
    score_badge = lambda s: (
        "🔥 Strong" if s >= 6 else
        "✅ Good"   if s >= 4 else
        "⚡ Moderate"
    )

    rows = ""
    for r in passed:
        fvg_badge = (
            f'<span class="badge fvg">✦ FVG @ {r["fvg_level"]}</span>'
            if r["fvg_found"] else
            '<span class="badge no-fvg">No FVG</span>'
        )
        score_cls = "strong" if r["score"] >= 6 else ("good" if r["score"] >= 4 else "moderate")
        rows += f"""
        <tr>
          <td class="sym">{r['symbol'].replace('.NS','').replace('.BO','')}</td>
          <td>₹{r['current_price']}</td>
          <td class="zone">₹{r['fib_79']} – ₹{r['fib_61']}</td>
          <td>{r['pa_pattern']}</td>
          <td>{fvg_badge}</td>
          <td>₹{r['stop_loss']}</td>
          <td class="tp">₹{r['tp1']}</td>
          <td class="tp2">₹{r['tp2']}</td>
          <td class="rr">{r['rr']}:1</td>
          <td><span class="score {score_cls}">{score_badge(r['score'])} ({r['score']}/7)</span></td>
        </tr>"""

    fail_rows = ""
    for r in failed[:50]:   # Show top 50 fails
        fail_rows += f"""
        <tr>
          <td class="sym">{r['symbol'].replace('.NS','').replace('.BO','')}</td>
          <td colspan="2" class="fail-reason">{r['reason']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Fibonacci Scanner Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:       #0a0e1a;
    --surface:  #111827;
    --card:     #1a2235;
    --border:   #1e2d45;
    --gold:     #f0b429;
    --green:    #22c55e;
    --red:      #ef4444;
    --blue:     #3b82f6;
    --teal:     #14b8a6;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --fvg:      #a78bfa;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
  }}

  /* ── HEADER ── */
  header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 2rem 2rem;
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(240,180,41,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
  }}
  .header-inner {{
    max-width: 1400px; margin: 0 auto;
    position: relative; z-index: 1;
    display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;
  }}
  .header-title {{ display: flex; align-items: center; gap: 1rem; }}
  .fib-icon {{
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--gold), #f59e0b);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 30px rgba(240,180,41,0.3);
  }}
  h1 {{
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
    color: #fff; letter-spacing: -0.02em;
  }}
  h1 span {{ color: var(--gold); }}
  .subtitle {{
    font-size: 0.8rem; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-top: 0.2rem;
  }}
  .scan-meta {{
    text-align: right;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem; color: var(--muted);
  }}
  .scan-meta strong {{ color: var(--gold); font-size: 0.9rem; }}

  /* ── STATS BAR ── */
  .stats-bar {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
  }}
  .stats-inner {{
    max-width: 1400px; margin: 0 auto;
    display: flex; gap: 1rem; flex-wrap: wrap;
  }}
  .stat-card {{
    flex: 1; min-width: 140px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    display: flex; flex-direction: column; gap: 0.25rem;
  }}
  .stat-card .label {{
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted);
  }}
  .stat-card .value {{
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
  }}
  .stat-card.pass  .value {{ color: var(--green); }}
  .stat-card.fail  .value {{ color: var(--red); }}
  .stat-card.skip  .value {{ color: var(--muted); }}
  .stat-card.total .value {{ color: var(--gold); }}

  /* ── LEGEND ── */
  .legend {{
    max-width: 1400px; margin: 1.5rem auto 0;
    padding: 0 2rem;
    display: flex; gap: 1rem; flex-wrap: wrap;
    font-size: 0.78rem; color: var(--muted);
  }}
  .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
  .legend-dot {{
    width: 8px; height: 8px; border-radius: 50%;
  }}

  /* ── MAIN CONTENT ── */
  main {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}

  .section-title {{
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--gold);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.75rem;
  }}
  .section-title::after {{
    content: ''; flex: 1; height: 1px;
    background: var(--border);
  }}

  /* ── TABLE ── */
  .table-wrap {{
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 2.5rem;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead tr {{
    background: var(--card);
    border-bottom: 2px solid var(--border);
  }}
  th {{
    padding: 0.85rem 1rem;
    text-align: left;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); white-space: nowrap;
  }}
  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
  td {{
    padding: 0.9rem 1rem;
    font-size: 0.875rem;
    white-space: nowrap;
  }}

  .sym {{
    font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 0.9rem;
    color: #fff;
    letter-spacing: 0.05em;
  }}
  .zone  {{ color: var(--gold); font-family: 'Space Mono', monospace; font-size: 0.8rem; }}
  .tp    {{ color: var(--green); font-family: 'Space Mono', monospace; }}
  .tp2   {{ color: var(--teal);  font-family: 'Space Mono', monospace; }}
  .rr    {{
    font-family: 'Space Mono', monospace; font-weight: 700;
    color: var(--blue);
  }}

  .badge {{
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.2rem 0.6rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; white-space: nowrap;
  }}
  .badge.fvg {{
    background: rgba(167,139,250,0.15);
    color: var(--fvg); border: 1px solid rgba(167,139,250,0.3);
  }}
  .badge.no-fvg {{
    background: rgba(100,116,139,0.1);
    color: var(--muted); border: 1px solid rgba(100,116,139,0.2);
  }}

  .score {{
    display: inline-flex; align-items: center;
    padding: 0.25rem 0.7rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
  }}
  .score.strong {{
    background: rgba(240,180,41,0.15);
    color: var(--gold); border: 1px solid rgba(240,180,41,0.3);
  }}
  .score.good {{
    background: rgba(34,197,94,0.12);
    color: var(--green); border: 1px solid rgba(34,197,94,0.25);
  }}
  .score.moderate {{
    background: rgba(59,130,246,0.12);
    color: var(--blue); border: 1px solid rgba(59,130,246,0.25);
  }}

  /* ── FAIL TABLE ── */
  .fail-table tbody tr {{ opacity: 0.6; }}
  .fail-reason {{ color: var(--muted); font-size: 0.8rem; }}

  /* ── FOOTER ── */
  .footer {{
    text-align: center; padding: 2rem;
    color: var(--muted); font-size: 0.75rem;
    border-top: 1px solid var(--border);
    max-width: 1400px; margin: 0 auto;
  }}
  .footer strong {{ color: var(--red); }}

  /* ── NO RESULTS ── */
  .no-results {{
    text-align: center; padding: 3rem;
    color: var(--muted); font-family: 'Space Mono', monospace;
  }}

  /* ── STRATEGY BOX ── */
  .strategy-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 2rem;
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.8;
  }}
  .strategy-box span {{ color: var(--text); }}
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="header-inner">
    <div class="header-title">
      <div class="fib-icon">φ</div>
      <div>
        <h1>Fibonacci <span>Scanner</span></h1>
        <p class="subtitle">61%–79% Entry Zone · FVG Confluence · SMC Strategy</p>
      </div>
    </div>
    <div class="scan-meta">
      <div>Scanned <strong>{total}</strong> symbols</div>
      <div style="margin-top:0.3rem">{scan_time}</div>
      <div style="margin-top:0.2rem">Daily Timeframe · 6 Month Data</div>
    </div>
  </div>
</header>

<!-- STATS BAR -->
<div class="stats-bar">
  <div class="stats-inner">
    <div class="stat-card total">
      <span class="label">Total Scanned</span>
      <span class="value">{total}</span>
    </div>
    <div class="stat-card pass">
      <span class="label">✅ Passed (All Conditions)</span>
      <span class="value">{len(passed)}</span>
    </div>
    <div class="stat-card fail">
      <span class="label">❌ Did Not Qualify</span>
      <span class="value">{len(failed)}</span>
    </div>
    <div class="stat-card skip">
      <span class="label">⚠️ Skipped / Error</span>
      <span class="value">{len(skipped)}</span>
    </div>
    <div class="stat-card" style="border-color: rgba(167,139,250,0.3);">
      <span class="label" style="color:var(--fvg)">FVG Confluence</span>
      <span class="value" style="color:var(--fvg)">{sum(1 for r in passed if r['fvg_found'])}</span>
    </div>
  </div>
</div>

<main>

  <!-- STRATEGY REMINDER -->
  <div class="strategy-box">
    <b style="color:var(--gold)">Strategy Logic: </b>
    <span>Uptrend (HH+HL) </span> → <span>Swing Low→High identified </span> →
    <span>Price in 61%–79% Fib Zone </span> → <span>FVG overlap (bonus) </span> →
    <span>Bullish PA confirmation (Hammer / Engulfing / Pin Bar) </span> →
    <span>R:R ≥ 2:1 </span> → <span style="color:var(--green)">LONG ENTRY</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <b style="color:var(--muted)">SL:</b> <span>Below Swing Low</span> &nbsp;
    <b style="color:var(--green)">TP1:</b> <span>0% (Swing High)</span> &nbsp;
    <b style="color:var(--teal)">TP2:</b> <span>-27% Extension</span>
  </div>

  <!-- PASSED STOCKS -->
  <p class="section-title">🎯 Shortlisted Stocks — {len(passed)} Found</p>

  {'<div class="no-results">No stocks passed all conditions. Try scanning again tomorrow — markets change daily.</div>' if not passed else f'''
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th>CMP</th>
          <th>Fib Zone (61%–79%)</th>
          <th>PA Signal</th>
          <th>FVG</th>
          <th>Stop Loss</th>
          <th style="color:var(--green)">TP1 (0%)</th>
          <th style="color:var(--teal)">TP2 (-27%)</th>
          <th>R:R</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>'''}

  <!-- FAILED STOCKS (collapsed) -->
  <details style="margin-bottom:2rem">
    <summary style="cursor:pointer; color:var(--muted); font-size:0.85rem; padding:0.5rem 0; outline:none;">
      📋 View Rejected Stocks ({len(failed)}) — click to expand
    </summary>
    <div class="table-wrap" style="margin-top:1rem">
      <table class="fail-table">
        <thead>
          <tr><th>Symbol</th><th colspan="2">Reason Failed</th></tr>
        </thead>
        <tbody>{fail_rows if fail_rows else '<tr><td colspan="3" style="color:var(--muted);text-align:center;padding:2rem">None</td></tr>'}</tbody>
      </table>
    </div>
  </details>

</main>

<div class="footer">
  <strong>⚠️ DISCLAIMER:</strong> This report is for <strong>educational purposes only</strong>.
  It is not financial advice. Trading involves significant risk.
  Always do your own research and consult a financial advisor before trading.
  Past performance does not guarantee future results.
</div>

</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────
#  MAIN — Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FIBONACCI RETRACEMENT SCANNER")
    print("  61%–79% Entry Zone + FVG + Bullish PA")
    print("=" * 60)

    # ── Load CSV ──────────────────────────────────────────────
    if not os.path.exists(CSV_FILE):
        print(f"\n❌ ERROR: '{CSV_FILE}' not found!")
        print(f"   Create a CSV with column '{SYMBOL_COLUMN}' listing stock symbols.")
        print("   Example:")
        print("     Symbol")
        print("     RELIANCE.NS")
        print("     TCS.NS")
        sys.exit(1)

    df_symbols = pd.read_csv(CSV_FILE)

    if SYMBOL_COLUMN not in df_symbols.columns:
        print(f"\n❌ ERROR: Column '{SYMBOL_COLUMN}' not found in CSV!")
        print(f"   Found columns: {list(df_symbols.columns)}")
        sys.exit(1)

    symbols = df_symbols[SYMBOL_COLUMN].dropna().str.strip().tolist()
    total   = len(symbols)
    print(f"\n📋 Loaded {total} symbols from {CSV_FILE}")
    print(f"📅 Timeframe: Daily | Period: {PERIOD}\n")

    # ── Scan ──────────────────────────────────────────────────
    passed  = []
    failed  = []
    skipped = []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>3}/{total}] Scanning {symbol:<20}", end=" ")
        result = scan_stock(symbol)

        if result["status"] == "pass":
            passed.append(result)
            fvg_tag = " ✦FVG" if result["fvg_found"] else ""
            print(f"✅ PASS — {result['pa_pattern']}{fvg_tag} | R:R {result['rr']}:1")
        elif result["status"] == "fail":
            failed.append(result)
            print(f"❌ {result['reason'][:55]}")
        else:
            skipped.append(result)
            print(f"⚠️  {result['reason'][:55]}")

    # Sort passed by score (highest first), then R:R
    passed.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)

    # ── Generate Report ───────────────────────────────────────
    scan_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
    html      = generate_html(passed, failed, skipped, scan_time, total)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  SCAN COMPLETE — {scan_time}")
    print("=" * 60)
    print(f"  Total Scanned : {total}")
    print(f"  ✅ Passed     : {len(passed)}")
    print(f"  ❌ Failed     : {len(failed)}")
    print(f"  ⚠️  Skipped   : {len(skipped)}")
    print(f"  ✦  With FVG  : {sum(1 for r in passed if r['fvg_found'])}")
    print(f"\n  📊 Report saved: {OUTPUT_FILE}")
    print("=" * 60)

    if passed:
        print("\n  🏆 TOP PICKS:")
        for r in passed[:5]:
            fvg = " ✦FVG" if r["fvg_found"] else ""
            print(f"     {r['symbol']:<18} | CMP: {r['current_price']:>8} | R:R {r['rr']}:1{fvg}")

    print("\n  Open index.html in your browser to see full report.\n")


if __name__ == "__main__":
    main()
