"""
╔══════════════════════════════════════════════════════════════╗
║     FIBONACCI RETRACEMENT SCANNER v2 — SMC Strategy         ║
║     61%–79% Entry Zone + FVG Confluence                      ║
║     Daily Timeframe | NSE Stocks | HTML Report               ║
╚══════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings, os, sys
import pytz

from datetime import datetime

warnings.filterwarnings('ignore')

CSV_FILE         = "stocks.csv"
SYMBOL_COLUMN    = "Symbol"
OUTPUT_FILE      = "index.html"
EXCHANGE_SUFFIX  = ".NS"
PERIOD           = "6mo"
INTERVAL         = "1d"
UPTREND_LOOKBACK = 40
MIN_RR           = 2.0
SL_BUFFER_PCT    = 0.003


def find_swing_points(df):
    highs = df['High'].values
    lows  = df['Low'].values
    n     = len(highs)
    lookback    = min(60, n)
    sh_rel_idx  = np.argmax(highs[-lookback:])
    sh_idx      = n - lookback + sh_rel_idx
    sh_price    = highs[sh_idx]
    search_start = max(0, sh_idx - 60)
    sl_rel_idx   = np.argmin(lows[search_start:sh_idx])
    sl_idx       = search_start + sl_rel_idx
    sl_price     = lows[sl_idx]
    return sl_price, sh_price, sl_idx, sh_idx


def check_uptrend(df, lookback=UPTREND_LOOKBACK):
    if len(df) < lookback:
        return False, "Not enough data"
    closes = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    ma20          = np.mean(closes[-20:])
    current_price = closes[-1]
    above_ma      = current_price > ma20
    mid    = lookback // 2
    fh_hi  = np.max(highs[-lookback:-mid])
    sh_hi  = np.max(highs[-mid:])
    fh_lo  = np.min(lows[-lookback:-mid])
    sh_lo  = np.min(lows[-mid:])
    hh = sh_hi > fh_hi
    hl = sh_lo > fh_lo
    if hh and hl and above_ma:
        return True, "HH + HL + Above MA20"
    reasons = []
    if not hh:       reasons.append("No HH")
    if not hl:       reasons.append("No HL")
    if not above_ma: reasons.append("Below MA20")
    return False, " | ".join(reasons)


def find_fvg_in_zone(df, zone_low, zone_high, from_idx=0):
    sub     = df.iloc[from_idx:].reset_index(drop=True)
    best_fvg = None
    for i in range(1, len(sub) - 1):
        gap_low  = float(sub.iloc[i - 1]['High'])
        gap_high = float(sub.iloc[i + 1]['Low'])
        if gap_high > gap_low:
            gap_mid = (gap_low + gap_high) / 2
            if not (gap_high < zone_low or gap_low > zone_high):
                best_fvg = gap_mid
    return (best_fvg is not None), best_fvg


def check_bullish_pa(df):
    if len(df) < 2:
        return False, "Not enough candles"
    c = df.iloc[-1]
    p = df.iloc[-2]
    o, h, l, cl = float(c['Open']), float(c['High']), float(c['Low']), float(c['Close'])
    body        = cl - o
    total_range = h - l
    if total_range < 1e-8:
        return False, "Doji"
    lower_wick  = min(o, cl) - l
    upper_wick  = h - max(o, cl)
    is_bull     = cl > o
    if is_bull and abs(body) > 0 and lower_wick >= 2 * abs(body) and upper_wick < abs(body):
        return True, "🔨 Hammer"
    if (is_bull and float(p['Close']) < float(p['Open'])
            and cl > float(p['Open']) and o < float(p['Close'])):
        return True, "🕯️ Bullish Engulfing"
    if is_bull and (lower_wick / total_range) > 0.60:
        return True, "📌 Pin Bar"
    return False, "No Pattern"


def calculate_rr(entry, sl, tp1):
    risk   = entry - sl
    reward = tp1 - entry
    if risk <= 0:
        return 0
    return round(reward / risk, 2)


def scan_stock(raw_symbol):
    symbol = raw_symbol.strip()
    yf_symbol = symbol if ("." in symbol) else (symbol + EXCHANGE_SUFFIX)
    try:
        df = yf.download(yf_symbol, period=PERIOD, interval=INTERVAL,
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 60:
            return {"symbol": symbol, "status": "skip", "reason": "Insufficient data / Not found"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna().reset_index()
        uptrend, uptrend_reason = check_uptrend(df)
        if not uptrend:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"No uptrend ({uptrend_reason})"}
        sl_price, sh_price, sl_idx, sh_idx = find_swing_points(df)
        price_range = sh_price - sl_price
        if price_range <= 0:
            return {"symbol": symbol, "status": "fail", "reason": "Invalid swing range"}
        fib_61  = sh_price - 0.61  * price_range
        fib_705 = sh_price - 0.705 * price_range
        fib_79  = sh_price - 0.79  * price_range
        fib_n27 = sh_price + 0.27  * price_range
        current_price = float(df['Close'].iloc[-1])
        in_zone       = (fib_79 <= current_price <= fib_61)
        if not in_zone:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"Price ₹{current_price:.1f} not in zone (₹{fib_79:.1f}–₹{fib_61:.1f})"}
        fvg_found, fvg_level = find_fvg_in_zone(df, fib_79, fib_61, from_idx=sl_idx)
        pa_valid, pa_pattern = check_bullish_pa(df)
        if not pa_valid:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"No bullish PA ({pa_pattern})"}
        entry_price = current_price
        stop_loss   = sl_price * (1 - SL_BUFFER_PCT)
        tp1         = sh_price
        tp2         = fib_n27
        rr          = calculate_rr(entry_price, stop_loss, tp1)
        if rr < MIN_RR:
            return {"symbol": symbol, "status": "fail",
                    "reason": f"Low R:R = {rr} (min {MIN_RR})"}
        score = (3
                 + (2 if fvg_found else 0)
                 + (1 if "Engulfing" in pa_pattern else 0)
                 + (1 if rr >= 3 else 0))
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
            "score":         score,
            "volume":        int(df.iloc[-1].get('Volume', 0)),
        }
    except Exception as e:
        return {"symbol": symbol, "status": "error", "reason": str(e)[:80]}


def score_label(s):
    if s >= 6: return ("🔥 Strong", "strong")
    if s >= 4: return ("✅ Good",   "good")
    return ("⚡ Moderate", "moderate")


def build_passed_rows(passed):
    rows = ""
    for r in passed:
        label, cls = score_label(r["score"])
        fvg_badge = (
            f'<span class="badge fvg">✦ FVG @ ₹{r["fvg_level"]}</span>'
            if r["fvg_found"] else
            '<span class="badge no-fvg">—</span>'
        )
        rows += f"""
        <tr>
          <td class="sym">{r['symbol']}</td>
          <td class="mono">₹{r['current_price']}</td>
          <td class="zone mono">₹{r['fib_79']} – ₹{r['fib_61']}</td>
          <td>{r['pa_pattern']}</td>
          <td>{fvg_badge}</td>
          <td class="mono red">₹{r['stop_loss']}</td>
          <td class="mono green">₹{r['tp1']}</td>
          <td class="mono teal">₹{r['tp2']}</td>
          <td class="mono rr">{r['rr']}:1</td>
          <td><span class="score {cls}">{label} ({r['score']}/7)</span></td>
        </tr>"""
    return rows


def build_fail_rows(failed):
    rows = ""
    for r in failed[:80]:
        rows += f"""
        <tr>
          <td class="sym">{r['symbol']}</td>
          <td class="fail-reason">{r['reason']}</td>
        </tr>"""
    return rows


def generate_html(passed, failed, skipped, scan_time, total):
    passed_rows = build_passed_rows(passed)
    fail_rows   = build_fail_rows(failed)
    fvg_count   = sum(1 for r in passed if r["fvg_found"])

    no_results_msg = """
    <div class="no-results">
      <div style="font-size:2rem;margin-bottom:1rem">🔍</div>
      No stocks passed all conditions today.<br>
      <span style="font-size:0.8rem;color:#64748b">Markets change daily — try again tomorrow.</span>
    </div>""" if not passed else ""

    passed_table = f"""
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th><th>CMP</th>
          <th>Fib Zone (61%–79%)</th>
          <th>PA Signal</th><th>FVG</th>
          <th>Stop Loss</th>
          <th style="color:#16a34a">TP1 (0%)</th>
          <th style="color:#0d9488">TP2 (−27%)</th>
          <th>R:R</th><th>Score</th>
        </tr></thead>
        <tbody>{passed_rows}</tbody>
      </table>
    </div>""" if passed else no_results_msg

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Fibonacci Scanner — {scan_time}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{{
  --bg:#f0f4f8;--surface:#ffffff;--card:#ffffff;--border:#e2e8ef;
  --gold:#d97706;--gold-light:#fef3c7;--green:#16a34a;--green-light:#dcfce7;
  --red:#dc2626;--red-light:#fee2e2;--blue:#2563eb;--blue-light:#dbeafe;
  --teal:#0d9488;--teal-light:#ccfbf1;--text:#0f172a;--muted:#64748b;
  --fvg:#7c3aed;--fvg-light:#ede9fe;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
  --shadow-md:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -1px rgba(0,0,0,.05);
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;font-size:14px}}

header{{
  background:linear-gradient(135deg,#1e3a5f 0%,#1e40af 50%,#0f172a 100%);
  padding:2rem 2.5rem;position:relative;overflow:hidden;
}}
header::before{{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 20% 50%,rgba(253,224,71,.08) 0%,transparent 60%),
             radial-gradient(ellipse at 80% 50%,rgba(147,197,253,.08) 0%,transparent 60%);
}}
.hdr{{max-width:1440px;margin:0 auto;position:relative;z-index:1;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1.5rem}}
.hdr-left{{display:flex;align-items:center;gap:1.2rem}}
.phi{{
  width:52px;height:52px;border-radius:14px;
  background:linear-gradient(135deg,#fbbf24,#f59e0b);
  display:flex;align-items:center;justify-content:center;
  font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:600;color:#1c1917;
  box-shadow:0 4px 20px rgba(251,191,36,.35);flex-shrink:0;
}}
h1{{font-size:1.35rem;font-weight:700;color:#fff;letter-spacing:-.02em}}
h1 span{{color:#fcd34d}}
.sub{{font-size:.7rem;color:rgba(255,255,255,.55);letter-spacing:.1em;text-transform:uppercase;margin-top:.25rem}}
.hdr-right{{text-align:right;font-size:.75rem;color:rgba(255,255,255,.55);font-family:'JetBrains Mono',monospace;line-height:1.9}}
.hdr-right strong{{color:#fcd34d;font-size:.92rem}}

.stats-wrap{{background:var(--surface);border-bottom:1px solid var(--border);padding:1.25rem 2.5rem;box-shadow:var(--shadow)}}
.stats{{max-width:1440px;margin:0 auto;display:flex;gap:.75rem;flex-wrap:wrap}}
.stat{{
  flex:1;min-width:130px;background:var(--bg);border:1px solid var(--border);
  border-radius:12px;padding:.9rem 1.2rem;box-shadow:var(--shadow);
}}
.stat .lbl{{font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:.3rem;font-weight:500}}
.stat .val{{font-family:'JetBrains Mono',monospace;font-size:1.75rem;font-weight:600}}
.stat.s-total .val{{color:#1e40af}}
.stat.s-pass  .val{{color:var(--green)}}
.stat.s-fail  .val{{color:var(--red)}}
.stat.s-skip  .val{{color:var(--muted)}}
.stat.s-fvg   .val{{color:var(--fvg)}}

main{{max-width:1440px;margin:0 auto;padding:1.75rem 2.5rem 4rem}}

.strat-box{{
  background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--gold);
  border-radius:10px;padding:.9rem 1.2rem;margin-bottom:1.75rem;
  font-size:.78rem;color:var(--muted);line-height:2;box-shadow:var(--shadow);
}}
.strat-box b{{color:var(--gold)}}
.strat-box span{{color:var(--text)}}

.sec-title{{
  font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;
  color:#1e40af;margin-bottom:1rem;font-weight:600;
  display:flex;align-items:center;gap:.75rem;
}}
.sec-title::after{{content:'';flex:1;height:1px;background:var(--border)}}

.table-wrap{{overflow-x:auto;border-radius:12px;border:1px solid var(--border);margin-bottom:2rem;box-shadow:var(--shadow-md);background:var(--surface)}}
table{{width:100%;border-collapse:collapse}}
thead tr{{background:#f8fafc;border-bottom:2px solid var(--border)}}
th{{
  padding:.8rem 1rem;text-align:left;font-size:.65rem;
  text-transform:uppercase;letter-spacing:.08em;color:var(--muted);white-space:nowrap;font-weight:600;
}}
tbody tr{{border-bottom:1px solid var(--border);transition:background .1s}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover{{background:#f8fafc}}
td{{padding:.85rem 1rem;font-size:.84rem;white-space:nowrap}}

.sym{{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.88rem;color:#1e40af;letter-spacing:.04em}}
.mono{{font-family:'JetBrains Mono',monospace;font-size:.82rem}}
.zone{{color:var(--gold);font-weight:500}}
.green{{color:var(--green);font-weight:500}}
.red{{color:var(--red);font-weight:500}}
.teal{{color:var(--teal);font-weight:500}}
.blue{{color:var(--blue)}}
.rr{{font-weight:700;color:#1e40af}}

.badge{{display:inline-flex;align-items:center;gap:.25rem;padding:.2rem .6rem;border-radius:20px;font-size:.7rem;font-weight:600}}
.badge.fvg{{background:var(--fvg-light);color:var(--fvg);border:1px solid rgba(124,58,237,.2)}}
.badge.no-fvg{{color:var(--muted)}}

.score{{display:inline-flex;align-items:center;padding:.22rem .65rem;border-radius:20px;font-size:.7rem;font-weight:700}}
.score.strong{{background:var(--gold-light);color:var(--gold);border:1px solid rgba(217,119,6,.25)}}
.score.good{{background:var(--green-light);color:var(--green);border:1px solid rgba(22,163,74,.22)}}
.score.moderate{{background:var(--blue-light);color:var(--blue);border:1px solid rgba(37,99,235,.22)}}

.fail-table tbody tr{{opacity:.7}}
.fail-reason{{color:var(--muted);font-size:.78rem}}

details summary{{cursor:pointer;color:var(--muted);font-size:.82rem;padding:.6rem 0;outline:none;user-select:none;font-weight:500}}
details summary:hover{{color:var(--text)}}

.no-results{{
  text-align:center;padding:4rem 2rem;color:var(--muted);
  background:var(--surface);border-radius:12px;border:1px solid var(--border);box-shadow:var(--shadow);
}}

footer{{
  text-align:center;padding:1.5rem 2rem;color:var(--muted);font-size:.72rem;
  border-top:1px solid var(--border);max-width:1440px;margin:0 auto;line-height:2;
}}
</style>
</head>
<body>

<header>
  <div class="hdr">
    <div class="hdr-left">
      <div class="phi">φ</div>
      <div>
        <h1>Fibonacci <span>Scanner</span></h1>
        <p class="sub">61%–79% Entry Zone · FVG Confluence · SMC Strategy · NSE</p>
      </div>
    </div>
    <div class="hdr-right">
      <div>Scanned <strong>{total}</strong> NSE symbols</div>
      <div>{scan_time}</div>
      <div>Daily (1D) · 6 Month Data · Min R:R 2:1</div>
    </div>
  </div>
</header>

<div class="stats-wrap">
  <div class="stats">
    <div class="stat s-total"><div class="lbl">Total Scanned</div><div class="val">{total}</div></div>
    <div class="stat s-pass"> <div class="lbl">✅ Passed</div>    <div class="val">{len(passed)}</div></div>
    <div class="stat s-fail"> <div class="lbl">❌ Rejected</div>  <div class="val">{len(failed)}</div></div>
    <div class="stat s-skip"> <div class="lbl">⚠️ Skipped</div>  <div class="val">{len(skipped)}</div></div>
    <div class="stat s-fvg"  style="border-color:rgba(124,58,237,.2)">
      <div class="lbl" style="color:var(--fvg)">✦ With FVG</div>
      <div class="val">{fvg_count}</div>
    </div>
  </div>
</div>

<main>
  <div class="strat-box">
    <b>Strategy: </b>
    <span>Uptrend (HH+HL) </span>→
    <span> Swing Low→High </span>→
    <span> Price in 61–79% Fib Zone </span>→
    <span> FVG overlap (bonus) </span>→
    <span> Hammer / Engulfing / Pin Bar </span>→
    <span> R:R ≥ 2:1 </span>→
    <span style="color:var(--green);font-weight:600"> LONG ENTRY</span>
    &nbsp;|&nbsp;
    <b>SL:</b> <span> Below Swing Low</span>
    &nbsp;<b>TP1:</b> <span>0% (Swing High)</span>
    &nbsp;<b>TP2:</b> <span>−27% Extension</span>
  </div>

  <p class="sec-title">🎯 Shortlisted Stocks — {len(passed)} Passed All Conditions</p>
  {passed_table}

  <details>
    <summary>📋 Rejected Stocks ({len(failed)}) — click to expand</summary>
    <div class="table-wrap" style="margin-top:.75rem">
      <table class="fail-table">
        <thead><tr><th>Symbol</th><th>Reason</th></tr></thead>
        <tbody>{fail_rows if fail_rows else '<tr><td colspan="2" style="text-align:center;padding:2rem;color:var(--muted)">None</td></tr>'}</tbody>
      </table>
    </div>
  </details>
</main>

<footer>
  ⚠️ <strong style="color:#dc2626">DISCLAIMER:</strong>
  This report is for <strong style="color:#dc2626">educational purposes only</strong> and is NOT financial advice.<br>
  Trading in equity markets involves significant risk of loss.
  Always consult a SEBI-registered financial advisor before taking any trade.
</footer>

</body>
</html>"""


def main():
    print("=" * 62)
    print("  FIBONACCI RETRACEMENT SCANNER v2")
    print("  61%–79% Zone · FVG Confluence · NSE Daily")
    print("=" * 62)

    if not os.path.exists(CSV_FILE):
        print(f"\n❌ '{CSV_FILE}' not found in current folder!")
        sys.exit(1)

    df_sym = pd.read_csv(CSV_FILE)
    col = None
    for c in df_sym.columns:
        if c.strip().lower() == SYMBOL_COLUMN.lower():
            col = c; break

    if col is None:
        print(f"\n❌ Column '{SYMBOL_COLUMN}' not found!")
        sys.exit(1)

    symbols = df_sym[col].dropna().str.strip().tolist()
    total   = len(symbols)
    print(f"\n📋 {total} symbols loaded from {CSV_FILE}")
    print(f"🔗 Exchange: NSE ({EXCHANGE_SUFFIX}) | Period: {PERIOD} | TF: {INTERVAL}\n")

    passed, failed, skipped = [], [], []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:>3}/{total}] {sym:<18}", end=" ", flush=True)
        result = scan_stock(sym)

        if result["status"] == "pass":
            passed.append(result)
            fvg_tag = " ✦FVG" if result["fvg_found"] else ""
            print(f"✅  {result['pa_pattern']}{fvg_tag}  |  R:R {result['rr']}:1  |  Score {result['score']}/7")
        elif result["status"] == "fail":
            failed.append(result)
            print(f"❌  {result['reason'][:52]}")
        else:
            skipped.append(result)
            print(f"⚠️  {result['reason'][:52]}")

    passed.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)

    ist = pytz.timezone('Asia/Kolkata')
    scan_time = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")
    html      = generate_html(passed, failed, skipped, scan_time, total)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)s

    print("\n" + "=" * 62)
    print(f"  SCAN COMPLETE  —  {scan_time}")
    print("=" * 62)
    print(f"  Total   : {total}")
    print(f"  ✅ Pass : {len(passed)}")
    print(f"  ❌ Fail : {len(failed)}")
    print(f"  ⚠️  Skip : {len(skipped)}")
    print(f"  ✦  FVG  : {sum(1 for r in passed if r['fvg_found'])}")
    print(f"\n  📊 Report → {OUTPUT_FILE}  (open in browser)")
    print("=" * 62)

    if passed:
        print("\n  🏆 TOP PICKS:")
        for r in passed[:5]:
            fvg = " ✦FVG" if r["fvg_found"] else ""
            print(f"     {r['symbol']:<16} CMP ₹{r['current_price']:<9} R:R {r['rr']}:1  Score {r['score']}/7{fvg}")
    print()


if __name__ == "__main__":
    main()
