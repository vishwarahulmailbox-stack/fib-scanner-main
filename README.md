# φ Fibonacci Scanner — NSE India

> Automated daily scanner for **Fibonacci 61%–79% retracement entries** with FVG confluence and bullish PA confirmation. Runs on GitHub Actions every weekday and publishes results to GitHub Pages.

🔗 **Live Report:** [vishwarahulmailbox-stack.github.io/fib-scanner-main](https://vishwarahulmailbox-stack.github.io/fib-scanner-main/)

---

## Strategy Logic

```
Uptrend (HH + HL + Above MA20)
  → Identify Swing Low → Swing High
    → Price in 61%–79% Fibonacci Zone
      → FVG overlap in zone (bonus)
        → Bullish PA (Hammer / Engulfing / Pin Bar)
          → R:R ≥ 2:1
            → ✅ LONG ENTRY
```

| Level | Description |
|-------|-------------|
| **Entry** | Current price inside 61%–79% retracement zone |
| **Stop Loss** | Below Swing Low (−0.3% buffer) |
| **TP1** | 0% — Swing High |
| **TP2** | −27% Extension above Swing High |
| **Min R:R** | 2:1 |

---

## Scoring (max 7)

| Condition | Points |
|-----------|--------|
| Base (all filters passed) | 3 |
| FVG found in zone | +2 |
| Bullish Engulfing pattern | +1 |
| R:R ≥ 3:1 | +1 |

🔥 Strong = 6–7 &nbsp;|&nbsp; ✅ Good = 4–5 &nbsp;|&nbsp; ⚡ Moderate = 3

---

## File Structure

```
fib-scanner-main/
├── fib_scanner.py          # Main scanner script
├── stocks.csv              # NSE watchlist (Symbol column)
├── index.html              # Generated report (published to GitHub Pages)
├── README.md
└── .github/
    └── workflows/
        └── fib_scanner.yml # GitHub Actions automation
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/vishwarahulmailbox-stack/fib-scanner-main.git
cd fib-scanner-main
pip install yfinance pandas numpy pytz
```

### 2. Add your stocks

Edit `stocks.csv` — one symbol per row, no `.NS` suffix needed:

```csv
Symbol
RELIANCE
TCS
HDFCBANK
INFY
```

### 3. Run locally

```bash
python fib_scanner.py
```

Opens `index.html` in your browser with the full report.

---

## Automation (GitHub Actions)

The scanner runs automatically **Monday–Friday at 2:00 PM IST** via GitHub Actions.

`.github/workflows/fib_scanner.yml`:
- Checks out repo
- Installs Python 3.11 + dependencies
- Runs `fib_scanner.py`
- Commits updated `index.html` back to `main`
- GitHub Pages serves the latest report live

**Manual trigger:** Go to Actions tab → `FIB Scanner` → `Run workflow`

---

## GitHub Pages Setup

1. Go to repo **Settings → Pages**
2. Source: `Deploy from a branch`
3. Branch: `main` / `/ (root)`
4. Save — your report will be live at:
   `https://<your-username>.github.io/fib-scanner-main/`

---

## Configuration

Edit the constants at the top of `fib_scanner.py`:

```python
CSV_FILE         = "stocks.csv"     # Your watchlist file
PERIOD           = "6mo"            # Data lookback period
INTERVAL         = "1d"             # Timeframe (daily)
UPTREND_LOOKBACK = 40               # Candles for HH/HL check
MIN_RR           = 2.0              # Minimum Risk:Reward ratio
SL_BUFFER_PCT    = 0.003            # 0.3% buffer below swing low
EXCHANGE_SUFFIX  = ".NS"            # .NS for NSE, .BO for BSE
```

---

## Disclaimer

> ⚠️ This tool is for **educational purposes only** and is **not financial advice**.
> Trading in equity markets involves significant risk of loss.
> Always consult a SEBI-registered financial advisor before taking any trade.
> Past performance does not guarantee future results.
