# Gold Forex Analyzer

A machine learning tool that generates trading signals for gold (XAU/USD).

## How It Works

This tool downloads 5 years of gold price data from Yahoo Finance and uses a Random Forest machine learning model to predict whether the price will go UP or DOWN over the next 3 days. It analyzes 10 different technical indicators including price returns, moving averages, RSI, and volatility to make its prediction. Based on the model's confidence level (above 60%), it generates a BUY, SELL, or NEUTRAL signal and calculates recommended stop-loss and take-profit levels using the Average True Range (ATR). The results are displayed as a clean PNG dashboard with a price chart, signal, trade plan, and market data.

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd gold-forex-analyzer

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the predictor
python gold_predictor.py

### Output
The tool generates:

A PNG dashboard (gold_signal_YYYYMMDD.png) with all trading information

Terminal output showing the signal, confidence, entry price, stop-loss, and take-profit levels

### Requirements
text
pandas, numpy, scikit-learn, yfinance, matplotlib
Disclaimer
⚠️ For educational purposes only. Not financial advice. Trading involves substantial risk of loss. Always backtest before using with real money.

text

---

This is clean, professional, and explains everything in one clear paragraph without being overwhelming!