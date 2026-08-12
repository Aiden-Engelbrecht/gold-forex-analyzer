# Gold Forex Analyzer - Simple ML Demo

## What This Does

This is a **single-file machine learning example** that predicts whether gold prices (XAU/USD) will go **UP** or **DOWN** tomorrow.

### How It Works (Step by Step)

1. **Download data** - Fetches 5 years of gold prices from Yahoo Finance
2. **Create features** - Calculates technical indicators:
   - Daily price change (%)
   - Price range (high-low) 
   - 5-day moving average
   - RSI (Relative Strength Index)
3. **Set the target** - Did price go up (1) or down (0) the next day?
4. **Train model** - Logistic Regression learns patterns
5. **Predict** - Based on today's data, predicts tomorrow's direction

### The Simple Math

Logistic regression calculates probability of "up": Probability = 1 / (1 + e^-(b0 + b1x1 + b2x2 + ...))

- If probability > 50% → predict UP
- If probability < 50% → predict DOWN

The model learns the weights (b0, b1, b2...) from historical data.

## How to Run

### 1. Clone and setup
```bash
git clone <your-repo-url>
cd gold-forex-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the program
python gold_predictor.py

