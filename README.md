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

Logistic regression calculates probability of "up":