"""
GOLD FOREX PREDICTOR - Simple Machine Learning Example
======================================================
Predicts if gold price will go UP or DOWN tomorrow using Logistic Regression.
Everything in one file for maximum simplicity!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: DOWNLOAD GOLD PRICE DATA
# ============================================================

print("📊 GOLD FOREX PREDICTOR")
print("=" * 50)
print("Downloading gold data from Yahoo Finance...")

# Download 5 years of gold data
gold = yf.Ticker("XAUUSD=X")
df = gold.history(period="5y")

# If that fails, try gold futures
if df.empty:
    print("Trying alternative ticker...")
    gold = yf.Ticker("GC=F")
    df = gold.history(period="5y")

if df.empty:
    raise Exception("Could not download data. Check internet connection.")

print(f"✅ Retrieved {len(df)} days of gold price data")
print(f"📅 From {df.index[0].date()} to {df.index[-1].date()}")
print()

# ============================================================
# STEP 2: CREATE FEATURES (Technical Indicators)
# ============================================================

print("Creating features from price data...")

# Make a copy and reset index
data = df.copy()
data['date'] = data.index
data = data.reset_index(drop=True)

# FEATURE 1: Daily price change percentage
data['price_change_pct'] = data['Close'].pct_change() * 100

# FEATURE 2: Price range (volatility)
data['high_low_range'] = (data['High'] - data['Low']) / data['Close'] * 100

# FEATURE 3: 5-day moving average
data['ma_5'] = data['Close'].rolling(window=5).mean()

# FEATURE 4: Distance from 5-day moving average
data['ma_5_distance'] = (data['Close'] - data['ma_5']) / data['ma_5'] * 100

# FEATURE 5: RSI (Relative Strength Index) - 14 days
def calculate_rsi(data, window=14):
    """Calculate RSI technical indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['rsi_14'] = calculate_rsi(data['Close'], 14)

# FEATURE 6: Volume change
data['volume_change'] = data['Volume'].pct_change() * 100

# ============================================================
# STEP 3: CREATE TARGET (What we want to predict)
# ============================================================

# Target: Did price go UP (1) or DOWN (0) tomorrow?
data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Remove rows with NaN values (from calculations)
data = data.dropna()
print(f"✅ Created {len(data)} valid data points after feature engineering")
print(f"   - Features: price_change, range, ma_5_distance, rsi, volume")
print()

# ============================================================
# STEP 4: PREPARE DATA FOR MACHINE LEARNING
# ============================================================

# Features (X) and target (y)
feature_columns = [
    'price_change_pct', 
    'high_low_range', 
    'ma_5_distance',
    'rsi_14', 
    'volume_change'
]

X = data[feature_columns]
y = data['target']

# Split into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Don't shuffle time-series data
)

print(f"Training set: {len(X_train)} days")
print(f"Testing set:  {len(X_test)} days")
print()

# ============================================================
# STEP 5: TRAIN THE MODEL
# ============================================================

print("Training Logistic Regression model...")

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Handle any imbalance
)

model.fit(X_train_scaled, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained!")
print(f"📊 Test Accuracy: {accuracy:.1%}")
print(f"💡 This means the model is {'better than' if accuracy > 0.5 else 'worse than'} random guessing")
print()

# Show which features are most important
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_[0]
})
feature_importance['Abs_Importance'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)

print("📈 Feature Importance (bigger = more influential):")
for _, row in feature_importance.iterrows():
    direction = "positive" if row['Coefficient'] > 0 else "negative"
    print(f"   {row['Feature']:18s}: {row['Coefficient']:+.3f} ({direction})")
print()

# ============================================================
# STEP 6: PREDICT TOMORROW'S PRICE DIRECTION
# ============================================================

# Get the latest data point (today)
latest = data.iloc[-1]
latest_features = latest[feature_columns].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_features)

# Make prediction
probability = model.predict_proba(latest_scaled)[0]  # [prob_down, prob_up]
prediction = model.predict(latest_scaled)[0]

# Get today's price info
today_price = latest['Close']
prev_close = data.iloc[-2]['Close'] if len(data) > 1 else today_price
change_pct = latest['price_change_pct']

# Display results
print("=" * 50)
print("📈 TODAY'S MARKET DATA")
print("=" * 50)
print(f"Latest Price:    ${today_price:.2f}")
print(f"Previous Close:  ${prev_close:.2f}")
print(f"Today's Change:  {change_pct:+.2f}%")
print()
print("🔍 FEATURES FOR TODAY:")
print(f"  Price Change %:   {latest['price_change_pct']:+.2f}%")
print(f"  High-Low Range:   {latest['high_low_range']:.2f}%")
print(f"  5-Day Avg Price:  ${latest['ma_5']:.2f}")
print(f"  RSI (14-day):     {latest['rsi_14']:.2f}")
print(f"  Volume Change:    {latest['volume_change']:+.1f}%")
print()
print("=" * 50)
print("🤖 MACHINE LEARNING PREDICTION")
print("=" * 50)
print(f"UP Probability:   {probability[1]:.1%}")
print(f"DOWN Probability: {probability[0]:.1%}")

if prediction == 1:
    print("⬆️  PREDICTION: GOLD WILL GO UP TOMORROW")
else:
    print("⬇️  PREDICTION: GOLD WILL GO DOWN TOMORROW")

print()
print(f"💡 Model Accuracy on historical data: {accuracy:.1%}")
print("⚠️  Remember: This is for educational purposes only!")
print("   Financial markets are unpredictable.")

# ============================================================
# STEP 7: EXTRA - EVALUATE PERFORMANCE
# ============================================================

print()
print("=" * 50)
print("📊 MODEL PERFORMANCE DETAILS")
print("=" * 50)

# Confusion matrix numbers
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Correct UP predictions:   {tp}")
print(f"Correct DOWN predictions: {tn}")
print(f"False UP predictions:     {fp}")
print(f"False DOWN predictions:   {fn}")
print()
print("💡 A perfect model would have 100% accuracy,")
print("   but financial data is noisy so 55-60% is realistic!")
print("   This demonstrates that ML isn't magic - it finds weak signals.")