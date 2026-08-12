"""
GOLD FOREX PREDICTOR - Simple Machine Learning Example with Dashboard
======================================================
Predicts if gold price will go UP or DOWN tomorrow using Logistic Regression.
Displays results as a visual dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up nice plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# STEP 1: DOWNLOAD GOLD PRICE DATA
# ============================================================

print("📊 GOLD FOREX PREDICTOR")
print("=" * 50)
print("Downloading gold data from Yahoo Finance...")

# Download 5 years of gold data - try different tickers
tickers = ["GC=F", "XAUUSD=X", "GLD"]
df = None

for ticker in tickers:
    try:
        print(f"Trying {ticker}...")
        gold = yf.Ticker(ticker)
        df = gold.history(period="5y")
        if not df.empty:
            print(f"✅ Success with {ticker}!")
            break
    except:
        continue

if df is None or df.empty:
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
    loss = loss.replace(0, np.nan)
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

# Remove rows with NaN or infinite values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

print(f"✅ Created {len(data)} valid data points after feature engineering")
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Training set: {len(X_train)} days")
print(f"Testing set:  {len(X_test)} days")
print()

# ============================================================
# STEP 5: TRAIN THE MODEL
# ============================================================

print("Training Logistic Regression model...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"✅ Model trained! Test Accuracy: {accuracy:.1%}")
print()

# ============================================================
# STEP 6: MAKE PREDICTION FOR TOMORROW
# ============================================================

# Get the latest data point (today)
latest = data.iloc[-1]
latest_features = latest[feature_columns].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_features)

# Make prediction
probability = model.predict_proba(latest_scaled)[0]
prediction = model.predict(latest_scaled)[0]

# Get today's price info
today_price = latest['Close']
prev_close = data.iloc[-2]['Close'] if len(data) > 1 else today_price
change_pct = latest['price_change_pct']

# Determine prediction text
up_prob = probability[1]
down_prob = probability[0]

if up_prob > 0.5:
    pred_text = '⬆️ UP'
    pred_color = '#00CC66'
else:
    pred_text = '⬇️ DOWN'
    pred_color = '#FF4444'

# ============================================================
# STEP 7: CREATE VISUAL DASHBOARD
# ============================================================

print("📊 Creating visual dashboard...")

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))
fig.suptitle('📈 GOLD FOREX PREDICTOR DASHBOARD', fontsize=20, fontweight='bold', y=0.98)

# === SUBPLOT 1: Price History ===
ax1 = plt.subplot(2, 2, 1)
ax1.plot(data['date'], data['Close'], color='#FFD700', linewidth=2, label='Gold Price')
ax1.axhline(y=today_price, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Current: ${today_price:.2f}')
ax1.set_title('Gold Price History (5 Years)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# === SUBPLOT 2: Prediction Gauge ===
ax2 = plt.subplot(2, 2, 2)

# Create a semi-circle gauge
theta = np.linspace(np.pi, 0, 100)
r = 1
x_gauge = r * np.cos(theta)
y_gauge = r * np.sin(theta)

# Background arc
ax2.plot(x_gauge, y_gauge, color='lightgray', linewidth=20, alpha=0.3)

# Colored arc based on prediction
if up_prob > 0.5:
    # Green for UP
    theta_colored = np.linspace(np.pi, np.pi * (1 - up_prob), 50)
    color = '#00CC66'
else:
    # Red for DOWN
    theta_colored = np.linspace(np.pi, np.pi * (1 - down_prob), 50)
    color = '#FF4444'

x_colored = r * np.cos(theta_colored)
y_colored = r * np.sin(theta_colored)
ax2.plot(x_colored, y_colored, color=color, linewidth=20)

# Center text
ax2.text(0, -0.3, f'{pred_text}', fontsize=28, fontweight='bold', 
         color=pred_color, ha='center', va='center')
ax2.text(0, -0.6, f'Confidence: {max(up_prob, down_prob):.1%}', 
         fontsize=14, ha='center', va='center')

# Add probability labels
ax2.text(-0.7, -0.1, f'DOWN\n{down_prob:.1%}', fontsize=10, ha='center', va='center')
ax2.text(0.7, -0.1, f'UP\n{up_prob:.1%}', fontsize=10, ha='center', va='center')

ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-0.8, 1.2)
ax2.set_aspect('equal')
ax2.set_title('Tomorrow\'s Prediction', fontsize=14, fontweight='bold')
ax2.axis('off')

# === SUBPLOT 3: Feature Importance ===
ax3 = plt.subplot(2, 2, 3)

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_[0]
})
feature_importance['Abs_Importance'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Coefficient')

colors = ['#FF4444' if x < 0 else '#00CC66' for x in feature_importance['Coefficient']]
bars = ax3.barh(feature_importance['Feature'], feature_importance['Coefficient'], 
                color=colors, alpha=0.7)
ax3.axvline(x=0, color='black', linewidth=0.5)
ax3.set_title('Feature Impact on Prediction', fontsize=14, fontweight='bold')
ax3.set_xlabel('Coefficient Value')
ax3.grid(True, alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(feature_importance.iterrows()):
    ax3.text(row['Coefficient'] + (0.02 if row['Coefficient'] > 0 else -0.08), 
             i, f'{row["Coefficient"]:.3f}', va='center', fontsize=9)

# === SUBPLOT 4: Today's Market Data ===
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Create a nice info card
info_text = f"""
═══════════════════════════════════════
        📊 TODAY'S MARKET DATA
═══════════════════════════════════════

  💰 Current Price:     ${today_price:.2f}
  📉 Previous Close:    ${prev_close:.2f}
  📊 Daily Change:      {change_pct:+.2f}%

  ─── Technical Indicators ───

  📈 Price Change:      {latest['price_change_pct']:+.2f}%
  📊 High-Low Range:    {latest['high_low_range']:.2f}%
  📉 5-Day MA:          ${latest['ma_5']:.2f}
  🔄 RSI (14-day):      {latest['rsi_14']:.2f}
  📊 Volume Change:     {latest['volume_change']:+.1f}%

  ─── Model Performance ───

  🎯 Accuracy:          {accuracy:.1%}
  ✅ Correct UP:        {tp}
  ✅ Correct DOWN:      {tn}
  ❌ False UP:          {fp}
  ❌ False DOWN:        {fn}

  📅 Last Updated:      {datetime.now().strftime('%Y-%m-%d %H:%M')}
═══════════════════════════════════════
"""

ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))

plt.tight_layout()

# Save the dashboard
filename = f"gold_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"✅ Dashboard saved as: {filename}")

# Display the dashboard
plt.show()

# ============================================================
# STEP 8: PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("🤖 PREDICTION SUMMARY")
print("=" * 60)
print(f"Prediction:      {pred_text}")
print(f"Confidence:      {max(up_prob, down_prob):.1%}")
print(f"Model Accuracy:  {accuracy:.1%}")
print(f"Dashboard saved: {filename}")
print("=" * 60)
print("⚠️  Remember: This is for educational purposes only!")
print("   Financial markets are unpredictable.")