"""
GOLD TRADING PREDICTOR - Professional Trading Dashboard
======================================================
Generates a PNG dashboard with real-time trading signals.
DISCLAIMER: For educational purposes. Backtest before using real money.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏆 GOLD TRADING PREDICTOR - Professional Dashboard")
print("=" * 80)
print("📊 Generating trading signal and dashboard PNG...")
print()

# ============================================================
# STEP 1: FETCH REAL-TIME DATA
# ============================================================

def fetch_gold_data(period="5y", interval="1d"):
    """Fetch gold data for analysis"""
    print("📥 Fetching real-time gold data...")
    
    tickers = ["GC=F", "XAUUSD=X", "GLD"]
    df = None
    
    for ticker in tickers:
        try:
            gold = yf.Ticker(ticker)
            df = gold.history(period=period, interval=interval)
            if not df.empty and len(df) > 100:
                print(f"✅ Using {ticker}")
                break
        except:
            continue
    
    if df is None or df.empty:
        raise Exception("Failed to fetch data")
    
    return df

# Fetch daily data
df = fetch_gold_data(period="5y", interval="1d")
print(f"✅ {len(df)} days of data")
print(f"📅 {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"💰 Current Price: ${df['Close'].iloc[-1]:.2f}")
print()

# ============================================================
# STEP 2: ADVANCED FEATURES
# ============================================================

def create_trading_features(data):
    """Create features specifically for trading signals"""
    df = data.copy()
    
    # Price returns at multiple timeframes
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{period}d'] = df['Close'].pct_change(periods=period) * 100
    
    # Moving averages (key trading levels)
    for period in [7, 20, 50, 100, 200]:
        df[f'ma_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'ma_ratio_{period}'] = (df['Close'] - df[f'ma_{period}']) / df[f'ma_{period}'] * 100
    
    # Price position relative to range
    df['range_high'] = df['High'].rolling(20).max()
    df['range_low'] = df['Low'].rolling(20).min()
    df['range_position'] = (df['Close'] - df['range_low']) / (df['range_high'] - df['range_low']) * 100
    
    # Volatility
    df['volatility'] = df['Close'].pct_change().rolling(20).std() * 100
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_position'] = (df['Close'] - df['bb_mid']) / (df['bb_std'] * 2) * 100
    
    # Volume
    df['volume_avg'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean() * 100
    
    # ATR (Average True Range) - for stop loss
    df['high_low'] = df['High'] - df['Low']
    df['high_close'] = abs(df['High'] - df['Close'].shift())
    df['low_close'] = abs(df['Low'] - df['Close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Support and resistance levels
    df['support_1'] = df['Low'].rolling(20).min()
    df['resistance_1'] = df['High'].rolling(20).max()
    df['support_2'] = df['Low'].rolling(50).min()
    df['resistance_2'] = df['High'].rolling(50).max()
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

# Create features
df = create_trading_features(df)
print(f"✅ Created {len(df)} trading data points")
print()

# ============================================================
# STEP 3: TRAIN PREDICTIVE MODEL
# ============================================================

def train_trading_model(df):
    """Train model for trading signals"""
    
    # Define features
    feature_cols = [col for col in df.columns if col not in [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Dividends', 'Stock Splits', 'range_high', 'range_low',
        'ema_12', 'ema_26', 'bb_mid', 'bb_std', 'high_low', 
        'high_close', 'low_close', 'tr', 'support_1', 
        'resistance_1', 'support_2', 'resistance_2',
        'volume_avg'
    ]]
    
    # Target: 3-day forward return (trading horizon)
    df['target_3d'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    
    # Use 3-day target for shorter-term trading
    y = df['target_3d']
    X = df[feature_cols]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Multiple models
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42
        ),
        'gbm': GradientBoostingClassifier(
            n_estimators=150, 
            learning_rate=0.05,
            random_state=42
        ),
        'lr': LogisticRegression(
            C=0.1, 
            max_iter=1000,
            random_state=42
        )
    }
    
    # Train and find best
    best_acc = 0
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"   {name}: {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
    
    print(f"✅ Best model accuracy: {best_acc:.2%}")
    
    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance = None
    
    return best_model, scaler, feature_cols, best_acc, importance

model, scaler, feature_cols, model_accuracy, importance = train_trading_model(df)
print()

# ============================================================
# STEP 4: GENERATE TRADING SIGNAL
# ============================================================

def generate_trading_signal(df, model, scaler, feature_cols):
    """Generate actionable trading signal"""
    
    # Latest data
    latest = df.iloc[-1]
    today_price = latest['Close']
    
    # Features for prediction
    latest_features = latest[feature_cols].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_features)
    
    # Probability
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(latest_scaled)[0]
        up_prob = prob[1]
        down_prob = prob[0]
    else:
        pred = model.predict(latest_scaled)[0]
        up_prob = 1.0 if pred == 1 else 0.0
        down_prob = 1.0 - up_prob
    
    # Trading signal
    confidence = max(up_prob, down_prob)
    
    if confidence > 0.60:
        if up_prob > 0.5:
            signal = "BUY"
            signal_color = "#00B894"
        else:
            signal = "SELL"
            signal_color = "#E74C3C"
    elif confidence > 0.55:
        if up_prob > 0.5:
            signal = "WEAK BUY"
            signal_color = "#FDCB6E"
        else:
            signal = "WEAK SELL"
            signal_color = "#FDCB6E"
    else:
        signal = "NEUTRAL"
        signal_color = "#95A5A6"
    
    # Calculate risk levels
    atr = latest['atr']
    price = latest['Close']
    
    # Stop Loss (1.5x ATR)
    stop_loss_distance = atr * 1.5
    if signal in ["BUY", "WEAK BUY"]:
        stop_loss = price - stop_loss_distance
        take_profit_1 = price + (stop_loss_distance * 1.5)
        take_profit_2 = price + (stop_loss_distance * 3)
        risk_reward = 1.5
    elif signal in ["SELL", "WEAK SELL"]:
        stop_loss = price + stop_loss_distance
        take_profit_1 = price - (stop_loss_distance * 1.5)
        take_profit_2 = price - (stop_loss_distance * 3)
        risk_reward = 1.5
    else:
        stop_loss = price
        take_profit_1 = price
        take_profit_2 = price
        risk_reward = 0
    
    # Position sizing
    risk_per_trade = 1.0
    position_size = risk_per_trade / ((stop_loss_distance / price) * 100)
    
    return {
        'signal': signal,
        'signal_color': signal_color,
        'up_prob': up_prob,
        'down_prob': down_prob,
        'confidence': confidence,
        'price': price,
        'atr': atr,
        'stop_loss': stop_loss,
        'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2,
        'risk_reward': risk_reward,
        'position_size': position_size,
        'risk_per_trade': risk_per_trade
    }

signal_data = generate_trading_signal(df, model, scaler, feature_cols)

# ============================================================
# STEP 5: BACKTEST PERFORMANCE
# ============================================================

def backtest_strategy(df, model, scaler, feature_cols):
    """Backtest the strategy on historical data"""
    
    print("\n📊 Running backtest...")
    
    # Generate predictions for all data points
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_scaled)
        predictions = (probs[:, 1] > 0.5).astype(int)
    else:
        predictions = model.predict(X_scaled)
    
    # Create signals
    df['signal'] = predictions
    
    # Calculate returns
    df['future_return'] = df['Close'].shift(-3) / df['Close'] - 1
    df['strategy_return'] = df['future_return'] * (2 * df['signal'] - 1)
    
    # Performance metrics
    total_return = df['strategy_return'].sum() * 100
    win_rate = (df['strategy_return'] > 0).mean() * 100
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252) if df['strategy_return'].std() > 0 else 0
    
    cum_returns = df['strategy_return'].cumsum()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max)
    max_drawdown = drawdown.min() * 100
    
    print(f"   Total Return:  {total_return:.2f}%")
    print(f"   Win Rate:      {win_rate:.1f}%")
    print(f"   Sharpe Ratio:  {sharpe:.2f}")
    print(f"   Max Drawdown:  {max_drawdown:.2f}%")
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

backtest_results = backtest_strategy(df, model, scaler, feature_cols)
print()

# ============================================================
# STEP 6: GENERATE PNG DASHBOARD
# ============================================================

print("📊 Generating professional PNG dashboard...")

# Create figure
fig = plt.figure(figsize=(20, 12), facecolor='#0a0a1a')
fig.suptitle('', fontsize=20, fontweight='bold', y=0.98)

# Set dark theme
plt.rcParams['text.color'] = '#ffffff'
plt.rcParams['axes.labelcolor'] = '#ffffff'
plt.rcParams['xtick.color'] = '#ffffff'
plt.rcParams['ytick.color'] = '#ffffff'

# === HEADER ===
ax_header = plt.axes([0, 0.95, 1, 0.05])
ax_header.axis('off')
header_text = "💰 GOLD TRADING PREDICTOR"
ax_header.text(0.5, 0.5, header_text, fontsize=30, fontweight='bold',
               color='#FDCB6E', ha='center', va='center')

# Subtitle
ax_sub = plt.axes([0, 0.90, 1, 0.05])
ax_sub.axis('off')
sub_text = f"XAU/USD • Signal Generated: {datetime.now().strftime('%B %d, %Y %H:%M')} • 5-Year Data"
ax_sub.text(0.5, 0, sub_text, fontsize=14, color='#BDC3C7', ha='center', va='center')

# === TOP LEFT: Price Chart with Technicals ===
ax1 = plt.subplot(2, 3, 1, facecolor='#1a1a2e')
ax1.set_facecolor('#1a1a2e')

# Plot last 100 days
recent_data = df[-100:]

# Price line
ax1.plot(recent_data.index, recent_data['Close'], 
         color='#FDCB6E', linewidth=2.5, label='Gold Price')

# Moving averages
ax1.plot(recent_data.index, recent_data['ma_20'], 
         color='#3498DB', linewidth=1.5, alpha=0.7, label='MA 20')
ax1.plot(recent_data.index, recent_data['ma_50'], 
         color='#E74C3C', linewidth=1.5, alpha=0.7, label='MA 50')

# Current price line
current_price = signal_data['price']
ax1.axhline(y=current_price, color='#FDCB6E', linestyle='--', 
            linewidth=1, alpha=0.5)

# Add current price label
ax1.text(recent_data.index[-1], current_price + 5, f'${current_price:.2f}', 
         fontsize=11, fontweight='bold', color='#FDCB6E', ha='right')

# Bollinger Bands
ax1.fill_between(recent_data.index, 
                  recent_data['bb_mid'] - 2*recent_data['bb_std'],
                  recent_data['bb_mid'] + 2*recent_data['bb_std'],
                  color='#3498DB', alpha=0.1)

ax1.set_title('Gold Price with Technicals', fontsize=14, fontweight='bold', color='#FDCB6E', pad=15)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Price (USD/oz)', fontsize=11)
ax1.legend(loc='upper left', framealpha=0.3, facecolor='#1a1a2e', fontsize=10)
ax1.grid(True, alpha=0.1)
ax1.tick_params(axis='x', rotation=0, colors='#BDC3C7')
ax1.tick_params(axis='y', colors='#BDC3C7')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#BDC3C7')
ax1.spines['left'].set_color('#BDC3C7')

# === TOP MIDDLE: Trading Signal ===
ax2 = plt.subplot(2, 3, 2, facecolor='#1a1a2e')

# Big signal text
signal = signal_data['signal']
color = signal_data['signal_color']

# Background glow
if signal in ["BUY", "WEAK BUY"]:
    glow_color = 'green'
elif signal in ["SELL", "WEAK SELL"]:
    glow_color = 'red'
else:
    glow_color = 'gray'

ax2.text(0.5, 0.7, signal, fontsize=44, fontweight='bold',
         color=color, ha='center', va='center', transform=ax2.transAxes)

# Confidence
ax2.text(0.5, 0.5, f'Confidence: {signal_data["confidence"]:.1%}', 
         fontsize=16, ha='center', va='center', transform=ax2.transAxes,
         color='#BDC3C7')

# Probability bars
ax2.text(0.25, 0.35, f'UP: {signal_data["up_prob"]:.1%}', fontsize=13, 
         ha='center', va='center', transform=ax2.transAxes, 
         color='#00B894', fontweight='bold')
ax2.text(0.75, 0.35, f'DOWN: {signal_data["down_prob"]:.1%}', fontsize=13, 
         ha='center', va='center', transform=ax2.transAxes, 
         color='#E74C3C', fontweight='bold')

# Progress bars
ax2.barh(0.25, signal_data['up_prob'], color='#00B894', height=0.04, alpha=0.8, 
         left=0, transform=ax2.transAxes)
ax2.barh(0.25, signal_data['down_prob'], color='#E74C3C', height=0.04, alpha=0.8, 
         left=signal_data['up_prob'], transform=ax2.transAxes)

# Model info
ax2.text(0.5, 0.12, f'Model Accuracy: {model_accuracy:.1%} • 3-Day Forecast', 
         fontsize=11, ha='center', va='center', transform=ax2.transAxes,
         color='#95A5A6')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Trading Signal', fontsize=16, fontweight='bold', color='#FDCB6E', pad=15)
ax2.axis('off')

# === TOP RIGHT: Trade Plan ===
ax3 = plt.subplot(2, 3, 3, facecolor='#1a1a2e')
ax3.axis('off')

# Trade plan box
trade_plan = f"""
╔══════════════════════════════════════╗
║         📊 TRADE PLAN               ║
╠══════════════════════════════════════╣
║                                      ║
║   Entry Price:    ${signal_data['price']:.2f}        ║
║   Stop Loss:      ${signal_data['stop_loss']:.2f}        ║
║   Take Profit 1:  ${signal_data['take_profit_1']:.2f}        ║
║   Take Profit 2:  ${signal_data['take_profit_2']:.2f}        ║
║                                      ║
║   Risk/Reward:    1:{signal_data['risk_reward']:.1f}         ║
║   Position Size:  {signal_data['position_size']:.2f} units   ║
║   Risk per Trade: {signal_data['risk_per_trade']:.1f}%       ║
║                                      ║
║   ATR (14-day):    ${signal_data['atr']:.2f}        ║
╚══════════════════════════════════════╝
"""

ax3.text(0.5, 0.5, trade_plan, transform=ax3.transAxes, fontsize=12,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', linespacing=1.3,
         color='#BDC3C7')

ax3.set_title('Trade Plan & Risk Management', fontsize=14, fontweight='bold', color='#FDCB6E', pad=15)

# === BOTTOM LEFT: Market Data ===
ax4 = plt.subplot(2, 3, 4, facecolor='#1a1a2e')
ax4.axis('off')

latest = df.iloc[-1]

market_data = f"""
╔══════════════════════════════════════╗
║         📈 MARKET DATA              ║
╠══════════════════════════════════════╣
║                                      ║
║   Current Price:    ${latest['Close']:.2f}     ║
║   Daily Change:     {latest['ret_1d']:+.2f}%      ║
║   High:             ${latest['High']:.2f}     ║
║   Low:              ${latest['Low']:.2f}     ║
║                                      ║
║   RSI (14):         {latest['rsi']:.1f}         ║
║   MACD:             {latest['macd']:.3f}       ║
║   Volatility:       {latest['volatility']:.2f}%      ║
║   Volume:           {latest['Volume']/1000:.0f}K        ║
╚══════════════════════════════════════╝
"""

ax4.text(0.5, 0.5, market_data, transform=ax4.transAxes, fontsize=11,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', linespacing=1.3,
         color='#BDC3C7')

ax4.set_title('Market Conditions', fontsize=14, fontweight='bold', color='#FDCB6E', pad=15)

# === BOTTOM MIDDLE: Feature Importance ===
ax5 = plt.subplot(2, 3, 5, facecolor='#1a1a2e')

if importance is not None:
    top_features = importance.head(8)
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax5.barh(y_pos, top_features['importance'], 
             color=plt.cm.YlOrRd(top_features['importance']/top_features['importance'].max()),
             alpha=0.8)
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([f.replace('_', ' ').title() for f in top_features['feature']], 
                        fontsize=10, color='#BDC3C7')
    ax5.set_xlabel('Importance', fontsize=11, color='#BDC3C7')
    ax5.set_title('Top Features', fontsize=14, fontweight='bold', color='#FDCB6E', pad=15)
    ax5.grid(True, alpha=0.1, axis='x')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_color('#BDC3C7')
    ax5.spines['left'].set_color('#BDC3C7')
    ax5.tick_params(colors='#BDC3C7')

# === BOTTOM RIGHT: Backtest Performance ===
ax6 = plt.subplot(2, 3, 6, facecolor='#1a1a2e')
ax6.axis('off')

performance = f"""
╔══════════════════════════════════════╗
║      📊 BACKTEST PERFORMANCE        ║
╠══════════════════════════════════════╣
║                                      ║
║   Total Return:    {backtest_results['total_return']:+.1f}%        ║
║   Win Rate:        {backtest_results['win_rate']:.1f}%         ║
║   Sharpe Ratio:    {backtest_results['sharpe']:.2f}         ║
║   Max Drawdown:    {backtest_results['max_drawdown']:.1f}%        ║
║                                      ║
║   ─── Performance Rating ───        ║
║                                      ║
║   {'⭐' * min(5, int(backtest_results['sharpe'] * 2))}  ({'Good' if backtest_results['sharpe'] > 1 else 'Fair' if backtest_results['sharpe'] > 0.5 else 'Poor'})
╚══════════════════════════════════════╝
"""

ax6.text(0.5, 0.5, performance, transform=ax6.transAxes, fontsize=11,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', linespacing=1.3,
         color='#BDC3C7')

ax6.set_title('Strategy Performance', fontsize=14, fontweight='bold', color='#FDCB6E', pad=15)

# === FOOTER ===
ax_footer = plt.axes([0, 0.01, 1, 0.04])
ax_footer.axis('off')

footer_text = "⚠️ DISCLAIMER: This is a TOOL for decision-making, not financial advice. Always use stop losses and proper risk management."
ax_footer.text(0.5, 0, footer_text, fontsize=10, color='#95A5A6', 
               ha='center', va='center', style='italic')

plt.tight_layout()

# Save PNG
filename = f"gold_trading_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
print(f"✅ Dashboard PNG saved: {filename}")

plt.show()

# ============================================================
# STEP 7: PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("📈 TRADING SIGNAL SUMMARY")
print("=" * 80)
print(f"Signal:          {signal_data['signal']}")
print(f"Confidence:      {signal_data['confidence']:.1%}")
print(f"Entry Price:     ${signal_data['price']:.2f}")
print(f"Stop Loss:       ${signal_data['stop_loss']:.2f}")
print(f"Take Profit 1:   ${signal_data['take_profit_1']:.2f}")
print(f"Take Profit 2:   ${signal_data['take_profit_2']:.2f}")
print(f"Risk/Reward:     1:{signal_data['risk_reward']:.1f}")
print(f"Model Accuracy:  {model_accuracy:.1%}")
print(f"Dashboard PNG:   {filename}")
print("=" * 80)
print("⚠️  Remember: This is a tool, not financial advice!")
print("=" * 80)