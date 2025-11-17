# INSTITUTIONAL PRESSURE MASTER ALGO - USING YOUR ANGLE ONE SETUP
import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np

warnings.filterwarnings("ignore")

# ---------------- KEEP YOUR EXISTING ANGLE ONE CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False
EXPIRY_RELAX_FACTOR = 0.7
GAMMA_VOL_SPIKE_THRESHOLD = 2.0
DELTA_OI_RATIO = 2.0
MOMENTUM_VOL_AMPLIFIER = 1.5

# --------- KEEP YOUR EXISTING EXPIRIES ---------
EXPIRIES = {
    "NIFTY": "18 NOV 2025",
    "BANKNIFTY": "25 NOV 2025", 
    "SENSEX": "20 NOV 2025"
}

# --------- KEEP YOUR EXISTING ANGLE ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- KEEP YOUR EXISTING TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --------- INSTITUTIONAL PRESSURE THRESHOLDS ---------
INSTITUTIONAL_THRESHOLDS = {
    "SENSEX": {
        "1m": {"min_points": 30, "volume_surge": 1.8, "range_expansion": 25, "efficiency_ratio": 1.3},
        "5m": {"min_points": 45, "volume_surge": 2.0, "range_expansion": 35, "efficiency_ratio": 1.5}
    },
    "BANKNIFTY": {
        "1m": {"min_points": 35, "volume_surge": 1.9, "range_expansion": 30, "efficiency_ratio": 1.4},
        "5m": {"min_points": 50, "volume_surge": 2.2, "range_expansion": 40, "efficiency_ratio": 1.6}
    },
    "NIFTY": {
        "1m": {"min_points": 25, "volume_surge": 1.7, "range_expansion": 20, "efficiency_ratio": 1.3},
        "5m": {"min_points": 40, "volume_surge": 1.9, "range_expansion": 30, "efficiency_ratio": 1.5}
    }
}

# --------- SIGNAL TRACKING ---------
active_signals = {}
signal_cooldown = {
    "1m": 120,   # Monitor next 2 candles (2 minutes)
    "5m": 300    # Monitor next 5min candle (5 minutes)
}
signal_counter = 0
all_generated_signals = []
daily_signals = []

STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- KEEP YOUR EXISTING MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- KEEP YOUR EXISTING STRIKE ROUNDING ---------
def round_strike(index, price):
    try:
        if price is None:
            return None
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": 
            return int(round(price / 100.0) * 100)
        elif index == "SENSEX": 
            return int(round(price / 100.0) * 100)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- KEEP YOUR EXISTING DATA FETCHING ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- KEEP YOUR EXISTING ANGLE ONE TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except:
        return {}

token_map=load_token_map()

# --------- KEEP YOUR EXISTING OPTION PRICE FETCHING ---------
def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token:
        return None
    for _ in range(retries):
        try:
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except:
            time.sleep(delay)
    return None

# --------- KEEP YOUR EXISTING SYMBOL VALIDATION ---------
def validate_option_symbol(index, symbol, strike, opttype):
    try:
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
            
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")
            month_code = expected_dt.strftime("%b").upper()
            expected_pattern = f"SENSEX{year_short}{month_code}"
            return expected_pattern in symbol.upper()
        else:
            expected_pattern = expected_dt.strftime("%d%b%y").upper()
            return expected_pattern in symbol.upper()
            
    except Exception as e:
        print(f"Symbol validation error: {e}")
        return False

def get_option_symbol(index, expiry_str, strike, opttype):
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")
            month_code = dt.strftime("%b").upper()
            day = dt.strftime("%d")
            symbol = f"SENSEX{year_short}{month_code}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        if validate_option_symbol(index, symbol, strike, opttype):
            return symbol
        else:
            print(f"⚠️ Generated symbol validation failed: {symbol}")
            return None
            
    except Exception as e:
        print(f"Error generating symbol: {e}")
        return None

# --------- INSTITUTIONAL PRESSURE ANALYZER ---------
class InstitutionalPressureAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def analyze_institutional_pressure(self, index, df, timeframe):
        """MAIN INSTITUTIONAL PRESSURE ANALYSIS"""
        try:
            if len(df) < 10:
                return None
            
            # Get current candle data
            current_row = df.iloc[-1]
            current_open = self.safe_float(current_row['Open'])
            current_high = self.safe_float(current_row['High'])
            current_low = self.safe_float(current_row['Low'])
            current_close = self.safe_float(current_row['Close'])
            current_volume = self.safe_int(current_row['Volume'])
            
            # Get threshold for this index and timeframe
            thresholds = INSTITUTIONAL_THRESHOLDS[index][timeframe]
            min_points = thresholds["min_points"]
            
            # Calculate candle move
            candle_move = abs(current_close - current_open)
            
            # Skip if below institutional threshold
            if candle_move < min_points:
                return None
            
            # Calculate institutional metrics
            metrics = self.calculate_institutional_metrics(df, current_open, current_high, current_low, current_close, current_volume)
            
            # Check if meets institutional criteria
            if self.is_institutional_signal(metrics, thresholds):
                direction = "CE" if current_close > current_open else "PE"
                return {
                    'direction': direction,
                    'points_moved': round(candle_move, 2),
                    'metrics': metrics,
                    'timeframe': timeframe,
                    'index_price': current_close
                }
            
            return None
            
        except Exception as e:
            print(f"Institutional analysis error: {e}")
            return None
    
    def safe_float(self, value):
        try:
            if hasattr(value, 'item'):
                return float(value.item())
            elif hasattr(value, 'iloc'):
                return float(value.iloc[0])
            else:
                return float(value)
        except:
            return 0.0

    def safe_int(self, value):
        try:
            if hasattr(value, 'item'):
                return int(value.item())
            elif hasattr(value, 'iloc'):
                return int(value.iloc[0])
            else:
                return int(value)
        except:
            return 0

    def calculate_institutional_metrics(self, df, curr_open, curr_high, curr_low, curr_close, curr_volume):
        """Calculate institutional pressure metrics"""
        try:
            # Get previous 5 candles for context
            prev_rows = []
            for i in range(2, 7):
                if len(df) >= i:
                    prev_rows.append(df.iloc[-i])
            
            # Extract previous data
            prev_volumes = []
            prev_closes = []
            prev_ranges_pct = []
            
            for row in prev_rows:
                prev_open = self.safe_float(row['Open'])
                prev_high = self.safe_float(row['High'])
                prev_low = self.safe_float(row['Low'])
                prev_close = self.safe_float(row['Close'])
                prev_volume = self.safe_int(row['Volume'])
                
                prev_volumes.append(prev_volume)
                prev_closes.append(prev_close)
                
                if prev_open > 0:
                    range_pct = (prev_high - prev_low) / prev_open * 100
                    prev_ranges_pct.append(range_pct)
            
            # Handle zero volumes
            base_volume = 50000
            if curr_volume == 0:
                movement_intensity = abs(curr_close - curr_open) / curr_open * 100 if curr_open > 0 else 0
                curr_volume = int(base_volume * (1 + movement_intensity * 8))
            
            # Volume Analysis
            avg_prev_volume = np.mean(prev_volumes) if prev_volumes else base_volume
            volume_surge_ratio = round(curr_volume / max(1, avg_prev_volume), 2)
            
            # Price Efficiency
            current_efficiency = abs(curr_close - curr_open) / (curr_high - curr_low) if (curr_high - curr_low) > 0 else 0
            prev_efficiencies = []
            
            for row in prev_rows:
                prev_open = self.safe_float(row['Open'])
                prev_high = self.safe_float(row['High'])
                prev_low = self.safe_float(row['Low'])
                if prev_high - prev_low > 0:
                    eff = abs(self.safe_float(row['Close']) - prev_open) / (prev_high - prev_low)
                    prev_efficiencies.append(eff)
            
            avg_prev_efficiency = np.mean(prev_efficiencies) if prev_efficiencies else current_efficiency
            efficiency_ratio = round(current_efficiency / max(0.01, avg_prev_efficiency), 2)
            
            # Range Expansion
            current_range_pct = (curr_high - curr_low) / curr_open * 100 if curr_open > 0 else 0
            avg_prev_range = np.mean(prev_ranges_pct) if prev_ranges_pct else current_range_pct
            range_expansion = round(((current_range_pct - avg_prev_range) / max(0.1, avg_prev_range)) * 100, 2)
            
            # Momentum
            if len(prev_closes) >= 3:
                short_momentum = (prev_closes[-1] - prev_closes[-3]) / prev_closes[-3] * 100
                medium_momentum = (prev_closes[-1] - prev_closes[0]) / prev_closes[0] * 100
                momentum_alignment = abs(short_momentum - medium_momentum)
                
                if momentum_alignment < 0.05:
                    momentum_pressure = "STRONG"
                elif momentum_alignment < 0.1:
                    momentum_pressure = "MODERATE"
                else:
                    momentum_pressure = "WEAK"
            else:
                momentum_pressure = "NEUTRAL"
            
            return {
                'volume_surge_ratio': volume_surge_ratio,
                'efficiency_ratio': efficiency_ratio,
                'range_expansion': range_expansion,
                'momentum_pressure': momentum_pressure,
                'current_volume': curr_volume,
                'avg_prev_volume': avg_prev_volume
            }
            
        except Exception as e:
            print(f"Metrics calculation error: {e}")
            return {
                'volume_surge_ratio': 0.0,
                'efficiency_ratio': 0.0,
                'range_expansion': 0.0,
                'momentum_pressure': "NEUTRAL",
                'current_volume': 0,
                'avg_prev_volume': 0
            }
    
    def is_institutional_signal(self, metrics, thresholds):
        """Check if signal meets institutional criteria"""
        try:
            if (metrics['volume_surge_ratio'] >= thresholds['volume_surge'] and
                metrics['efficiency_ratio'] >= thresholds['efficiency_ratio'] and
                metrics['range_expansion'] >= thresholds['range_expansion'] and
                metrics['momentum_pressure'] == "STRONG"):
                return True
            return False
        except:
            return False

# --------- SIGNAL DEDUPLICATION AND COOLDOWN ---------
def can_send_signal(index, timeframe):
    """Check if we can send signal based on cooldown"""
    global active_signals
    
    current_time = time.time()
    signal_key = f"{index}_{timeframe}"
    
    # Check if signal is already active for this index+timeframe
    if signal_key in active_signals:
        signal_data = active_signals[signal_key]
        cooldown_period = signal_cooldown[timeframe]
        
        # Check if still in monitoring period
        if current_time - signal_data['timestamp'] < cooldown_period:
            return False
    
    return True

def update_signal_tracking(index, timeframe, signal_id):
    """Update tracking for sent signals"""
    global active_signals
    
    signal_key = f"{index}_{timeframe}"
    active_signals[signal_key] = {
        'signal_id': signal_id,
        'timestamp': time.time(),
        'timeframe': timeframe
    }

def clear_completed_signal(signal_id):
    """Clear signal from active tracking when completed"""
    global active_signals
    active_signals = {k: v for k, v in active_signals.items() if v['signal_id'] != signal_id}

# --------- UPDATED SIGNAL GENERATION ---------
def analyze_index_signal(index):
    """Analyze both 1min and 5min timeframes for institutional pressure"""
    analyzer = InstitutionalPressureAnalyzer()
    
    # Check 1min data first
    df1 = fetch_index_data(index, "1m", "1d")
    if df1 is not None and len(df1) >= 10:
        signal_1m = analyzer.analyze_institutional_pressure(index, df1, "1m")
        if signal_1m and can_send_signal(index, "1m"):
            return signal_1m, df1
    
    # Check 5min data
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is not None and len(df5) >= 10:
        signal_5m = analyzer.analyze_institutional_pressure(index, df5, "5m")
        if signal_5m and can_send_signal(index, "5m"):
            return signal_5m, df5
    
    return None

# --------- UPDATED SIGNAL SENDING ---------
def send_signal(index, signal_data, df):
    global signal_counter, all_generated_signals
    
    direction = signal_data['direction']
    timeframe = signal_data['timeframe']
    metrics = signal_data['metrics']
    index_price = signal_data['index_price']
    
    # Calculate strike
    strike = round_strike(index, index_price)
    if strike is None:
        return
    
    # Generate symbol
    symbol = get_option_symbol(index, EXPIRIES[index], strike, direction)
    if symbol is None:
        return
    
    # Fetch option price with 1 rupee precision
    option_price = fetch_option_price(symbol)
    if not option_price:
        return
    
    entry = round(option_price)  # 1 rupee increments
    
    # Calculate institutional targets
    points_moved = signal_data['points_moved']
    if direction == "CE":
        base_move = max(points_moved * 0.8, 40)  # Minimum 40 points
        targets = [
            entry + int(base_move * 1.0),
            entry + int(base_move * 1.8),
            entry + int(base_move * 2.8),
            entry + int(base_move * 4.0)
        ]
        sl = entry - int(base_move * 0.8)
    else:  # PE
        base_move = max(points_moved * 0.8, 40)  # Minimum 40 points
        targets = [
            entry + int(base_move * 1.0),
            entry + int(base_move * 1.8),
            entry + int(base_move * 2.8),
            entry + int(base_move * 4.0)
        ]
        sl = entry - int(base_move * 0.8)
    
    # Ensure 1 rupee increments for all prices
    entry = int(entry)
    targets = [int(t) for t in targets]
    sl = int(sl)
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    # Update signal tracking
    update_signal_tracking(index, timeframe, signal_id)
    
    # Prepare signal data for EOD report
    signal_record = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": direction,
        "strategy": f"INSTITUTIONAL_PRESSURE_{timeframe}",
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": False,
        "index_price": index_price,
        "timeframe": timeframe,
        "metrics": metrics,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "final_pnl": "0"
    }
    
    all_generated_signals.append(signal_record)
    
    # Create message with calculated values
    msg = (f"🟢 {index} {strike} {direction}\n"
           f"SYMBOL: {symbol}\n"
           f"ABOVE {entry}\n"
           f"TARGETS: {targets_str}\n"
           f"SL: {sl}\n"
           f"FAKEOUT: NO\n"
           f"STRATEGY: INSTITUTIONAL PRESSURE {timeframe}\n"
           f"SIGNAL ID: {signal_id}\n"
           f"─────────────────────────────\n"
           f"📊 CALCULATED VALUES:\n"
           f"• Points Moved: {signal_data['points_moved']}\n"
           f"• Volume Surge: {metrics['volume_surge_ratio']}x\n"
           f"• Price Efficiency: {metrics['efficiency_ratio']}x\n"
           f"• Range Expansion: {metrics['range_expansion']}%\n"
           f"• Momentum: {metrics['momentum_pressure']}\n"
           f"• Index Price: {index_price}\n"
           f"• Timeframe: {timeframe}\n"
           f"─────────────────────────────")
    
    thread_id = send_telegram(msg)
    
    # Start monitoring
    monitor_signal(symbol, entry, targets, sl, timeframe, thread_id, signal_id, signal_record)

# --------- UPDATED MONITORING WITH TIMEFRAME-BASED DURATION ---------
def monitor_signal(symbol, entry, targets, sl, timeframe, thread_id, signal_id, signal_record):
    def monitoring_thread():
        global daily_signals
        
        monitoring_duration = signal_cooldown[timeframe]  # 2min for 1m, 5min for 5m
        start_time = time.time()
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        
        while time.time() - start_time < monitoring_duration:
            if should_stop_trading():
                break
                
            price = fetch_option_price(symbol)
            if price:
                price = int(price)  # 1 rupee increments
                
                if price > max_price_reached:
                    max_price_reached = price
                    if price > entry:
                        send_telegram(f"📈 {symbol} moving up: {price}", reply_to=thread_id)
                
                # Check targets
                for i, target in enumerate(targets):
                    if price >= target and not targets_hit[i]:
                        send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                        targets_hit[i] = True
                        signal_record['targets_hit'] = sum(targets_hit)
                
                # Check SL
                if price <= sl:
                    send_telegram(f"🛑 {symbol}: SL hit at ₹{sl}", reply_to=thread_id)
                    break
            
            time.sleep(10)  # Check every 10 seconds
        
        # Monitoring period completed
        send_telegram(f"⏰ {symbol}: Monitoring period completed", reply_to=thread_id)
        
        # Calculate final P&L
        final_pnl = max_price_reached - entry
        signal_record.update({
            "max_price_reached": max_price_reached,
            "targets_hit": sum(targets_hit),
            "final_pnl": f"+{final_pnl}" if final_pnl > 0 else f"{final_pnl}"
        })
        
        daily_signals.append(signal_record)
        clear_completed_signal(signal_id)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# --------- KEEP YOUR EXISTING EOD REPORT SYSTEM ---------
def send_individual_signal_reports():
    global all_generated_signals, daily_signals
    
    all_signals = daily_signals + all_generated_signals
    
    seen_ids = set()
    unique_signals = []
    for signal in all_signals:
        sid = signal.get('signal_id')
        if not sid:
            continue
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_signals.append(signal)
    
    if not unique_signals:
        send_telegram("📊 END OF DAY REPORT\nNo institutional signals generated today.")
        return
    
    send_telegram(f"🕒 INSTITUTIONAL PRESSURE EOD REPORT - {(datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y')}\n"
                  f"📈 Total Signals: {len(unique_signals)}\n"
                  f"─────────────────────────────")
    
    for i, signal in enumerate(unique_signals, 1):
        targets_hit_list = []
        if signal.get('targets_hit', 0) > 0:
            for j in range(signal.get('targets_hit', 0)):
                if j < len(signal.get('targets', [])):
                    targets_hit_list.append(str(signal['targets'][j]))
        
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
        msg = (f"📊 INSTITUTIONAL SIGNAL #{i}\n"
               f"─────────────────────────────\n"
               f"🕒 Time: {signal.get('timestamp','?')}\n"
               f"📈 Index: {signal.get('index','?')}\n"
               f"🎯 Strike: {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"⏰ Timeframe: {signal.get('timeframe','?')}\n\n"
               
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]} // {targets_for_disp[1]} // {targets_for_disp[2]} // {targets_for_disp[3]}\n"
               f"🛑 STOP LOSS: ₹{signal.get('sl','?')}\n\n"
               
               f"📊 PERFORMANCE:\n"
               f"• Targets Hit: {signal.get('targets_hit', 0)}/4\n")
        
        if targets_hit_list:
            msg += f"• Targets Achieved: {', '.join(targets_hit_list)}\n"
        
        msg += (f"• Max Price: ₹{signal.get('max_price_reached', signal.get('entry_price','?'))}\n"
                f"• Final P&L: {signal.get('final_pnl', '0')} points\n"
                f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    # Calculate summary
    total_pnl = 0.0
    successful_trades = 0
    for signal in unique_signals:
        pnl_str = signal.get("final_pnl", "0")
        try:
            if pnl_str.startswith("+"):
                total_pnl += float(pnl_str[1:])
                successful_trades += 1
            elif pnl_str.startswith("-"):
                total_pnl -= float(pnl_str[1:])
        except:
            pass
    
    summary_msg = (f"📈 INSTITUTIONAL PERFORMANCE SUMMARY\n"
                   f"─────────────────────────────\n"
                   f"• Total Signals: {len(unique_signals)}\n"
                   f"• Successful Trades: {successful_trades}\n"
                   f"• Success Rate: {(successful_trades/len(unique_signals))*100:.1f}%\n"
                   f"• Total P&L: ₹{total_pnl:+.2f}\n"
                   f"─────────────────────────────")
    
    send_telegram(summary_msg)
    send_telegram("✅ INSTITUTIONAL TRADING DAY COMPLETED! See you tomorrow! 🚀")

# --------- UPDATED TRADE THREAD ---------
def trade_thread(index):
    """Generate institutional pressure signals"""
    result = analyze_index_signal(index)
    
    if result:
        signal_data, df = result
        send_signal(index, signal_data, df)

# --------- UPDATED MAIN LOOP ---------
def run_algo_parallel():
    if not is_market_open(): 
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed - Institutional Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_telegram("📊 GENERATING INSTITUTIONAL EOD REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ EOD Report Error: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()
            EOD_REPORT_SENT = True
            
        return
        
    threads = []
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    
    for index in indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- KEEP YOUR EXISTING MAIN LOOP STRUCTURE ---------
while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market closed - Institutional Algorithm waiting...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING INSTITUTIONAL EOD REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 INSTITUTIONAL PRESSURE ALGO STARTED\n"
                         "✅ SENSEX: 30+ points, 1.8x volume\n"
                         "✅ BANKNIFTY: 35+ points, 1.9x volume\n" 
                         "✅ NIFTY: 25+ points, 1.7x volume\n"
                         "✅ Smart Monitoring: 2min(1m) / 5min(5m)\n"
                         "✅ 1 Rupee Increments\n"
                         "✅ Institutional Accuracy Targets")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing! Preparing Institutional EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING INSTITUTIONAL TRADES...")
                time.sleep(20)
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ EOD Report Error: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Institutional algo error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
