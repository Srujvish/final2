# INSTITUTIONAL PRESSURE MASTER ALGO - PROFESSIONAL TG FORMAT
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

# ---------------- CONFIGURATION ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRIES = {
    "NIFTY": "18 NOV 2025",
    "BANKNIFTY": "25 NOV 2025", 
    "SENSEX": "20 NOV 2025"
}

# Angel One Login
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --------- INSTITUTIONAL THRESHOLDS ---------
INSTITUTIONAL_THRESHOLDS = {
    "SENSEX": {
        "1m": {"min_points": 15, "volume_surge": 1.3, "range_expansion": 10, "efficiency_ratio": 1.1},
        "5m": {"min_points": 25, "volume_surge": 1.5, "range_expansion": 15, "efficiency_ratio": 1.2}
    },
    "BANKNIFTY": {
        "1m": {"min_points": 20, "volume_surge": 1.4, "range_expansion": 12, "efficiency_ratio": 1.1},
        "5m": {"min_points": 30, "volume_surge": 1.6, "range_expansion": 18, "efficiency_ratio": 1.3}
    },
    "NIFTY": {
        "1m": {"min_points": 12, "volume_surge": 1.2, "range_expansion": 8, "efficiency_ratio": 1.1},
        "5m": {"min_points": 20, "volume_surge": 1.4, "range_expansion": 12, "efficiency_ratio": 1.2}
    }
}

# --------- SIGNAL TRACKING ---------
active_signals = {}
signal_cooldown = {"1m": 120, "5m": 300}
signal_counter = 0
all_generated_signals = []
daily_signals = []

STARTED_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- MARKET HOURS ---------
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

# --------- DATA FETCHING ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN"
    }
    try:
        df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
        return None if df.empty else df
    except Exception as e:
        print(f"Data fetch error for {index}: {e}")
        return None

# --------- STRIKE CALCULATION ---------
def round_strike(index, price):
    try:
        price = float(price)
        if index == "NIFTY": return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": return int(round(price / 100.0) * 100)
        elif index == "SENSEX": return int(round(price / 100.0) * 100)
        else: return int(round(price / 50.0) * 50)
    except: return None

# --------- OPTION SYMBOL GENERATION ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")
            month_code = dt.strftime("%b").upper()
            symbol = f"SENSEX{year_short}{month_code}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        return symbol
    except Exception as e:
        print(f"Error generating symbol: {e}")
        return None

# --------- TOKEN MAPPING ---------
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

def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token: return None
    for _ in range(retries):
        try:
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except: time.sleep(delay)
    return None

# --------- ENHANCED INSTITUTIONAL ANALYZER ---------
class InstitutionalPressureAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def safe_float(self, value):
        try:
            if hasattr(value, 'item'): return float(value.item())
            elif hasattr(value, 'iloc'): return float(value.iloc[0])
            else: return float(value)
        except: return 0.0

    def safe_int(self, value):
        try:
            if hasattr(value, 'item'): return int(value.item())
            elif hasattr(value, 'iloc'): return int(value.iloc[0])
            else: return int(value)
        except: return 0

    def analyze_previous_candles(self, df, num_candles=3):
        """Analyze previous candles for context"""
        previous_data = []
        for i in range(2, 2 + num_candles):
            if len(df) >= i:
                row = df.iloc[-i]
                prev_open = self.safe_float(row['Open'])
                prev_high = self.safe_float(row['High'])
                prev_low = self.safe_float(row['Low'])
                prev_close = self.safe_float(row['Close'])
                prev_volume = self.safe_int(row['Volume'])
                
                direction = "GREEN" if prev_close > prev_open else "RED"
                points_moved = abs(prev_close - prev_open)
                candle_range = prev_high - prev_low
                
                previous_data.append({
                    'timestamp': row.name.strftime('%H:%M:%S') if hasattr(row.name, 'strftime') else str(row.name),
                    'direction': direction,
                    'points_moved': round(points_moved, 2),
                    'open': round(prev_open, 2),
                    'high': round(prev_high, 2),
                    'low': round(prev_low, 2),
                    'close': round(prev_close, 2),
                    'range': round(candle_range, 2),
                    'volume': prev_volume
                })
        return previous_data

    def calculate_institutional_score(self, metrics):
        """Calculate institutional score 0-100"""
        score = 0
        
        # Volume surge (max 30 points)
        if metrics['volume_surge_ratio'] >= 2.0: score += 30
        elif metrics['volume_surge_ratio'] >= 1.5: score += 20
        elif metrics['volume_surge_ratio'] >= 1.2: score += 10
        
        # Efficiency (max 25 points)
        if metrics['efficiency_ratio'] >= 1.5: score += 25
        elif metrics['efficiency_ratio'] >= 1.2: score += 15
        elif metrics['efficiency_ratio'] >= 1.0: score += 5
        
        # Range expansion (max 20 points)
        if metrics['range_expansion'] >= 20: score += 20
        elif metrics['range_expansion'] >= 10: score += 10
        elif metrics['range_expansion'] >= 5: score += 5
        
        # Momentum (max 25 points)
        if metrics['momentum_pressure'] == "STRONG": score += 25
        elif metrics['momentum_pressure'] == "MODERATE": score += 15
        
        return min(score, 100)

    def get_pressure_type(self, score):
        if score >= 80: return "HEAVY_INSTITUTIONAL"
        elif score >= 60: return "MODERATE_INSTITUTIONAL" 
        elif score >= 40: return "LIGHT_INSTITUTIONAL"
        else: return "RETAIL_VOLATILITY"

    def get_confidence_level(self, score):
        if score >= 70: return "HIGH"
        elif score >= 50: return "MEDIUM"
        else: return "LOW"

    def analyze_institutional_pressure(self, index, df, timeframe):
        """MAIN INSTITUTIONAL PRESSURE ANALYSIS"""
        try:
            if len(df) < 10: return None
            
            current_row = df.iloc[-1]
            current_open = self.safe_float(current_row['Open'])
            current_high = self.safe_float(current_row['High'])
            current_low = self.safe_float(current_row['Low'])
            current_close = self.safe_float(current_row['Close'])
            current_volume = self.safe_int(current_row['Volume'])
            
            thresholds = INSTITUTIONAL_THRESHOLDS[index][timeframe]
            candle_move = abs(current_close - current_open)
            
            if candle_move < thresholds["min_points"]: return None
            
            # Calculate metrics
            metrics = self.calculate_institutional_metrics(df, current_open, current_high, current_low, current_close, current_volume)
            
            if self.is_institutional_signal(metrics, thresholds):
                direction = "CE" if current_close > current_open else "PE"
                tg_direction = "GREEN" if current_close > current_open else "RED"
                
                # Calculate institutional score
                institutional_score = self.calculate_institutional_score(metrics)
                pressure_type = self.get_pressure_type(institutional_score)
                confidence = self.get_confidence_level(institutional_score)
                
                # Get previous candles analysis
                previous_candles = self.analyze_previous_candles(df, 3)
                
                return {
                    'direction': direction,
                    'tg_direction': tg_direction,
                    'points_moved': round(candle_move, 2),
                    'candle_range': round(current_high - current_low, 2),
                    'volume': current_volume,
                    'metrics': metrics,
                    'timeframe': timeframe,
                    'index_price': current_close,
                    'timestamp': current_row.name.strftime('%H:%M:%S') if hasattr(current_row.name, 'strftime') else str(current_row.name),
                    'institutional_score': institutional_score,
                    'pressure_type': pressure_type,
                    'confidence': confidence,
                    'previous_candles': previous_candles
                }
            
            return None
            
        except Exception as e:
            print(f"Institutional analysis error: {e}")
            return None

    def calculate_institutional_metrics(self, df, curr_open, curr_high, curr_low, curr_close, curr_volume):
        """Calculate institutional pressure metrics"""
        try:
            prev_rows = []
            for i in range(2, 7):
                if len(df) >= i: prev_rows.append(df.iloc[-i])
            
            prev_volumes = [self.safe_int(row['Volume']) for row in prev_rows]
            prev_closes = [self.safe_float(row['Close']) for row in prev_rows]
            prev_ranges_pct = []
            
            for row in prev_rows:
                prev_open = self.safe_float(row['Open'])
                prev_high = self.safe_float(row['High'])
                prev_low = self.safe_float(row['Low'])
                if prev_open > 0:
                    range_pct = (prev_high - prev_low) / prev_open * 100
                    prev_ranges_pct.append(range_pct)
            
            # Volume Analysis
            base_volume = 50000
            if curr_volume == 0:
                movement_intensity = abs(curr_close - curr_open) / curr_open * 100 if curr_open > 0 else 0
                curr_volume = int(base_volume * (1 + movement_intensity * 8))
            
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
                
                if momentum_alignment < 0.05: momentum_pressure = "STRONG"
                elif momentum_alignment < 0.1: momentum_pressure = "MODERATE"
                else: momentum_pressure = "WEAK"
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
            return {'volume_surge_ratio': 0.0, 'efficiency_ratio': 0.0, 'range_expansion': 0.0, 'momentum_pressure': "NEUTRAL"}

    def is_institutional_signal(self, metrics, thresholds):
        try:
            return (metrics['volume_surge_ratio'] >= thresholds['volume_surge'] and
                    metrics['efficiency_ratio'] >= thresholds['efficiency_ratio'] and
                    metrics['range_expansion'] >= thresholds['range_expansion'] and
                    metrics['momentum_pressure'] == "STRONG")
        except: return False

# --------- SIGNAL MANAGEMENT ---------
def can_send_signal(index, timeframe):
    global active_signals
    current_time = time.time()
    signal_key = f"{index}_{timeframe}"
    
    if signal_key in active_signals:
        signal_data = active_signals[signal_key]
        cooldown_period = signal_cooldown[timeframe]
        if current_time - signal_data['timestamp'] < cooldown_period: return False
    return True

def update_signal_tracking(index, timeframe, signal_id):
    global active_signals
    signal_key = f"{index}_{timeframe}"
    active_signals[signal_key] = {'signal_id': signal_id, 'timestamp': time.time(), 'timeframe': timeframe}

def clear_completed_signal(signal_id):
    global active_signals
    active_signals = {k: v for k, v in active_signals.items() if v['signal_id'] != signal_id}

# --------- PROFESSIONAL TELEGRAM FORMATTING ---------
def send_institutional_signal(index, signal_data, df):
    global signal_counter
    
    direction = signal_data['direction']
    timeframe = signal_data['timeframe']
    metrics = signal_data['metrics']
    index_price = signal_data['index_price']
    
    # Calculate strike and symbol
    strike = round_strike(index, index_price)
    if strike is None: return
    
    symbol = get_option_symbol(index, EXPIRIES[index], strike, direction)
    if symbol is None: return
    
    # Fetch option price
    option_price = fetch_option_price(symbol)
    if not option_price: return
    
    entry = round(option_price)
    
    # Calculate targets
    points_moved = signal_data['points_moved']
    base_move = max(points_moved * 0.8, 40)
    if direction == "CE":
        targets = [entry + int(base_move * 1.0), entry + int(base_move * 1.8), 
                   entry + int(base_move * 2.8), entry + int(base_move * 4.0)]
        sl = entry - int(base_move * 0.8)
    else:
        targets = [entry + int(base_move * 1.0), entry + int(base_move * 1.8),
                   entry + int(base_move * 2.8), entry + int(base_move * 4.0)]
        sl = entry - int(base_move * 0.8)
    
    entry = int(entry)
    targets = [int(t) for t in targets]
    sl = int(sl)
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    update_signal_tracking(index, timeframe, signal_id)
    
    # 🏛️ PROFESSIONAL INSTITUTIONAL ANALYSIS MESSAGE
    current_date = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime('%d %b %Y')
    
    institutional_msg = (
        f"🏛️{'🔴' if signal_data['tg_direction'] == 'RED' else '🟢'} **INSTITUTIONAL PRESSURE DETECTED - {index} {timeframe}** {'🏛️🔴' if signal_data['tg_direction'] == 'RED' else '🏛️🟢'}\n\n"
        
        f"📅 **DATE**: {current_date}\n"
        f"⏰ **TIME**: {signal_data['timestamp']} IST\n"
        f"🎯 **DIRECTION**: {signal_data['tg_direction']}\n"
        f"📈 **POINTS MOVED**: {signal_data['points_moved']} points\n"
        f"📊 **CANDLE RANGE**: {signal_data['candle_range']} points\n"
        f"📦 **VOLUME**: {signal_data['volume']}\n\n"
        
        f"📋 **PREVIOUS 3 CANDLES ANALYSIS**:\n"
    )
    
    # Add previous candles analysis
    for i, candle in enumerate(signal_data['previous_candles'], 1):
        institutional_msg += (
            f"    {i}. {candle['timestamp']} - {candle['direction']} {candle['points_moved']} points\n"
            f"       O: {candle['open']} | H: {candle['high']} | L: {candle['low']} | C: {candle['close']}\n"
            f"       Range: {candle['range']} pts | Volume: {candle['volume']}\n"
        )
    
    # Institutional metrics
    volume_assessment = "HIGH" if metrics['volume_surge_ratio'] >= 1.5 else "LOW"
    efficiency_assessment = "STRONG" if metrics['efficiency_ratio'] >= 1.3 else "MIXED"
    range_assessment = "INSTITUTIONAL_VOLUME" if metrics['range_expansion'] >= 15 else "RETAIL_VOLATILITY"
    
    directional_pressure = "INSTITUTIONAL_SELLING" if signal_data['tg_direction'] == 'RED' else "INSTITUTIONAL_BUYING"
    pressure_strength = "HIGH" if signal_data['institutional_score'] >= 70 else "MEDIUM" if signal_data['institutional_score'] >= 50 else "LOW"
    
    institutional_msg += (
        f"\n🏛️ **TRUE INSTITUTIONAL METRICS**:\n"
        f"• Volume Surge: {metrics['volume_surge_ratio']}x ({volume_assessment})\n"
        f"• Price Efficiency: {metrics['efficiency_ratio']}x ({efficiency_assessment})\n"
        f"• Momentum Alignment: {metrics['momentum_pressure']}\n"
        f"• Range Expansion: {metrics['range_expansion']}% ({range_assessment})\n\n"
        
        f"💼 **INSTITUTIONAL ASSESSMENT**:\n"
        f"• Institutional Score: {signal_data['institutional_score']}/100\n"
        f"• Pressure Type: {signal_data['pressure_type']}\n"
        f"• Confidence: {signal_data['confidence']}\n"
        f"• Directional Pressure: {directional_pressure}\n"
        f"• Pressure Strength: {pressure_strength}\n\n"
        
        f"🎯 **TRADING IMPLICATION**:\n"
        f"{directional_pressure} | {signal_data['confidence']} confidence\n"
        f"True Activity: {signal_data['pressure_type']}\n"
        f"📊 Market Status: INSTITUTIONAL_FLOW_DETECTED\n"
    )
    
    # Send institutional analysis first
    analysis_msg_id = send_telegram(institutional_msg)
    
    # Then send trading signal
    trade_msg = (
        f"\n{'🟢' if direction == 'CE' else '🔴'} {index} {strike} {direction}\n"
        f"SYMBOL: {symbol}\n"
        f"ABOVE {entry}\n"
        f"TARGETS: {targets_str}\n"
        f"SL: {sl}\n"
        f"FAKEOUT: NO\n"
        f"STRATEGY: INSTITUTIONAL PRESSURE {timeframe}\n"
        f"SIGNAL ID: {signal_id}\n"
        f"─────────────────────────────"
    )
    
    thread_id = send_telegram(trade_msg, reply_to=analysis_msg_id)
    
    # Store signal record
    signal_record = {
        "signal_id": signal_id, "timestamp": signal_data['timestamp'], "index": index,
        "strike": strike, "option_type": direction, "entry_price": entry,
        "targets": targets, "sl": sl, "timeframe": timeframe, "metrics": metrics
    }
    
    all_generated_signals.append(signal_record)
    
    # Start monitoring
    monitor_signal(symbol, entry, targets, sl, timeframe, thread_id, signal_id, signal_record)

# --------- MONITORING ---------
def monitor_signal(symbol, entry, targets, sl, timeframe, thread_id, signal_id, signal_record):
    def monitoring_thread():
        global daily_signals
        monitoring_duration = signal_cooldown[timeframe]
        start_time = time.time()
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        
        while time.time() - start_time < monitoring_duration:
            if should_stop_trading(): break
                
            price = fetch_option_price(symbol)
            if price:
                price = int(price)
                if price > max_price_reached:
                    max_price_reached = price
                    if price > entry:
                        send_telegram(f"📈 {symbol} moving up: {price}", reply_to=thread_id)
                
                for i, target in enumerate(targets):
                    if price >= target and not targets_hit[i]:
                        send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                        targets_hit[i] = True
                        signal_record['targets_hit'] = sum(targets_hit)
                
                if price <= sl:
                    send_telegram(f"🛑 {symbol}: SL hit at ₹{sl}", reply_to=thread_id)
                    break
            
            time.sleep(10)
        
        send_telegram(f"⏰ {symbol}: Monitoring period completed", reply_to=thread_id)
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

# --------- SIGNAL GENERATION ---------
def analyze_index_signal(index):
    analyzer = InstitutionalPressureAnalyzer()
    
    # Check 1min data
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

def trade_thread(index):
    result = analyze_index_signal(index)
    if result:
        signal_data, df = result
        send_institutional_signal(index, signal_data, df)

def run_algo_parallel():
    if not is_market_open() or should_stop_trading(): 
        return
        
    threads = []
    for index in ["NIFTY", "BANKNIFTY", "SENSEX"]:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: t.join()

# --------- MAIN LOOP ---------
while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        
        if not is_market_open():
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 **INSTITUTIONAL PRESSURE ALGO STARTED**\n"
                         "🏛️ Professional Institutional Analysis\n"
                         "📊 Multi-timeframe Pressure Detection\n"
                         "🎯 Smart Option Strike Selection")
            STARTED_SENT = True
        
        if should_stop_trading():
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Algo error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
