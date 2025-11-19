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

# ---------------- CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False
GAMMA_VOL_SPIKE_THRESHOLD = 2.0
DELTA_OI_RATIO = 2.0
MOMENTUM_VOL_AMPLIFIER = 1.5

# STRONGER CONFIRMATION THRESHOLDS
VCP_CONTRACTION_RATIO = 0.6
FAULTY_BASE_BREAK_THRESHOLD = 0.25
WYCKOFF_VOLUME_SPRING = 2.2
LIQUIDITY_SWEEP_DISTANCE = 0.005
PEAK_REJECTION_WICK_RATIO = 0.8
FVG_GAP_THRESHOLD = 0.0025
VOLUME_GAP_IMBALANCE = 2.5
OTE_RETRACEMENT_LEVELS = [0.618, 0.786]
DEMAND_SUPPLY_ZONE_LOOKBACK = 20

# --------- EXPIRIES FOR SELECTED INDICES ---------
EXPIRIES = {
    "NIFTY": "18 NOV 2025",
    "BANKNIFTY": "25 NOV 2025", 
    "SENSEX": "13 NOV 2025",
    "MIDCPNIFTY": "25 NOV 2025"
}

# --------- SELECTED STRATEGIES ---------
STRATEGY_NAMES = {
    "institutional_price_action": "INSTITUTIONAL PRICE ACTION",
    "opening_play": "OPENING PLAY", 
    "gamma_squeeze": "GAMMA SQUEEZE",
    "liquidity_sweeps": "LIQUIDITY SWEEP",
    "volume_gap_imbalance": "VOLUME GAP IMBALANCE",
    "ote_retracement": "OTE RETRACEMENT", 
    "demand_supply_zones": "DEMAND SUPPLY ZONES",
    "pullback_reversal": "PULLBACK REVERSAL",
    "liquidity_zone": "LIQUIDITY ZONE"
}

# --------- SIGNAL MANAGEMENT ---------
daily_signals = []
signal_counter = 0
all_generated_signals = []
signaled_strikes = {}
active_trades = {}

# 🚨 CRITICAL: SIGNAL QUEUE SYSTEM
signal_queue_active = False
current_signal_data = None
last_signal_time = 0
SIGNAL_TIMEOUT = 300  # 5 minutes
last_activity_time = 0

# 🚨 CRITICAL: Global stop flag
stop_all_monitoring = False

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

try:
    client = SmartConnect(api_key=API_KEY)
    session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
    feedToken = client.getfeedToken()
except Exception as e:
    print(f"Login failed: {e}")

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STARTED_SENT = False
STOP_SENT = False
EOD_REPORT_SENT = False
MARKET_CLOSED_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=10).json()
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print(f"Telegram error: {e}")
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

# --------- STRIKE ROUNDING WITH ONE-STEP AWAY ---------
def round_strike(index, price):
    try:
        if price is None:
            return None
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            nearest = int(round(price / 50.0) * 50)
            # One step away - if index at 26790, return 26850 instead of 26800
            if abs(nearest - price) <= 25:  # Very close to strike
                return nearest + 50 if nearest >= price else nearest - 50
            return nearest
            
        elif index == "BANKNIFTY": 
            nearest = int(round(price / 100.0) * 100)
            if abs(nearest - price) <= 50:
                return nearest + 100 if nearest >= price else nearest - 100
            return nearest
            
        elif index == "SENSEX": 
            nearest = int(round(price / 100.0) * 100)
            if abs(nearest - price) <= 50:
                return nearest + 100 if nearest >= price else nearest - 100
            return nearest
            
        elif index == "MIDCPNIFTY": 
            nearest = int(round(price / 25.0) * 25)
            if abs(nearest - price) <= 12.5:
                return nearest + 25 if nearest >= price else nearest - 25
            return nearest
            
    except Exception:
        return None
    return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA FOR SELECTED INDICES ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    try:
        df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
        return None if df.empty else df
    except:
        return None

# --------- LOAD TOKEN MAP ---------
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

# --------- SAFE LTP FETCH ---------
def fetch_option_price(symbol, retries=2, delay=2):
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

# --------- DETECT LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    high_series = ensure_series(df['High']).dropna()
    low_series = ensure_series(df['Low']).dropna()
    try:
        if len(high_series) <= lookback:
            high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
        else:
            high_pool = float(high_series.rolling(lookback).max().iloc[-2])
    except Exception:
        high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
    try:
        if len(low_series) <= lookback:
            low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
        else:
            low_pool = float(low_series.rolling(lookback).min().iloc[-2])
    except Exception:
        low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')

    if math.isnan(high_pool) and len(high_series)>0:
        high_pool = float(high_series.max())
    if math.isnan(low_pool) and len(low_series)>0:
        low_pool = float(low_series.min())

    return round(high_pool,0), round(low_pool,0)

# --------- INSTITUTIONAL LIQUIDITY HUNT ---------
def institutional_liquidity_hunt(index, df):
    prev_high = None
    prev_low = None
    try:
        prev_high_val = ensure_series(df['High']).iloc[-2]
        prev_low_val = ensure_series(df['Low']).iloc[-2]
        prev_high = float(prev_high_val) if not (isinstance(prev_high_val,float) and math.isnan(prev_high_val)) else None
        prev_low = float(prev_low_val) if not (isinstance(prev_low_val,float) and math.isnan(prev_low_val)) else None
    except Exception:
        prev_high = None
        prev_low = None

    high_zone, low_zone = detect_liquidity_zone(df, lookback=15)

    last_close_val = None
    try:
        lc = ensure_series(df['Close']).iloc[-1]
        if isinstance(lc, float) and math.isnan(lc):
            last_close_val = None
        else:
            last_close_val = float(lc)
    except Exception:
        last_close_val = None

    if last_close_val is None:
        highest_ce_oi_strike = None
        highest_pe_oi_strike = None
    else:
        highest_ce_oi_strike = round_strike(index, last_close_val + 100)
        highest_pe_oi_strike = round_strike(index, last_close_val - 100)

    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

    return bull_liquidity, bear_liquidity

def liquidity_zone_entry_check(price, bull_liq, bear_liq):
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None

    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 10:  # Slightly wider zone for better entries
                return "CE"
        except:
            continue
    for zone in bear_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 10:
                return "PE"
        except:
            continue

    return None

# 🚨 INSTITUTIONAL PRICE ACTION LAYER 🚨
def institutional_price_action_signal(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_high = high.iloc[-8:-1].max()
        recent_low = low.iloc[-8:-1].min()
        current_close = close.iloc[-1]
        
        vol_avg = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # Strong institutional breakout
        if (current_close > recent_high and 
            current_vol > vol_avg * 2.0 and
            current_close > close.iloc[-2] and
            close.iloc[-2] > close.iloc[-3]):
            return "CE"
            
        if (current_close < recent_low and
            current_vol > vol_avg * 2.0 and
            current_close < close.iloc[-2] and
            close.iloc[-2] < close.iloc[-3]):
            return "PE"
            
        current_body = abs(close.iloc[-1] - close.iloc[-2])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], close.iloc[-2])
        lower_wick = min(close.iloc[-1], close.iloc[-2]) - low.iloc[-1]
        
        # Strong rejection patterns
        if (upper_wick > current_body * 2.0 and
            current_vol > vol_avg * 1.8 and
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
            
        if (lower_wick > current_body * 2.0 and
            current_vol > vol_avg * 1.8 and
            close.iloc[-1] > close.iloc[-2]):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 INSTITUTIONAL MOMENTUM CONFIRMATION 🚨
def institutional_momentum_confirmation(index, df, proposed_signal):
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 5:
            return False
            
        if proposed_signal == "CE":
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
            if volume.iloc[-1] < volume.rolling(10).mean().iloc[-1] * 1.3:
                return False
                
        elif proposed_signal == "PE":
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
            if volume.iloc[-1] < volume.rolling(10).mean().iloc[-1] * 1.3:
                return False
                
        return True
        
    except Exception:
        return False

# 🚨 PRIORITY 1: OPENING-RANGE INSTITUTIONAL PLAY 🚨
def institutional_opening_play(index, df):
    try:
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
        
    volume = ensure_series(df['Volume'])
    vol_avg = volume.rolling(8).mean().iloc[-1] if len(volume) >= 8 else volume.mean()
    vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)
    
    # Strong opening breakouts
    if current_price > prev_high + 20 and vol_ratio > 1.5: return "CE"
    if current_price < prev_low - 20 and vol_ratio > 1.5: return "PE"
    if current_price > prev_close + 30 and vol_ratio > 1.4: return "CE"
    if current_price < prev_close - 30 and vol_ratio > 1.4: return "PE"
    return None

# 🚨 GAMMA SQUEEZE / EXPIRY LAYER 🚨
def is_expiry_day_for_index(index):
    try:
        ex = EXPIRIES.get(index)
        if not ex: return False
        dt = datetime.strptime(ex, "%d %b %Y")
        today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
        return dt.date() == today
    except Exception:
        return False

def detect_gamma_squeeze(index, df):
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume']); 
        high = ensure_series(df['High']); low = ensure_series(df['Low'])
        if len(close) < 6: return None
        
        vol_avg = volume.rolling(15).mean().iloc[-1] if len(volume)>=15 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg>0 else 1)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        
        if vol_ratio > GAMMA_VOL_SPIKE_THRESHOLD and abs(speed) > 0.005:
            if speed > 0:
                return "CE"
            else:
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LIQUIDITY SWEEPS 🚨
def detect_liquidity_sweeps(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_highs = high.iloc[-10:-2]
        recent_lows = low.iloc[-10:-2]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        if (current_high > liquidity_high * (1 + LIQUIDITY_SWEEP_DISTANCE) and
            current_close < liquidity_high * 0.998 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.8):
            return "PE"
            
        if (current_low < liquidity_low * (1 - LIQUIDITY_SWEEP_DISTANCE) and
            current_close > liquidity_low * 1.002 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.8):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 VOLUME GAP IMBALANCE 🚨
def detect_volume_gap_imbalance(df):
    try:
        volume = ensure_series(df['Volume'])
        close = ensure_series(df['Close'])
        
        if len(volume) < 15:
            return None
            
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-15:].mean()
        price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        
        if (current_volume > avg_volume * VOLUME_GAP_IMBALANCE and
            abs(price_change) > 0.006):  # Stronger price move required
            if price_change > 0:
                return "CE"
            else:
                return "PE"
    except Exception:
        return None
    return None

# 🚨 OTE (Optimal Trade Entry) 🚨
def detect_ote_retracement(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 15:
            return None
            
        swing_high = high.iloc[-15:-5].max()
        swing_low = low.iloc[-15:-5].min()
        swing_range = swing_high - swing_low
        
        current_price = close.iloc[-1]
        
        for level in OTE_RETRACEMENT_LEVELS:
            ote_level = swing_high - (swing_range * level)
            
            if (abs(current_price - ote_level) / ote_level < 0.002 and
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3]):
                return "CE"
                
            ote_level = swing_low + (swing_range * level)
            if (abs(current_price - ote_level) / ote_level < 0.002 and
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3]):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 DEMAND AND SUPPLY ZONES 🚨
def detect_demand_supply_zones(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < DEMAND_SUPPLY_ZONE_LOOKBACK + 5:
            return None
            
        lookback = DEMAND_SUPPLY_ZONE_LOOKBACK
        
        # Find significant demand zones (support)
        demand_zones = []
        for i in range(3, len(low)-2):
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
                demand_zones.append(low.iloc[i])
        
        # Find significant supply zones (resistance)  
        supply_zones = []
        for i in range(3, len(high)-2):
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
                supply_zones.append(high.iloc[i])
        
        current_price = close.iloc[-1]
        
        # Check recent demand zones
        for zone in demand_zones[-3:]:
            if (abs(current_price - zone) / zone < 0.003 and
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3] and
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.5):
                return "CE"
                
        # Check recent supply zones
        for zone in supply_zones[-3:]:
            if (abs(current_price - zone) / zone < 0.003 and
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3] and
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.5):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 PULLBACK REVERSAL 🚨
def detect_pullback_reversal(df):
    try:
        close = ensure_series(df['Close'])
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 6:
            return None

        # Bullish pullback: Price was above EMA21, pulled back to EMA21, now bouncing
        if (close.iloc[-6] > ema21.iloc[-6] and close.iloc[-3] <= ema21.iloc[-3] and 
            close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 55 and 
            close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
            return "CE"

        # Bearish pullback: Price was below EMA21, pulled back to EMA21, now rejecting
        if (close.iloc[-6] < ema21.iloc[-6] and close.iloc[-3] >= ema21.iloc[-3] and 
            close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 45 and 
            close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 SIGNAL QUEUE MANAGEMENT 🚨
def can_send_new_signal():
    global signal_queue_active, current_signal_data, last_activity_time
    
    # No signal active
    if not signal_queue_active:
        return True
        
    # Check if 5 minutes passed without activity
    if time.time() - last_activity_time > SIGNAL_TIMEOUT:
        signal_queue_active = False
        current_signal_data = None
        send_telegram("⏰ 5-minute timeout - Releasing signal queue")
        return True
        
    return False

def update_signal_activity():
    global last_activity_time
    last_activity_time = time.time()

def set_signal_active(signal_data):
    global signal_queue_active, current_signal_data, last_activity_time
    signal_queue_active = True
    current_signal_data = signal_data
    last_activity_time = time.time()

def set_signal_complete():
    global signal_queue_active, current_signal_data
    signal_queue_active = False
    current_signal_data = None

# 🚨 STRIKE COOLDOWN SYSTEM 🚨
def can_send_strike(index, strike, option_type):
    key = f"{index}_{strike}_{option_type}"
    if key in signaled_strikes:
        if time.time() - signaled_strikes[key] < 3600:  # 1 hour cooldown
            return False
    signaled_strikes[key] = time.time()
    return True

# 🚨 INSTITUTIONAL TARGETS 🚨
def get_institutional_targets(entry_price, direction):
    if direction == "CE":
        target1 = round(entry_price * 1.015)
        target2 = round(entry_price * 1.030)  
        target3 = round(entry_price * 1.050)
        target4 = round(entry_price * 1.080)
    else:
        target1 = round(entry_price * 0.985)
        target2 = round(entry_price * 0.970)
        target3 = round(entry_price * 0.950)
        target4 = round(entry_price * 0.920)
    
    return [target1, target2, target3, target4]

# --------- UPDATED STRATEGY CHECK WITH PRIORITY ---------
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    # Check signal queue
    if not can_send_new_signal():
        return None

    # 🚨 PRIORITY 1: OPENING PLAY
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        opening_range_bias = OPENING_PLAY_ENABLED and (OPENING_START <= t <= OPENING_END)
        if opening_range_bias:
            op_sig = institutional_opening_play(index, df5)
            if op_sig and institutional_momentum_confirmation(index, df5, op_sig):
                return op_sig, df5, False, "opening_play"
    except Exception:
        pass

    # 🚨 PRIORITY 2: INSTITUTIONAL PRICE ACTION
    institutional_pa_signal = institutional_price_action_signal(df5)
    if institutional_pa_signal:
        if institutional_momentum_confirmation(index, df5, institutional_pa_signal):
            return institutional_pa_signal, df5, False, "institutional_price_action"

    # 🚨 PRIORITY 3: GAMMA SQUEEZE
    try:
        gamma = detect_gamma_squeeze(index, df5)
        if gamma and is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE:
            return gamma, df5, False, "gamma_squeeze"
    except Exception:
        pass

    # 🚨 OTHER STRATEGIES
    strategies = [
        (detect_liquidity_sweeps, "liquidity_sweeps"),
        (detect_volume_gap_imbalance, "volume_gap_imbalance"),
        (detect_ote_retracement, "ote_retracement"),
        (detect_demand_supply_zones, "demand_supply_zones"),
        (detect_pullback_reversal, "pullback_reversal")
    ]
    
    for strategy_func, strategy_key in strategies:
        signal = strategy_func(df5)
        if signal and institutional_momentum_confirmation(index, df5, signal):
            return signal, df5, False, strategy_key

    # 🚨 FINAL: LIQUIDITY ZONE
    last_close = float(close5.iloc[-1])
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq)
    if liquidity_side:
        return liquidity_side, df5, False, "liquidity_zone"

    return None

# --------- SYMBOL FORMAT ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    dt=datetime.strptime(expiry_str,"%d %b %Y")
    
    if index == "SENSEX":
        year_short = dt.strftime("%y")
        month_code = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        return f"SENSEX{year_short}{month_code}{strike}{opttype}"
    elif index == "MIDCPNIFTY":
        return f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    else:
        return f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"

# --------- INSTITUTIONAL FLOW CHECKS ---------
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None

    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(15).mean().iloc[-1]) if len(vol5) >= 15 else float(vol5.mean())

    if vol_latest > vol_avg*2.5 and abs(last_close-prev_close)/prev_close>0.008:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg*2.0:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg*2.0:
        return "PE"
    
    return None

def institutional_flow_confirm(index, base_signal, df5):
    flow = institutional_flow_signal(index, df5)
    
    if flow and flow != 'BOTH' and flow != base_signal:
        return False

    try:
        close = ensure_series(df5['Close'])
        last_close = float(close.iloc[-1])
        
        high_zone, low_zone = detect_liquidity_zone(df5, lookback=20)
        if base_signal == 'CE' and last_close >= high_zone:
            return False
        if base_signal == 'PE' and last_close <= low_zone:
            return False

        return True
    except Exception:
        return False

# 🚨 UPDATED MONITORING WITH SIGNAL QUEUE 🚨
def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        first_target_hit = False
        
        while True:
            if should_stop_trading():
                break
                
            price = fetch_option_price(symbol)
            if not price: 
                time.sleep(10)
                continue
                
            price = round(price)
            
            if price > max_price_reached:
                max_price_reached = price
                update_signal_activity()  # Update activity on new highs
                
            if not entry_price_achieved:
                if price >= entry:
                    send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                    entry_price_achieved = True
                    last_high = price
                    signal_data["entry_status"] = "ENTERED"
                    update_signal_activity()
            else:
                if price > last_high:
                    last_high = price
                    update_signal_activity()
                
                # Check first target
                if not first_target_hit and price >= targets[0]:
                    send_telegram(f"🎯 {symbol}: 1st Target Hit at ₹{targets[0]}", reply_to=thread_id)
                    targets_hit[0] = True
                    first_target_hit = True
                    update_signal_activity()
                    
                    # 🚨 CRITICAL: Release signal queue after 1st target
                    time.sleep(2)
                    set_signal_complete()
                    send_telegram("🟢 Signal queue released - Next signal available")
                
                # Check other targets
                for i in range(1, len(targets)):
                    if price >= targets[i] and not targets_hit[i]:
                        send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{targets[i]}", reply_to=thread_id)
                        targets_hit[i] = True
                        update_signal_activity()
                
                # Check SL hit
                if price <= sl:
                    send_telegram(f"🔴 {symbol}: SL Hit at ₹{sl}", reply_to=thread_id)
                    update_signal_activity()
                    
                    # 🚨 CRITICAL: Release signal queue after SL hit
                    time.sleep(2)
                    set_signal_complete()
                    send_telegram("🟢 Signal queue released - Next signal available")
                    break
                    
                if all(targets_hit):
                    send_telegram(f"🏆 {symbol}: ALL TARGETS HIT!", reply_to=thread_id)
                    update_signal_activity()
                    break
            
            time.sleep(8)
        
        # Final P&L calculation
        try:
            if first_target_hit:
                final_pnl = f"+{targets[0] - entry:.2f}"
            elif price <= sl:
                final_pnl = f"-{entry - sl:.2f}"
            else:
                final_pnl = "0"
        except:
            final_pnl = "0"
            
        signal_data.update({
            "targets_hit": sum(targets_hit),
            "max_price_reached": max_price_reached,
            "first_target_hit": first_target_hit,
            "final_pnl": final_pnl
        })
        daily_signals.append(signal_data)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# 🚨 UPDATED SIGNAL SENDING WITH QUEUE MANAGEMENT 🚨
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    # Check signal queue
    if not can_send_new_signal():
        return
        
    current_df = fetch_index_data(index, "5m", "2d")
    if current_df is None:
        return
        
    signal_detection_price = float(ensure_series(current_df["Close"]).iloc[-1])
    
    # 🚨 ONE-STEP AWAY STRIKE
    strike = round_strike(index, signal_detection_price)
    if strike is None:
        return
        
    # 🚨 STRIKE COOLDOWN CHECK
    if not can_send_strike(index, strike, side):
        return
        
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    entry = round(option_price)
    
    # 🚨 INSTITUTIONAL TARGETS
    targets = get_institutional_targets(entry, side)
    sl = round(option_price * 0.85) if side == "CE" else round(option_price * 1.15)
    
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": side,
        "strategy": strategy_name,
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": fakeout,
        "index_price": signal_detection_price,
        "entry_status": "PENDING"
    }
    
    all_generated_signals.append(signal_data.copy())
    
    # 🚨 CLEAN ORGANIZED MESSAGE
    msg = (f"🟢 GIT🔊 {index} {strike} {side}\n"
           f"🏷️ {strategy_name}\n"
           f"🔹 Strike: {strike}\n"
           f"💰 Entry: ₹{entry}\n"
           f"🎯 Targets: {targets[0]} // {targets[1]} // {targets[2]} // {targets[3]}\n"
           f"🛑 SL: ₹{sl}\n"
           f"📊 Index: {signal_detection_price:.2f}\n"
           f"🆔 {signal_id}")
         
    thread_id = send_telegram(msg)
    
    if thread_id:
        # 🚨 SET SIGNAL ACTIVE IN QUEUE
        set_signal_active(signal_data)
        
        trade_id = f"{symbol}_{int(time.time())}"
        active_trades[trade_id] = {
            "symbol": symbol, 
            "entry": entry, 
            "sl": sl, 
            "targets": targets, 
            "thread": thread_id, 
            "status": "OPEN",
            "signal_data": signal_data
        }
        
        monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# 🚨 UPDATED TRADE THREAD 🚨
def trade_thread(index):
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    side, df, fakeout, strategy_key = result
    
    df5 = fetch_index_data(index, "5m", "2d")
    if institutional_flow_confirm(index, side, df5):
        send_signal(index, side, df, fakeout, strategy_key)

# --------- MAIN LOOP (SELECTED INDICES) ---------
def run_algo_parallel():
    global stop_all_monitoring
    
    if not is_market_open(): 
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed - Stopping monitoring...")
            STOP_SENT = True
            
        stop_all_monitoring = True
        time.sleep(30)
        
        if not EOD_REPORT_SENT:
            send_individual_signal_reports()
            EOD_REPORT_SENT = True
            
        return
        
    threads = []
    selected_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in selected_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join(timeout=25)

# 🚨 EOD REPORT SYSTEM 🚨
def send_individual_signal_reports():
    global daily_signals, all_generated_signals
    
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
        send_telegram("📊 END OF DAY REPORT\nNo signals generated today.")
        return
    
    send_telegram(f"📈 END OF DAY REPORT - { (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y') }\n"
                  f"📊 Total Signals: {len(unique_signals)}\n"
                  f"────────────────────")
    
    for i, signal in enumerate(unique_signals, 1):
        msg = (f"📊 SIGNAL #{i}\n"
               f"────────────────────\n"
               f"🕒 {signal.get('timestamp','?')} | {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"🏷️ {signal.get('strategy','?')}\n"
               f"💰 Entry: ₹{signal.get('entry_price','?')}\n"
               f"🎯 Targets Hit: {signal.get('targets_hit',0)}/4\n"
               f"📈 Max Price: ₹{signal.get('max_price_reached',signal.get('entry_price','?'))}\n"
               f"💵 P&L: {signal.get('final_pnl','0')}\n"
               f"🆔 {signal.get('signal_id','?')}\n"
               f"────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
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
    
    summary_msg = (f"📈 DAY SUMMARY\n"
                   f"────────────────────\n"
                   f"• Signals: {len(unique_signals)}\n"
                   f"• Successful: {successful_trades}\n"
                   f"• Success Rate: {(successful_trades/len(unique_signals))*100:.1f}%\n"
                   f"• Total P&L: ₹{total_pnl:+.2f}\n"
                   f"────────────────────")
    
    send_telegram(summary_msg)

# 🚨 MAIN LOOP WITH SIGNAL QUEUE 🚨
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False
STARTED_SENT = False
STOP_SENT = False

while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now

        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market Closed - Waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 GIT ULTIMATE ALGO STARTED\n"
                         "✅ 4 Indices | 8 Strategies\n"
                         "✅ Signal Queue Management\n"
                         "✅ Institutional Entries")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing - Preparing EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            if not EOD_REPORT_SENT:
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(20)
        
    except Exception as e:
        time.sleep(30)
