#INDEXBASED + EOD NOT COMMING - FIXED VERSION
# ENHANCED WITH INSTITUTIONAL MULTI-TIMEFRAME ANALYSIS

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

# ---------------- INSTITUTIONAL CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False
EXPIRY_RELAX_FACTOR = 0.7
GAMMA_VOL_SPIKE_THRESHOLD = 2.0
DELTA_OI_RATIO = 2.0
MOMENTUM_VOL_AMPLIFIER = 1.5

# INSTITUTIONAL MULTI-TIMEFRAME CONFIG
INSTITUTIONAL_TIMEFRAMES = ["1d", "1h", "15m", "5m"]  # Daily, Hourly, 15-min, 5-min
INSTITUTIONAL_LOOKBACK_DAYS = 60  # 60 days for institutional analysis
LIQUIDITY_THRESHOLD_MULTIPLIER = 0.5  # 50% of strike interval

# INSTITUTIONAL BLAST DETECTION
BLAST_MOMENTUM_THRESHOLD = 0.006
INSTITUTIONAL_SWEEP_DISTANCE = 0.003

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

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- KEEP ONLY YOUR 10 STRATEGIES + BLAST ---------
STRATEGY_NAMES = {
    "institutional_price_action": "INSTITUTIONAL PRICE ACTION",
    "opening_play": "OPENING PLAY", 
    "gamma_squeeze": "GAMMA SQUEEZE",
    "liquidity_sweeps": "LIQUIDITY SWEEP",
    "volume_gap_imbalance": "VOLUME GAP IMBALANCE",
    "ote_retracement": "OTE RETRACEMENT",
    "demand_supply_zones": "DEMAND SUPPLY ZONES",
    "pullback_reversal": "PULLBACK REVERSAL",
    "orderflow_mimic": "ORDERFLOW MIMIC",
    "bottom_fishing": "BOTTOM FISHING",
    "liquidity_zone": "LIQUIDITY ZONE",
    "institutional_blast": "INSTITUTIONAL BLAST"
}

# --------- ENHANCED TRACKING FOR REPORTS ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- NEW: SIGNAL DEDUPLICATION AND COOLDOWN TRACKING ---------
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1200

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {
        "INSTITUTIONAL PRICE ACTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "OPENING PLAY": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "GAMMA SQUEEZE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY SWEEP": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "VOLUME GAP IMBALANCE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "OTE RETRACEMENT": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "DEMAND SUPPLY ZONES": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "PULLBACK REVERSAL": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "ORDERFLOW MIMIC": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "BOTTOM FISHING": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY ZONE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "INSTITUTIONAL BLAST": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
    }

initialize_strategy_tracking()

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

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

# --------- STRIKE ROUNDING FOR KEPT INDICES ---------
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
        elif index == "MIDCPNIFTY": 
            return int(round(price / 25.0) * 25)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- ENHANCED: FETCH INSTITUTIONAL DATA WITH MULTI-TIMEFRAME ---------
def fetch_institutional_data(index, interval="5m", period="2d", extended_period=False):
    """Enhanced data fetching for institutional analysis"""
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    
    try:
        # For institutional analysis, fetch more data
        if extended_period:
            if interval == "1d":
                period = "6mo"  # 6 months for daily
            elif interval == "1h":
                period = "60d"  # 60 days for hourly
            elif interval == "15m":
                period = "30d"  # 30 days for 15-min
            elif interval == "5m":
                period = "7d"   # 7 days for 5-min
        
        df = yf.download(symbol_map[index], period=period, interval=interval, 
                        auto_adjust=True, progress=False, threads=True)
        
        if df.empty:
            return None
            
        # Calculate institutional indicators
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20'].replace(0, 1)
        
        return df
    except Exception as e:
        print(f"Error fetching {interval} data for {index}: {e}")
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

# 🚨 FIXED: STRICT EXPIRY VALIDATION FUNCTIONS 🚨
def validate_option_symbol(index, symbol, strike, opttype):
    try:
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")
            month_code = expected_dt.strftime("%b").upper()
            day = expected_dt.strftime("%d")
            expected_pattern = f"SENSEX{day}{month_code}{year_short}"
            symbol_upper = symbol.upper()
            if expected_pattern in symbol_upper:
                return True
            else:
                return False
        else:
            expected_pattern = expected_dt.strftime("%d%b%y").upper()
            symbol_upper = symbol.upper()
            if expected_pattern in symbol_upper:
                return True
            else:
                return False
    except Exception as e:
        return False

# 🚨 FIXED: GET OPTION SYMBOL WITH STRICT EXPIRY VALIDATION 🚨
def get_option_symbol(index, expiry_str, strike, opttype):
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")
            month_code = dt.strftime("%b").upper()
            day = dt.strftime("%d")
            symbol = f"SENSEX{day}{month_code}{year_short}{strike}{opttype}"
        elif index == "MIDCPNIFTY":
            symbol = f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        if validate_option_symbol(index, symbol, strike, opttype):
            return symbol
        else:
            return None
    except Exception as e:
        return None

# --------- ENHANCED: INSTITUTIONAL LIQUIDITY DETECTION ---------
def detect_institutional_liquidity_zones(index, df_5min):
    """
    INSTITUTIONAL MULTI-TIMEFRAME LIQUIDITY DETECTION
    Analyzes Daily, Hourly, 15-min data to find real liquidity zones
    """
    all_liquidity_levels = []
    
    # 1. DAILY TIMEFRAME LIQUIDITY (6 months data)
    df_daily = fetch_institutional_data(index, "1d", extended_period=True)
    if df_daily is not None and len(df_daily) > 20:
        # Daily swing highs/lows
        daily_highs = df_daily['High'].rolling(5).max().dropna()
        daily_lows = df_daily['Low'].rolling(5).min().dropna()
        
        # Recent daily highs/lows (last 20 days)
        recent_daily_high = daily_highs.iloc[-1] if len(daily_highs) > 0 else None
        recent_daily_low = daily_lows.iloc[-1] if len(daily_lows) > 0 else None
        
        if recent_daily_high and not math.isnan(recent_daily_high):
            all_liquidity_levels.append(float(recent_daily_high))
        if recent_daily_low and not math.isnan(recent_daily_low):
            all_liquidity_levels.append(float(recent_daily_low))
    
    # 2. HOURLY TIMEFRAME LIQUIDITY (60 days data)
    df_hourly = fetch_institutional_data(index, "1h", extended_period=True)
    if df_hourly is not None and len(df_hourly) > 48:  # At least 2 days of hourly data
        # Hourly swing points
        hourly_highs = df_hourly['High'].rolling(10).max().dropna()
        hourly_lows = df_hourly['Low'].rolling(10).min().dropna()
        
        # Recent hourly highs/lows
        recent_hourly_high = hourly_highs.iloc[-1] if len(hourly_highs) > 0 else None
        recent_hourly_low = hourly_lows.iloc[-1] if len(hourly_lows) > 0 else None
        
        if recent_hourly_high and not math.isnan(recent_hourly_high):
            all_liquidity_levels.append(float(recent_hourly_high))
        if recent_hourly_low and not math.isnan(recent_hourly_low):
            all_liquidity_levels.append(float(recent_hourly_low))
    
    # 3. 15-MIN TIMEFRAME LIQUIDITY (30 days data)
    df_15min = fetch_institutional_data(index, "15m", extended_period=True)
    if df_15min is not None and len(df_15min) > 96:  # At least 24 hours of 15-min data
        # 15-min swing points
        min15_highs = df_15min['High'].rolling(20).max().dropna()
        min15_lows = df_15min['Low'].rolling(20).min().dropna()
        
        # Recent 15-min highs/lows
        recent_min15_high = min15_highs.iloc[-1] if len(min15_highs) > 0 else None
        recent_min15_low = min15_lows.iloc[-1] if len(min15_lows) > 0 else None
        
        if recent_min15_high and not math.isnan(recent_min15_high):
            all_liquidity_levels.append(float(recent_min15_high))
        if recent_min15_low and not math.isnan(recent_min15_low):
            all_liquidity_levels.append(float(recent_min15_low))
    
    # 4. CURRENT 5-MIN DATA (from parameter)
    if df_5min is not None and len(df_5min) > 20:
        # 5-min swing points
        min5_highs = df_5min['High'].rolling(10).max().dropna()
        min5_lows = df_5min['Low'].rolling(10).min().dropna()
        
        # Recent 5-min highs/lows
        recent_min5_high = min5_highs.iloc[-1] if len(min5_highs) > 0 else None
        recent_min5_low = min5_lows.iloc[-1] if len(min5_lows) > 0 else None
        
        if recent_min5_high and not math.isnan(recent_min5_high):
            all_liquidity_levels.append(float(recent_min5_high))
        if recent_min5_low and not math.isnan(recent_min5_low):
            all_liquidity_levels.append(float(recent_min5_low))
    
    # 5. VWAP LEVELS (from all timeframes)
    for timeframe, df in [("1d", df_daily), ("1h", df_hourly), ("15m", df_15min), ("5m", df_5min)]:
        if df is not None and 'VWAP' in df.columns and len(df) > 0:
            vwap = df['VWAP'].iloc[-1]
            if not math.isnan(vwap):
                all_liquidity_levels.append(float(vwap))
    
    # Remove duplicates and sort
    unique_levels = sorted(list(set([round(l, 2) for l in all_liquidity_levels if l is not None and not math.isnan(l)])))
    
    # Separate into bullish (support) and bearish (resistance) zones
    current_price = float(df_5min['Close'].iloc[-1]) if df_5min is not None and len(df_5min) > 0 else None
    
    if current_price:
        bullish_zones = [l for l in unique_levels if l < current_price]
        bearish_zones = [l for l in unique_levels if l > current_price]
    else:
        bullish_zones = unique_levels[:len(unique_levels)//2]
        bearish_zones = unique_levels[len(unique_levels)//2:]
    
    return bullish_zones, bearish_zones

# --------- ENHANCED: INSTITUTIONAL LIQUIDITY HUNT ---------
def institutional_liquidity_hunt(index, df):
    """
    ENHANCED with multi-timeframe analysis
    """
    # Get institutional liquidity zones
    bullish_zones, bearish_zones = detect_institutional_liquidity_zones(index, df)
    
    # Current price
    last_close_val = None
    try:
        lc = ensure_series(df['Close']).iloc[-1]
        if isinstance(lc, float) and math.isnan(lc):
            last_close_val = None
        else:
            last_close_val = float(lc)
    except Exception:
        last_close_val = None
    
    # Add OI-based strikes (your original logic)
    if last_close_val is not None:
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)
    else:
        highest_ce_oi_strike = None
        highest_pe_oi_strike = None
    
    # Combine all liquidity sources
    bull_liquidity = []
    if bullish_zones:
        # Take nearest 3 bullish zones
        nearest_bullish = sorted(bullish_zones, reverse=True)[:3]
        bull_liquidity.extend(nearest_bullish)
    if highest_pe_oi_strike is not None:
        bull_liquidity.append(highest_pe_oi_strike)
    
    bear_liquidity = []
    if bearish_zones:
        # Take nearest 3 bearish zones
        nearest_bearish = sorted(bearish_zones)[:3]
        bear_liquidity.extend(nearest_bearish)
    if highest_ce_oi_strike is not None:
        bear_liquidity.append(highest_ce_oi_strike)
    
    # Remove duplicates and None values
    bull_liquidity = sorted(list(set([b for b in bull_liquidity if b is not None])))
    bear_liquidity = sorted(list(set([b for b in bear_liquidity if b is not None])))
    
    return bull_liquidity, bear_liquidity

# --------- ENHANCED: LIQUIDITY ZONE ENTRY CHECK ---------
def liquidity_zone_entry_check(price, bull_liq, bear_liq, index):
    """
    ENHANCED with dynamic threshold based on strike interval
    """
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None
    
    # Dynamic threshold based on index
    if index == "NIFTY":
        threshold = 25  # Half of 50 strike interval
    elif index == "BANKNIFTY":
        threshold = 50  # Half of 100 strike interval
    elif index == "SENSEX":
        threshold = 50  # Half of 100 strike interval
    elif index == "MIDCPNIFTY":
        threshold = 12  # Half of 25 strike interval
    else:
        threshold = 25
    
    # Check bullish zones for CE signal
    for zone in bull_liq:
        if zone is None: 
            continue
        try:
            if abs(price - zone) <= threshold:
                return "CE"
        except:
            continue
    
    # Check bearish zones for PE signal
    for zone in bear_liq:
        if zone is None: 
            continue
        try:
            if abs(price - zone) <= threshold:
                return "PE"
        except:
            continue
    
    # Check if price outside all zones
    valid_bear = [z for z in bear_liq if z is not None]
    valid_bull = [z for z in bull_liq if z is not None]
    
    if valid_bear and valid_bull:
        try:
            if price > max(valid_bear) or price < min(valid_bull):
                return "BOTH"
        except:
            return None
    
    return None

# --------- ENHANCED: INSTITUTIONAL MOMENTUM CONFIRMATION ---------
def institutional_momentum_confirmation(index, df, proposed_signal):
    """
    ENHANCED: Check entry at PREVIOUS candle, not current
    """
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_price = ensure_series(df['Open'])
        
        if len(close) < 5:
            return False
        
        # ENTRY AT PREVIOUS CANDLE (Your fix)
        # Check if previous candle showed initiation
        if proposed_signal == "CE":
            # Previous candle should be bullish initiation
            prev_close = close.iloc[-2]
            prev_open = open_price.iloc[-2]
            prev_high = high.iloc[-2]
            prev_low = low.iloc[-2]
            
            # Bullish candle with volume
            if (prev_close > prev_open and  # Green candle
                prev_close > (prev_high + prev_low) / 2 and  # Closed in upper half
                volume.iloc[-2] > volume.iloc[-10:-2].mean() * 1.2):  # Volume surge
                return True
                
        elif proposed_signal == "PE":
            # Previous candle should be bearish initiation
            prev_close = close.iloc[-2]
            prev_open = open_price.iloc[-2]
            prev_high = high.iloc[-2]
            prev_low = low.iloc[-2]
            
            # Bearish candle with volume
            if (prev_close < prev_open and  # Red candle
                prev_close < (prev_high + prev_low) / 2 and  # Closed in lower half
                volume.iloc[-2] > volume.iloc[-10:-2].mean() * 1.2):  # Volume surge
                return True
        
        # Fallback to original logic if above fails
        if proposed_signal == "CE":
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        elif proposed_signal == "PE":
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        return True
        
    except Exception:
        return False

# 🚨 KEEPING ALL YOUR ORIGINAL STRATEGIES EXACTLY AS THEY WERE 🚨
# (Only enhancing the liquidity detection part)

# 🚨 NEW: INSTITUTIONAL BLAST DETECTOR 🚨
def detect_institutional_blast(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low']) 
        close = ensure_series(df['Close'])
        open_price = ensure_series(df['Open'])
        
        if len(close) < 10: return None
        
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        prev_body = abs(close.iloc[-2] - open_price.iloc[-2])
        
        if (current_body > prev_body * 2.0 and
            close.iloc[-1] > high.iloc[-2] and
            (close.iloc[-1] - open_price.iloc[-1]) > 0):
            return "CE"
            
        if (current_body > prev_body * 2.0 and
            close.iloc[-1] < low.iloc[-2] and
            (close.iloc[-1] - open_price.iloc[-1]) < 0):
            return "PE"
            
        recent_high = high.iloc[-5:-1].max()
        recent_low = low.iloc[-5:-1].min()
        
        if (high.iloc[-1] > recent_high * (1 + INSTITUTIONAL_SWEEP_DISTANCE) and
            close.iloc[-1] < recent_high * 0.998):
            return "PE"
            
        if (low.iloc[-1] < recent_low * (1 - INSTITUTIONAL_SWEEP_DISTANCE) and
            close.iloc[-1] > recent_low * 1.002):
            return "CE"
            
    except Exception as e:
        return None
    return None

# 🚨 NEW: INSTITUTIONAL PRICE ACTION LAYER 🚨
def institutional_price_action_signal(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        current_close = close.iloc[-1]
        
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if (current_close > recent_high and 
            current_vol > vol_avg * 1.8 and
            current_close > close.iloc[-2] and
            close.iloc[-2] > close.iloc[-3]):
            return "CE"
            
        if (current_close < recent_low and
            current_vol > vol_avg * 1.8 and
            current_close < close.iloc[-2] and
            close.iloc[-2] < close.iloc[-3]):
            return "PE"
            
        current_body = abs(close.iloc[-1] - close.iloc[-2])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], close.iloc[-2])
        lower_wick = min(close.iloc[-1], close.iloc[-2]) - low.iloc[-1]
        
        if (upper_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
            
        if (lower_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] > close.iloc[-2]):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 LAYER 1: OPENING-RANGE INSTITUTIONAL PLAY 🚨
def institutional_opening_play(index, df):
    try:
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
        
    volume = ensure_series(df['Volume'])
    vol_avg = volume.rolling(10).mean().iloc[-1] if len(volume) >= 10 else volume.mean()
    vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)
    
    if current_price > prev_high + 15 and vol_ratio > 1.3: return "CE"
    if current_price < prev_low - 15 and vol_ratio > 1.3: return "PE"
    if current_price > prev_close + 25 and vol_ratio > 1.2: return "CE"
    if current_price < prev_close - 25 and vol_ratio > 1.2: return "PE"
    return None

# 🚨 LAYER 2: GAMMA SQUEEZE / EXPIRY LAYER 🚨
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
        
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg>0 else 1)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        
        try:
            url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            df_s = pd.DataFrame(requests.get(url,timeout=10).json())
            df_s['symbol'] = df_s['symbol'].str.upper()
            df_index = df_s[df_s['symbol'].str.contains(index)]
            df_index['oi'] = pd.to_numeric(df_index.get('oi',0), errors='coerce').fillna(0)
            ce_oi = df_index[df_index['symbol'].str.endswith("CE")]['oi'].sum()
            pe_oi = df_index[df_index['symbol'].str.endswith("PE")]['oi'].sum()
        except Exception:
            ce_oi = pe_oi = 0
        
        if vol_ratio > GAMMA_VOL_SPIKE_THRESHOLD and abs(speed) > 0.003:
            if speed > 0:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (ce_oi / (pe_oi+1e-6)) * 0.1)
                return {'side':'CE','confidence':conf}
            else:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (pe_oi / (ce_oi+1e-6)) * 0.1)
                return {'side':'PE','confidence':conf}
    except Exception:
        return None
    return None

# 🚨 NEW: EXPIRY DAY GAMMA BLAST (AFTER 1 PM) 🚨
def expiry_day_gamma_blast(index, df):
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time = ist_now.time()
        
        if not is_expiry_day_for_index(index) or current_time < dtime(13, 0):
            return None
            
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 10:
            return None
            
        current_vol = volume.iloc[-1]
        vol_avg_5 = volume.rolling(5).mean().iloc[-1]
        vol_avg_20 = volume.rolling(20).mean().iloc[-1]
        
        price_change_5min = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        price_change_15min = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        
        if (current_vol > vol_avg_20 * 3.0 and
            abs(price_change_5min) > 0.008 and
            abs(price_change_15min) > 0.015):
            
            if price_change_5min > 0 and price_change_15min > 0:
                return "CE"
            elif price_change_5min < 0 and price_change_15min < 0:
                return "PE"
                
    except Exception:
        return None
    return None

# 🚨 LAYER 11: LIQUIDITY SWEEPS 🚨
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
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
            return "PE"
            
        if (current_low < liquidity_low * (1 - LIQUIDITY_SWEEP_DISTANCE) and
            current_close > liquidity_low * 1.002 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 14: VOLUME GAP IMBALANCE 🚨
def detect_volume_gap_imbalance(df):
    try:
        volume = ensure_series(df['Volume'])
        close = ensure_series(df['Close'])
        
        if len(volume) < 20:
            return None
            
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-20:].mean()
        price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        
        if (current_volume > avg_volume * VOLUME_GAP_IMBALANCE and
            abs(price_change) > 0.004):
            if price_change > 0:
                return "CE"
            else:
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 15: OTE (Optimal Trade Entry) 🚨
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
            
            if (abs(current_price - ote_level) / ote_level < 0.0015 and
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3]):
                return "CE"
                
            ote_level = swing_low + (swing_range * level)
            if (abs(current_price - ote_level) / ote_level < 0.0015 and
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3]):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 16: DEMAND AND SUPPLY ZONES 🚨
def detect_demand_supply_zones(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < DEMAND_SUPPLY_ZONE_LOOKBACK + 5:
            return None
            
        lookback = DEMAND_SUPPLY_ZONE_LOOKBACK
        
        demand_lows = low.rolling(3, center=True).min().dropna()
        significant_demand = demand_lows[demand_lows == demand_lows.rolling(5).min()]
        
        supply_highs = high.rolling(3, center=True).max().dropna()
        significant_supply = supply_highs[supply_highs == supply_highs.rolling(5).max()]
        
        current_price = close.iloc[-1]
        
        for zone in significant_demand.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.002 and
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3] and
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
                return "CE"
                
        for zone in significant_supply.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.002 and
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3] and
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 6: PULLBACK REVERSAL 🚨
def detect_pullback_reversal(df):
    try:
        close = ensure_series(df['Close'])
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 6:
            return None

        if (close.iloc[-6] > ema21.iloc[-6] and close.iloc[-3] <= ema21.iloc[-3] and 
            close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 55 and 
            close.iloc[-1] > close.iloc[-2]):
            return "CE"

        if (close.iloc[-6] < ema21.iloc[-6] and close.iloc[-3] >= ema21.iloc[-3] and 
            close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 45 and 
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 7: ORDERFLOW MIMIC LOGIC 🚨
def mimic_orderflow_logic(df):
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); 
        low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 4:
            return None

        body = (high - low).abs(); wick_top = (high - close).abs(); wick_bottom = (close - low).abs()
        body_last = body.iloc[-1] if body.iloc[-1] != 0 else 1.0
        wick_top_ratio = wick_top.iloc[-1] / body_last
        wick_bottom_ratio = wick_bottom.iloc[-1] / body_last
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg and vol_avg > 0 else 1)

        if (close.iloc[-1] > close.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-3] - 3 and 
            wick_top_ratio > 0.7 and vol_ratio > 1.5):
            return "PE"

        if (close.iloc[-1] < close.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-3] + 3 and 
            wick_bottom_ratio > 0.7 and vol_ratio > 1.5):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 17: BOTTOM-FISHING 🚨
def detect_bottom_fishing(index, df):
    try:
        close = ensure_series(df['Close'])
        low = ensure_series(df['Low'])
        high = ensure_series(df['High'])
        volume = ensure_series(df['Volume'])
        if len(close) < 6: 
            return None

        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        last_close = float(close.iloc[-1])

        wick = last_close - low.iloc[-1]
        body = abs(close.iloc[-1] - close.iloc[-2])
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)

        if wick > body * 2.0 and vol_ratio > 1.5:
            for zone in bull_liq:
                if zone and abs(last_close - zone) <= 3:
                    return "CE"

        bear_wick = high.iloc[-1] - last_close
        if bear_wick > body * 2.0 and vol_ratio > 1.5:
            for zone in bear_liq:
                if zone and abs(last_close - zone) <= 3:
                    return "PE"
    except:
        return None
    return None

# --------- ENHANCED: INSTITUTIONAL FLOW CHECKS ---------
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None

    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(20).mean().iloc[-1]) if len(vol5) >= 20 else float(vol5.mean())

    if vol_latest > vol_avg*2.0 and abs(last_close-prev_close)/prev_close>0.005:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg*1.5:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg*1.5:
        return "PE"
    
    # ENHANCED: Use institutional liquidity zones
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    try:
        if last_close >= max(bear_liq) if bear_liq else False:
            return "PE"
        elif last_close <= min(bull_liq) if bull_liq else False:
            return "CE"
    except:
        return None
    return None

# --------- OI + DELTA FLOW DETECTION ---------
def oi_delta_flow_signal(index):
    try:
        url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        df_index=df[df['symbol'].str.contains(index)]
        if 'oi' not in df_index.columns:
            return None
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['oi_change'] = df_index['oi'].diff().fillna(0)
        ce_sum = df_index[df_index['symbol'].str.endswith("CE")]['oi_change'].sum()
        pe_sum = df_index[df_index['symbol'].str.endswith("PE")]['oi_change'].sum()
        if ce_sum>pe_sum*DELTA_OI_RATIO: return "CE"
        if pe_sum>ce_sum*DELTA_OI_RATIO: return "PE"
        if ce_sum>0 and pe_sum>0: return "BOTH"
    except:
        return None

# --------- ENHANCED: INSTITUTIONAL CONFIRMATION LAYER ---------
def institutional_confirmation_layer(index, df5, base_signal):
    try:
        close = ensure_series(df5['Close'])
        last_close = float(close.iloc[-1])
        
        # ENHANCED: Use institutional liquidity zones
        bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
        
        if base_signal == 'CE':
            # For CE: Should not be at resistance
            if bear_liq and last_close >= max(bear_liq):
                return False
        if base_signal == 'PE':
            # For PE: Should not be at support
            if bull_liq and last_close <= min(bull_liq):
                return False

        return True
    except Exception:
        return False

def institutional_flow_confirm(index, base_signal, df5):
    flow = institutional_flow_signal(index, df5)
    oi_flow = oi_delta_flow_signal(index)

    if flow and flow != 'BOTH' and flow != base_signal:
        return False
    if oi_flow and oi_flow != 'BOTH' and oi_flow != base_signal:
        return False

    if not institutional_confirmation_layer(index, df5, base_signal):
        return False

    return True

# --------- ENHANCED: UPDATED STRATEGY CHECK WITH INSTITUTIONAL IMPROVEMENTS ---------
def analyze_index_signal(index):
    df5 = fetch_institutional_data(index, "5m", extended_period=True)  # ENHANCED: 7 days data
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    last_close = float(close5.iloc[-1])
    prev_close = float(close5.iloc[-2])

    # 🚨 NEW: INSTITUTIONAL BLAST DETECTION (HIGHEST PRIORITY)
    blast_signal = detect_institutional_blast(df5)
    if blast_signal:
        if institutional_momentum_confirmation(index, df5, blast_signal):
            return blast_signal, df5, False, "institutional_blast"

    # 🚨 NEW: INSTITUTIONAL PRICE ACTION (HIGH PRIORITY) 🚨
    institutional_pa_signal = institutional_price_action_signal(df5)
    if institutional_pa_signal:
        if institutional_momentum_confirmation(index, df5, institutional_pa_signal):
            return institutional_pa_signal, df5, False, "institutional_price_action"

    # 🚨 LAYER 0: OPENING-PLAY PRIORITY 🚨
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        opening_range_bias = OPENING_PLAY_ENABLED and (OPENING_START <= t <= OPENING_END)
        if opening_range_bias:
            op_sig = institutional_opening_play(index, df5)
            if op_sig:
                fakeout = False
                # ENHANCED: Use institutional liquidity zones
                bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
                try:
                    if op_sig == "CE" and bear_liq and last_close >= max(bear_liq): 
                        fakeout = True
                    if op_sig == "PE" and bull_liq and last_close <= min(bull_liq): 
                        fakeout = True
                except:
                    fakeout = False
                return op_sig, df5, fakeout, "opening_play"
    except Exception:
        pass

    # 🚨 LAYER 1: EXPIRY / GAMMA DETECTION 🚨
    try:
        gamma = detect_gamma_squeeze(index, df5)
        if gamma:
            if is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE and not EXPIRY_INFO_ONLY:
                cand = gamma['side']
                oi_flow = oi_delta_flow_signal(index)
                if institutional_flow_confirm(index, cand, df5):
                    return cand, df5, False, "gamma_squeeze"
                if gamma['confidence'] > 0.6 and oi_flow == cand:
                    return cand, df5, False, "gamma_squeeze"
    except Exception:
        pass

    # 🚨 LAYER 2: LIQUIDITY SWEEPS (High Priority) 🚨
    sweep_sig = detect_liquidity_sweeps(df5)
    if sweep_sig:
        if institutional_momentum_confirmation(index, df5, sweep_sig):
            return sweep_sig, df5, True, "liquidity_sweeps"

    # 🚨 LAYER 14: VOLUME GAP IMBALANCE 🚨
    volume_sig = detect_volume_gap_imbalance(df5)
    if volume_sig:
        if institutional_momentum_confirmation(index, df5, volume_sig):
            return volume_sig, df5, False, "volume_gap_imbalance"

    # 🚨 LAYER 15: OTE RETRACEMENT 🚨
    ote_sig = detect_ote_retracement(df5)
    if ote_sig:
        if institutional_momentum_confirmation(index, df5, ote_sig):
            return ote_sig, df5, False, "ote_retracement"

    # 🚨 LAYER 16: DEMAND & SUPPLY ZONES 🚨
    ds_sig = detect_demand_supply_zones(df5)
    if ds_sig:
        if institutional_momentum_confirmation(index, df5, ds_sig):
            return ds_sig, df5, False, "demand_supply_zones"

    # 🚨 LAYER 14: PULLBACK REVERSAL 🚨
    pull_sig = detect_pullback_reversal(df5)
    if pull_sig:
        if institutional_momentum_confirmation(index, df5, pull_sig):
            return pull_sig, df5, False, "pullback_reversal"

    # 🚨 LAYER 15: ORDERFLOW MIMIC 🚨
    flow_sig = mimic_orderflow_logic(df5)
    if flow_sig:
        if institutional_momentum_confirmation(index, df5, flow_sig):
            return flow_sig, df5, False, "orderflow_mimic"

    # 🚨 LAYER 16: BOTTOM-FISHING 🚨
    bottom_sig = detect_bottom_fishing(index, df5)
    if bottom_sig:
        if institutional_momentum_confirmation(index, df5, bottom_sig):
            return bottom_sig, df5, False, "bottom_fishing"

    # ENHANCED: Final fallback: Institutional Liquidity-based entry
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq, index)  # Added index parameter
    if liquidity_side:
        if institutional_momentum_confirmation(index, df5, liquidity_side):
            return liquidity_side, df5, False, "liquidity_zone"

    return None

# --------- SIGNAL DEDUPLICATION AND COOLDOWN CHECK ---------
def can_send_signal(index, strike, option_type):
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    if strike_key in active_strikes:
        return False
        
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            return False
    
    return True

def update_signal_tracking(index, strike, option_type, signal_id):
    global active_strikes, last_signal_time
    
    strike_key = f"{index}_{strike}_{option_type}"
    active_strikes[strike_key] = {
        'signal_id': signal_id,
        'timestamp': time.time(),
        'targets_hit': 0
    }
    
    last_signal_time[index] = time.time()

def update_signal_progress(signal_id, targets_hit):
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

def clear_completed_signal(signal_id):
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}

# --------- FIXED: ENHANCED TRADE MONITORING AND TRACKING ---------
active_trades = {}

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        
        if not isinstance(targets_hit, (list, tuple)):
            targets_hit = list(targets_hit) if targets_hit is not None else [False]*len(targets)
        if len(targets_hit) < len(targets):
            targets_hit = list(targets_hit) + [False] * (len(targets) - len(targets_hit))
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        else:
            if max_price <= sl:
                diff = sl - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
            else:
                diff = max_price - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
    except Exception:
        return "0"

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        weakness_sent = False
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        last_activity_time = time.time()
        signal_id = signal_data.get('signal_id')
        
        while True:
            current_time = time.time()
            
            if not in_trade and (current_time - last_activity_time) > 1200:
                send_telegram(f"⏰ {symbol}: No activity for 20 minutes. Allowing new signals.", reply_to=thread_id)
                clear_completed_signal(signal_id)
                break
                
            if should_stop_trading():
                try:
                    final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                except Exception:
                    final_pnl = "0"
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "zero_targets": sum(targets_hit) == 0,
                    "no_new_highs": max_price_reached <= entry,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                clear_completed_signal(signal_id)
                break
                
            price = fetch_option_price(symbol)
            if price:
                last_activity_time = current_time
                price = round(price)
                
                if price > max_price_reached:
                    max_price_reached = price
                
                if not in_trade:
                    if price >= entry:
                        send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                        in_trade = True
                        entry_price_achieved = True
                        last_high = price
                        signal_data["entry_status"] = "ENTERED"
                else:
                    if price > last_high:
                        send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                        last_high = price
                    elif not weakness_sent and price < sl * 1.05:
                        send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                        weakness_sent = True
                    
                    current_targets_hit = sum(targets_hit)
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            current_targets_hit = sum(targets_hit)
                            update_signal_progress(signal_id, current_targets_hit)
                    
                    if price <= sl:
                        send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade. ALLOWING NEW SIGNAL.", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "zero_targets": sum(targets_hit) == 0,
                            "no_new_highs": max_price_reached <= entry,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
                        
                    if current_targets_hit >= 2:
                        update_signal_progress(signal_id, current_targets_hit)
                    
                    if all(targets_hit):
                        send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! Trade completed successfully!", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "zero_targets": False,
                            "no_new_highs": False,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# --------- FIXED: WORKING EOD REPORT SYSTEM ---------
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
    
    send_telegram(f"🕒 END OF DAY SIGNAL REPORT - { (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y') }\n"
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
        
        msg = (f"📊 SIGNAL #{i} - {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"─────────────────────────────\n"
               f"📅 Date: {(datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y')}\n"
               f"🕒 Time: {signal.get('timestamp','?')}\n"
               f"📈 Index: {signal.get('index','?')}\n"
               f"🎯 Strike: {signal.get('strike','?')}\n"
               f"🔰 Type: {signal.get('option_type','?')}\n"
               f"🏷️ Strategy: {signal.get('strategy','?')}\n\n"
               
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]} // {targets_for_disp[1]} // {targets_for_disp[2]} // {targets_for_disp[3]}\n"
               f"🛑 STOP LOSS: ₹{signal.get('sl','?')}\n\n"
               
               f"📊 PERFORMANCE:\n"
               f"• Entry Status: {signal.get('entry_status', 'PENDING')}\n"
               f"• Targets Hit: {signal.get('targets_hit', 0)}/4\n")
        
        if targets_hit_list:
            msg += f"• Targets Achieved: {', '.join(targets_hit_list)}\n"
        
        msg += (f"• Max Price Reached: ₹{signal.get('max_price_reached', signal.get('entry_price','?'))}\n"
                f"• Final P&L: {signal.get('final_pnl', '0')} points\n\n"
                
                f"⚡ Fakeout: {'YES' if signal.get('fakeout') else 'NO'}\n"
                f"📈 Index Price at Signal: {signal.get('index_price','?')}\n"
                f"🆔 Signal ID: {signal.get('signal_id','?')}\n"
                f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    total_pnl = 0.0
    successful_trades = 0
    for signal in unique_signals:
        pnl_str = signal.get("final_pnl", "0")
        try:
            if isinstance(pnl_str, str) and pnl_str.startswith("+"):
                total_pnl += float(pnl_str[1:])
                successful_trades += 1
            elif isinstance(pnl_str, str) and pnl_str.startswith("-"):
                total_pnl -= float(pnl_str[1:])
        except:
            pass
    
    summary_msg = (f"📈 DAY SUMMARY\n"
                   f"─────────────────────────────\n"
                   f"• Total Signals: {len(unique_signals)}\n"
                   f"• Successful Trades: {successful_trades}\n"
                   f"• Success Rate: {(successful_trades/len(unique_signals))*100:.1f}%\n"
                   f"• Total P&L: ₹{total_pnl:+.2f}\n"
                   f"─────────────────────────────")
    
    send_telegram(summary_msg)
    
    send_telegram("✅ END OF DAY REPORTS COMPLETED! See you tomorrow at 9:15 AM! 🚀")

# 🚨 ENHANCED: UPDATED SIGNAL SENDING WITH INSTITUTIONAL IMPROVEMENTS 🚨
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    
    # ENHANCED: Use dynamic strike selection based on institutional levels
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
    
    if side == "CE":
        # For CE: Find nearest support level below price
        if bull_liq:
            # Get the highest support level below current price
            supports_below = [s for s in bull_liq if s < signal_detection_price]
            if supports_below:
                strike = max(supports_below)
            else:
                strike = round_strike(index, signal_detection_price - 50)
        else:
            strike = round_strike(index, signal_detection_price)
    else:  # PE
        # For PE: Find nearest resistance level above price
        if bear_liq:
            # Get the lowest resistance level above current price
            resistances_above = [r for r in bear_liq if r > signal_detection_price]
            if resistances_above:
                strike = min(resistances_above)
            else:
                strike = round_strike(index, signal_detection_price + 50)
        else:
            strike = round_strike(index, signal_detection_price)
    
    if strike is None:
        return
        
    if not can_send_signal(index, strike, side):
        return
        
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    if symbol is None:
        return
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    entry = round(option_price)
    
    high = ensure_series(df["High"])
    low = ensure_series(df["Low"])
    close = ensure_series(df["Close"])
    
    # ENHANCED: Use institutional liquidity zones for target calculation
    if side == "CE":
        # Find next resistance level for target calculation
        if bear_liq:
            # Get the nearest resistance above
            resistances_above = [r for r in bear_liq if r > signal_detection_price]
            if resistances_above:
                nearest_resistance = min(resistances_above)
                price_gap = nearest_resistance - signal_detection_price
            else:
                # Use dynamic gap based on index
                if index == "NIFTY":
                    price_gap = 100
                elif index == "BANKNIFTY":
                    price_gap = 200
                elif index == "SENSEX":
                    price_gap = 300
                else:
                    price_gap = 80
        else:
            # Fallback
            if index == "NIFTY":
                price_gap = 100
            elif index == "BANKNIFTY":
                price_gap = 200
            else:
                price_gap = 150
        
        base_move = max(price_gap * 0.3, 40)
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.8)
        
    else:  # PE
        # Find next support level for target calculation
        if bull_liq:
            # Get the nearest support below
            supports_below = [s for s in bull_liq if s < signal_detection_price]
            if supports_below:
                nearest_support = max(supports_below)
                price_gap = signal_detection_price - nearest_support
            else:
                # Use dynamic gap based on index
                if index == "NIFTY":
                    price_gap = 100
                elif index == "BANKNIFTY":
                    price_gap = 200
                elif index == "SENSEX":
                    price_gap = 300
                else:
                    price_gap = 80
        else:
            # Fallback
            if index == "NIFTY":
                price_gap = 100
            elif index == "BANKNIFTY":
                price_gap = 200
            else:
                price_gap = 150
        
        base_move = max(price_gap * 0.3, 40)
        # FIXED: For PE, targets should be BELOW entry
        targets = [
            round(entry - base_move * 1.0),
            round(entry - base_move * 1.8),
            round(entry - base_move * 2.8),
            round(entry - base_move * 4.0)
        ]
        sl = round(entry + base_move * 0.8)  # SL above for PE
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
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
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "zero_targets": True,
        "no_new_highs": True,
        "final_pnl": "0"
    }
    
    update_signal_tracking(index, strike, side, signal_id)
    
    all_generated_signals.append(signal_data.copy())
    
    msg = (f"🟢 {index} {strike} {side}\n"
           f"SYMBOL: {symbol}\n"
           f"ABOVE {entry}\n"
           f"TARGETS: {targets_str}\n"
           f"SL: {sl}\n"
           f"FAKEOUT: {'YES' if fakeout else 'NO'}\n"
           f"STRATEGY: {strategy_name}\n"
           f"SIGNAL ID: {signal_id}")
         
    thread_id = send_telegram(msg)
    
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol, 
        "entry": entry, 
        "sl": sl, 
        "targets": targets, 
        "thread": thread_id, 
        "status": "OPEN",
        "index": index,
        "signal_data": signal_data
    }
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# --------- ENHANCED: UPDATED TRADE THREAD WITH INSTITUTIONAL IMPROVEMENTS ---------
def trade_thread(index):
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    if len(result) == 4:
        side, df, fakeout, strategy_key = result
    else:
        side, df, fakeout = result
        strategy_key = "unknown"
    
    df5 = fetch_institutional_data(index, "5m", extended_period=True)  # ENHANCED: 7 days data
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    oi_signal = oi_delta_flow_signal(index)
    final_signal = oi_signal or inst_signal or side

    if final_signal == "BOTH":
        for s in ["CE", "PE"]:
            if institutional_flow_confirm(index, s, df5):
                send_signal(index, s, df, fakeout, strategy_key)
        return
    elif final_signal:
        if df is None: 
            df = df5
        if institutional_flow_confirm(index, final_signal, df5):
            send_signal(index, final_signal, df, fakeout, strategy_key)
    else:
        return

# --------- ENHANCED: MAIN LOOP (KEPT INDICES ONLY) ---------
def run_algo_parallel():
    if not is_market_open(): 
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()
            EOD_REPORT_SENT = True
            send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
        return
        
    threads = []
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- FIXED: START WITH WORKING EOD SYSTEM ---------
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ EOD Report completed! Algorithm will resume tomorrow.")
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - 4 Indices Running\n"
                         "✅ Institutional Multi-Timeframe Analysis (Daily/Hourly/15-min/5-min)\n"
                         "✅ Enhanced Liquidity Zone Detection with 60-day lookback\n"
                         "✅ Entry at Previous Candle (Not Current)\n"
                         "✅ Dynamic Thresholds Based on Strike Intervals\n"
                         "✅ EOD Reports Working\n"
                         "✅ STRICT EXPIRY ENFORCEMENT")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Preparing EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING TRADES...")
                time.sleep(20)
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
