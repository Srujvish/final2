#INDEXBASED + EOD NOT COMMING - INSTITUTIONAL VERSION

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
OPENING_PLAY_ENABLED = False  # Disabled - too many fakeouts
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False
EXPIRY_RELAX_FACTOR = 0.7
GAMMA_VOL_SPIKE_THRESHOLD = 2.0
DELTA_OI_RATIO = 2.0
MOMENTUM_VOL_AMPLIFIER = 1.5

# 🚨 **INSTITUTIONAL GRADE THRESHOLDS** 🚨
BLAST_VOLUME_THRESHOLD = 5.0  # 5x volume spike for REAL institutional moves
BLAST_PRICE_MOVE_PCT = 0.015   # 1.5% minimum move in single candle
SWEEP_DISTANCE_PCT = 0.01      # 1.0% sweep through levels
INSTITUTIONAL_SPREAD_RATIO = 2.0  # Body to wick ratio

# INSTITUTIONAL CONFIRMATION THRESHOLDS
VCP_CONTRACTION_RATIO = 0.6
FAULTY_BASE_BREAK_THRESHOLD = 0.25
WYCKOFF_VOLUME_SPRING = 2.5  # Increased for institutional moves
LIQUIDITY_SWEEP_DISTANCE = 0.008  # Increased
PEAK_REJECTION_WICK_RATIO = 0.85
FVG_GAP_THRESHOLD = 0.003
VOLUME_GAP_IMBALANCE = 3.0  # Increased
OTE_RETRACEMENT_LEVELS = [0.618, 0.786]
DEMAND_SUPPLY_ZONE_LOOKBACK = 20

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- INSTITUTIONAL STRATEGIES ONLY ---------
STRATEGY_NAMES = {
    "institutional_stop_hunt": "INSTITUTIONAL STOP HUNT",
    "liquidity_absorption": "LIQUIDITY ABSORPTION",
    "false_breakout_trap": "FALSE BREAKOUT TRAP", 
    "liquidity_zone": "LIQUIDITY ZONE",
    "institutional_blast": "INSTITUTIONAL BLAST",
    "sweep_order_detection": "SWEEP ORDER DETECTION"
}

# --------- ENHANCED TRACKING FOR REPORTS ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- NEW: SIGNAL DEDUPLICATION AND COOLDOWN TRACKING ---------
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1800  # Increased to 30 minutes for institutional moves

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {
        "INSTITUTIONAL STOP HUNT": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY ABSORPTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "FALSE BREAKOUT TRAP": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY ZONE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "INSTITUTIONAL BLAST": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "SWEEP ORDER DETECTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
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

# --------- FETCH INDEX DATA FOR KEPT INDICES ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

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
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)

    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

    return bull_liquidity, bear_liquidity

# 🚨 **ENHANCED LIQUIDITY ZONE CHECK WITH INSTITUTIONAL FILTERS** 🚨
def liquidity_zone_entry_check(price, bull_liq, bear_liq):
    """
    ENHANCED with volume and direction check
    """
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None

    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:  # Within 5 points
                return "PE"  # Bull liquidity broken = PE entry
        except:
            continue
    for zone in bear_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:  # Within 5 points
                return "CE"  # Bear liquidity broken = CE entry
        except:
            continue

    valid_bear = [z for z in bear_liq if z is not None]
    valid_bull = [z for z in bull_liq if z is not None]
    if valid_bear and valid_bull:
        try:
            if price > max(valid_bear) or price < min(valid_bull):
                return "BOTH"
        except:
            return None
    return None

# 🚨 **NEW: INSTITUTIONAL STOP HUNT DETECTION** 🚨
def detect_stop_hunt(df):
    """
    Institutions HUNT STOPS before big moves.
    Pattern: Quick spike through level + quick rejection + reversal
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_price = ensure_series(df['Open'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 8:
            return None
        
        # Find recent liquidity levels (last 30 mins)
        recent_highs = high.iloc[-6:-1]
        recent_lows = low.iloc[-6:-1]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        # Volume analysis
        vol_avg = volume.rolling(10).mean().iloc[-1] if len(volume) >= 10 else volume.mean()
        current_vol = volume.iloc[-1]
        
        # 🚨 BEARISH STOP HUNT (then UP move - CE entry)
        # Pattern: Price sweeps LOW (hunts bull stops), rejects, closes HIGH
        if (current_low < liquidity_low * (1 - 0.006) and  # Sweeps below recent low
            current_close > liquidity_low * 1.003 and       # Closes well above
            current_close > prev_close and                   # Green candle
            current_vol > vol_avg * 1.8 and                 # High volume
            (current_high - current_close) < (current_close - current_low) * 0.5):  # Small upper wick
            return "CE"  # Bull stops hunted, now UP move
        
        # 🚨 BULLISH STOP HUNT (then DOWN move - PE entry)
        # Pattern: Price sweeps HIGH (hunts bear stops), rejects, closes LOW
        if (current_high > liquidity_high * (1 + 0.006) and  # Sweeps above recent high
            current_close < liquidity_high * 0.997 and       # Closes well below
            current_close < prev_close and                   # Red candle
            current_vol > vol_avg * 1.8 and                 # High volume
            (current_close - current_low) < (current_high - current_close) * 0.5):  # Small lower wick
            return "PE"  # Bear stops hunted, now DOWN move
    
    except Exception:
        return None
    return None

# 🚨 **NEW: LIQUIDITY ABSORPTION DETECTION** 🚨
def detect_liquidity_absorption(df):
    """
    Institutions ABSORB selling/buying before big moves.
    Pattern: High volume + small range + closing near extremes
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 5:
            return None
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        current_open = ensure_series(df['Open']).iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Volume analysis
        vol_avg_10 = volume.rolling(10).mean().iloc[-1] if len(volume) >= 10 else volume.mean()
        
        # Calculate candle properties
        candle_range = current_high - current_low
        body_size = abs(current_close - current_open)
        upper_wick = current_high - max(current_close, current_open)
        lower_wick = min(current_close, current_open) - current_low
        
        # 🚨 BULLISH ABSORPTION (CE entry)
        # Pattern: High volume + small range + closing near HIGH (absorbing selling)
        if (current_volume > vol_avg_10 * 2.5 and        # Very high volume
            candle_range < candle_range.iloc[-2] * 0.7 and  # Smaller range than previous
            current_close > current_open and                # Green candle
            lower_wick > body_size * 1.2 and                # Long lower wick (absorption)
            upper_wick < body_size * 0.3):                  # Small upper wick
            return "CE"
        
        # 🚨 BEARISH ABSORPTION (PE entry)
        # Pattern: High volume + small range + closing near LOW (absorbing buying)
        if (current_volume > vol_avg_10 * 2.5 and        # Very high volume
            candle_range < candle_range.iloc[-2] * 0.7 and  # Smaller range than previous
            current_close < current_open and                # Red candle
            upper_wick > body_size * 1.2 and                # Long upper wick (absorption)
            lower_wick < body_size * 0.3):                  # Small lower wick
            return "PE"
    
    except Exception:
        return None
    return None

# 🚨 **NEW: FALSE BREAKOUT TRAP DETECTION** 🚨
def detect_false_breakout_trap(df):
    """
    Institutions create FALSE BREAKOUTS to trap retail.
    Pattern: Break key level + quick reversal + volume confirmation
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
        
        # Find key levels (last 30-60 minutes)
        recent_highs = high.iloc[-12:-2]
        recent_lows = low.iloc[-12:-2]
        
        resistance_level = recent_highs.max()
        support_level = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        # Volume analysis
        vol_avg = volume.rolling(15).mean().iloc[-1] if len(volume) >= 15 else volume.mean()
        current_vol = volume.iloc[-1]
        
        # 🚨 FALSE BULL BREAKOUT (then DOWN - PE entry)
        # Pattern: Break above resistance + close below + high volume
        if (current_high > resistance_level * 1.003 and  # Breaks resistance
            current_close < resistance_level * 0.997 and  # Closes below
            current_close < prev_close and                 # Red candle
            current_vol > vol_avg * 2.0 and               # High volume (trap volume)
            (current_close - current_low) < (current_high - current_close) * 0.4):  # Small lower wick
            return "PE"  # Bull trap - now DOWN
        
        # 🚨 FALSE BEAR BREAKOUT (then UP - CE entry)
        # Pattern: Break below support + close above + high volume
        if (current_low < support_level * 0.997 and      # Breaks support
            current_close > support_level * 1.003 and     # Closes above
            current_close > prev_close and                # Green candle
            current_vol > vol_avg * 2.0 and               # High volume (trap volume)
            (current_high - current_close) < (current_close - current_low) * 0.4):  # Small upper wick
            return "CE"  # Bear trap - now UP
    
    except Exception:
        return None
    return None

# 🚨 **IMPROVED LIQUIDITY ZONE STRATEGY** 🚨
def improved_liquidity_zone_strategy(index, df):
    """
    ENHANCED version of your perfect entry strategy
    Combines liquidity zone with institutional confirmation
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 15:
            return None
        
        # Get liquidity zones
        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        current_price = float(close.iloc[-1])
        
        # Volume confirmation
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        current_vol = volume.iloc[-1]
        
        # Check BEAR liquidity zone (CE entry)
        for zone in bear_liq:
            if zone is None:
                continue
            try:
                # Price near bear liquidity zone
                if abs(current_price - zone) <= 8:
                    # INSTITUTIONAL CONFIRMATION:
                    # 1. Volume spike
                    # 2. Momentum confirmation
                    # 3. Not a false breakout
                    if (current_vol > vol_avg * 1.8 and
                        close.iloc[-1] > close.iloc[-2] and
                        close.iloc[-2] > close.iloc[-3] and
                        high.iloc[-1] > high.iloc[-2]):  # Making higher highs
                        return "CE"
            except:
                continue
        
        # Check BULL liquidity zone (PE entry)
        for zone in bull_liq:
            if zone is None:
                continue
            try:
                # Price near bull liquidity zone
                if abs(current_price - zone) <= 8:
                    # INSTITUTIONAL CONFIRMATION:
                    if (current_vol > vol_avg * 1.8 and
                        close.iloc[-1] < close.iloc[-2] and
                        close.iloc[-2] < close.iloc[-3] and
                        low.iloc[-1] < low.iloc[-2]):  # Making lower lows
                        return "PE"
            except:
                continue
    
    except Exception:
        return None
    return None

# 🚨 **ENHANCED INSTITUTIONAL BLAST DETECTOR** 🚨
def detect_institutional_blast(df):
    """
    Detect REAL institutional moves (not retail spikes)
    Higher thresholds, better confirmation
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_price = ensure_series(df['Open'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 8:
            return None
        
        # Current candle data
        current_open = open_price.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Previous candle data
        prev_open = open_price.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_close = close.iloc[-2]
        prev_volume = volume.iloc[-2]
        
        # Calculate moves
        current_body = abs(current_close - current_open)
        prev_body = abs(prev_close - prev_open)
        
        # Calculate volume average (20 period)
        if len(volume) >= 20:
            vol_avg_20 = volume.rolling(20).mean().iloc[-1]
        else:
            vol_avg_20 = volume.mean()
        
        # 🚨 **STRICT INSTITUTIONAL BLAST CRITERIA** 🚨
        # CHECK 1: MASSIVE GREEN BLAST CANDLE (CE)
        if (current_close > current_open and                    # Green candle
            current_body > prev_body * 3.0 and                  # Body 3x previous (not 2x)
            current_volume > vol_avg_20 * BLAST_VOLUME_THRESHOLD and  # 5x volume
            (current_close - current_low) > current_body * 3.0 and    # Very small lower wick
            current_close > prev_high + (prev_high * 0.005) and       # Breaks high with buffer
            (current_close - prev_close) / prev_close > BLAST_PRICE_MOVE_PCT):  # 1.5%+ move
            return "CE"
        
        # CHECK 2: MASSIVE RED BLAST CANDLE (PE)
        elif (current_close < current_open and                  # Red candle
              current_body > prev_body * 3.0 and                # Body 3x previous
              current_volume > vol_avg_20 * BLAST_VOLUME_THRESHOLD and  # 5x volume
              (current_high - current_close) > current_body * 3.0 and   # Very small upper wick
              current_close < prev_low - (prev_low * 0.005) and         # Breaks low with buffer
              (prev_close - current_close) / prev_close > BLAST_PRICE_MOVE_PCT):  # 1.5%+ move
            return "PE"
    
    except Exception:
        return None
    return None

# 🚨 **ENHANCED SWEEP ORDER DETECTION** 🚨
def detect_sweep_orders(df):
    """
    Detect institutional sweep orders (liquidity grabs)
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 15:
            return None
        
        # Find recent liquidity levels (last hour)
        recent_highs = high.iloc[-12:-2]
        recent_lows = low.iloc[-12:-2]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        # Volume average
        vol_avg = volume.rolling(15).mean().iloc[-1] if len(volume) >= 15 else volume.mean()
        current_vol = volume.iloc[-1]
        
        # 🚨 BEARISH SWEEP (PE) - Sweep highs then drop
        if (current_high > liquidity_high * (1 + SWEEP_DISTANCE_PCT) and
            current_close < liquidity_high * 0.995 and  # Closes well below
            current_vol > vol_avg * 3.0 and             # High volume
            (current_high - current_close) > (current_close - current_low) * 2.0):  # Long upper wick
            return "PE"
        
        # 🚨 BULLISH SWEEP (CE) - Sweep lows then rally
        elif (current_low < liquidity_low * (1 - SWEEP_DISTANCE_PCT) and
              current_close > liquidity_low * 1.005 and  # Closes well above
              current_vol > vol_avg * 3.0 and             # High volume
              (current_close - current_low) > (current_high - current_close) * 2.0):  # Long lower wick
            return "CE"
    
    except Exception:
        return None
    return None

# 🚨 **INSTITUTIONAL MOMENTUM CONFIRMATION** 🚨
def institutional_momentum_confirmation(index, df, proposed_signal):
    """
    STRICT confirmation for institutional moves
    """
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 6:
            return False
        
        # Volume confirmation
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        current_vol = volume.iloc[-1]
        
        if current_vol < vol_avg * 1.5:  # Must have volume
            return False
        
        if proposed_signal == "CE":
            # Must have: 3 consecutive higher closes AND expanding range
            if not (close.iloc[-1] > close.iloc[-2] and 
                    close.iloc[-2] > close.iloc[-3] and
                    close.iloc[-3] > close.iloc[-4]):
                return False
            
            # Range expansion
            current_range = high.iloc[-1] - low.iloc[-1]
            prev_range = high.iloc[-2] - low.iloc[-2]
            if current_range < prev_range * 0.8:  # Range should expand
                return False
                
        elif proposed_signal == "PE":
            # Must have: 3 consecutive lower closes AND expanding range
            if not (close.iloc[-1] < close.iloc[-2] and 
                    close.iloc[-2] < close.iloc[-3] and
                    close.iloc[-3] < close.iloc[-4]):
                return False
            
            # Range expansion
            current_range = high.iloc[-1] - low.iloc[-1]
            prev_range = high.iloc[-2] - low.iloc[-2]
            if current_range < prev_range * 0.8:  # Range should expand
                return False
        
        return True
        
    except Exception:
        return False

# --------- INSTITUTIONAL FLOW CHECKS ---------
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
    
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    try:
        if last_close>=high_zone: return "PE"
        elif last_close<=low_zone: return "CE"
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

# --------- SIMPLIFIED CONFIRMATION ---------
def institutional_confirmation_layer(index, df5, base_signal):
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

# 🚨 **UPDATED INSTITUTIONAL SIGNAL ANALYSIS** 🚨
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    last_close = float(close5.iloc[-1])

    # 🚨 **PRIORITY 1: INSTITUTIONAL STOP HUNT (Highest Probability)**
    stop_hunt_signal = detect_stop_hunt(df5)
    if stop_hunt_signal:
        if institutional_momentum_confirmation(index, df5, stop_hunt_signal):
            return stop_hunt_signal, df5, False, "institutional_stop_hunt"

    # 🚨 **PRIORITY 2: LIQUIDITY ABSORPTION**
    absorption_signal = detect_liquidity_absorption(df5)
    if absorption_signal:
        if institutional_momentum_confirmation(index, df5, absorption_signal):
            return absorption_signal, df5, False, "liquidity_absorption"

    # 🚨 **PRIORITY 3: FALSE BREAKOUT TRAP**
    trap_signal = detect_false_breakout_trap(df5)
    if trap_signal:
        if institutional_momentum_confirmation(index, df5, trap_signal):
            return trap_signal, df5, False, "false_breakout_trap"

    # 🚨 **PRIORITY 4: IMPROVED LIQUIDITY ZONE (Your perfect entry)**
    liquidity_signal = improved_liquidity_zone_strategy(index, df5)
    if liquidity_signal:
        if institutional_momentum_confirmation(index, df5, liquidity_signal):
            return liquidity_signal, df5, False, "liquidity_zone"

    # 🚨 **PRIORITY 5: INSTITUTIONAL BLAST (Real moves only)**
    blast_signal = detect_institutional_blast(df5)
    if blast_signal:
        if institutional_momentum_confirmation(index, df5, blast_signal):
            return blast_signal, df5, False, "institutional_blast"

    # 🚨 **PRIORITY 6: SWEEP ORDER DETECTION**
    sweep_signal = detect_sweep_orders(df5)
    if sweep_signal:
        if institutional_momentum_confirmation(index, df5, sweep_signal):
            return sweep_signal, df5, False, "sweep_order_detection"

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

# 🚨 FIXED: UPDATED SIGNAL SENDING WITH STRICT EXPIRY VALIDATION 🚨
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
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
    
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
    
    # 🚨 **INSTITUTIONAL-SIZED TARGETS** 🚨
    if side == "CE":
        if bull_liq:
            nearest_bull_zone = max([z for z in bull_liq if z is not None])
            price_gap = nearest_bull_zone - signal_detection_price
        else:
            price_gap = signal_detection_price * 0.01  # 1% for institutional
        
        # LARGER TARGETS FOR INSTITUTIONAL MOVES
        if strategy_key in ["institutional_stop_hunt", "institutional_blast"]:
            base_move = max(price_gap * 0.5, 80)  # 80 points for high-probability
        else:
            base_move = max(price_gap * 0.4, 60)  # 60 points for others
        
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.6)  # Tighter SL for institutional
        
    else:
        if bear_liq:
            nearest_bear_zone = min([z for z in bear_liq if z is not None])
            price_gap = signal_detection_price - nearest_bear_zone
        else:
            price_gap = signal_detection_price * 0.01  # 1% for institutional
        
        # LARGER TARGETS FOR INSTITUTIONAL MOVES
        if strategy_key in ["institutional_stop_hunt", "institutional_blast"]:
            base_move = max(price_gap * 0.5, 80)  # 80 points for high-probability
        else:
            base_move = max(price_gap * 0.4, 60)  # 60 points for others
        
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.6)  # Tighter SL for institutional
    
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
    
    # 🚨 **INSTITUTIONAL ALERTS** 🚨
    if strategy_key in ["institutional_stop_hunt", "institutional_blast"]:
        msg = (f"💥 **INSTITUTIONAL MOVE DETECTED** 💥\n"
               f"🎯 {index} {strike} {side}\n"
               f"SYMBOL: {symbol}\n"
               f"ENTRY ABOVE: ₹{entry}\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ INSTITUTIONAL FLOW - HIGH PROBABILITY")
    else:
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

# --------- FIXED: UPDATED TRADE THREAD WITH ISOLATED INDICES ---------
def trade_thread(index):
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    if len(result) == 4:
        side, df, fakeout, strategy_key = result
    else:
        side, df, fakeout = result
        strategy_key = "unknown"
    
    df5 = fetch_index_data(index, "5m", "2d")
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

# --------- FIXED: MAIN LOOP (KEPT INDICES ONLY) ---------
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
            send_telegram("🚀 **INSTITUTIONAL ALGO ACTIVATED**\n"
                         "✅ Institutional Stop Hunt: ACTIVE\n"
                         "✅ Liquidity Absorption: ACTIVE\n"
                         "✅ False Breakout Traps: ACTIVE\n"
                         "✅ Improved Liquidity Zone: ACTIVE\n"
                         "✅ Institutional Blast: ACTIVE\n"
                         "✅ Sweep Orders: ACTIVE\n"
                         "🎯 ONLY 6 HIGH-PROBABILITY STRATEGIES\n"
                         "⚠️ SIGNALS BEFORE BIG MOVES ONLY")
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
