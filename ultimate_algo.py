# ULTIMATE INSTITUTIONAL INTELLIGENCE ANALYZER WITH COMPLETE 3-CANDLE ANALYSIS

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

# --------- INSTITUTIONAL MONITORING CONFIG ---------
BIG_CANDLE_THRESHOLD = 20  # Changed to 20 points as requested
MOVE_TIME_WINDOW = 20
ANALYSIS_COOLDOWN = 30

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")

def angel_one_login():
    """Login to Angel One without error messages"""
    try:
        TOTP = pyotp.TOTP(TOTP_SECRET).now()
        client = SmartConnect(api_key=API_KEY)
        session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
        return client
    except Exception:
        return None

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=5)
        return True
    except:
        return False

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

# --------- DATA FETCHING ---------
def fetch_index_data_complete(index, interval="5m"):
    """Fetch complete data for analysis with multiple timeframes"""
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        
        # Fetch data for today
        today = datetime.now().strftime("%Y-%m-%d")
        df = yf.download(symbol_map[index], start=today, interval=interval, progress=False)
        
        if df.empty:
            return None
            
        return df
        
    except Exception:
        return None

def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- COMPLETE 3-CANDLE INSTITUTIONAL ANALYZER ---------
class CompleteInstitutionalAnalyzer:
    def __init__(self):
        self.client = angel_one_login()
        self.last_analysis_time = {}
        self.analyzed_candles = set()
        
    def analyze_previous_3_candles_detailed(self, df, big_candle_idx):
        """COMPLETE analysis of previous 3 candles before big move"""
        try:
            if len(df) <= big_candle_idx or big_candle_idx < 3:
                return None
            
            # Get current big candle and previous 3 candles
            current_candle = df.iloc[big_candle_idx]
            prev1_candle = df.iloc[big_candle_idx-1]
            prev2_candle = df.iloc[big_candle_idx-2]  
            prev3_candle = df.iloc[big_candle_idx-3]
            
            # Calculate big candle details
            big_candle_move = abs(current_candle['Close'] - current_candle['Open'])
            direction = "GREEN" if current_candle['Close'] > current_candle['Open'] else "RED"
            
            analysis = {
                'timestamp': df.index[big_candle_idx],
                'time_str': df.index[big_candle_idx].strftime('%H:%M:%S'),
                'direction': direction,
                'points_moved': round(float(big_candle_move), 2),
                'candle_range': round(float(current_candle['High'] - current_candle['Low']), 2),
                'volume': int(current_candle['Volume']),
                
                # Previous 3 candles COMPLETE information
                'prev_candles': []
            }
            
            # Analyze previous 3 candles in extreme detail
            prev_candles = [prev3_candle, prev2_candle, prev1_candle]
            for i, candle in enumerate(prev_candles):
                candle_data = {
                    'time': df.index[big_candle_idx-3+i].strftime('%H:%M:%S'),
                    'open': round(float(candle['Open']), 2),
                    'high': round(float(candle['High']), 2), 
                    'low': round(float(candle['Low']), 2),
                    'close': round(float(candle['Close']), 2),
                    'points_move': round(abs(float(candle['Close']) - float(candle['Open'])), 2),
                    'direction': "GREEN" if candle['Close'] > candle['Open'] else "RED",
                    'volume': int(candle['Volume']),
                    'range': round(float(candle['High'] - candle['Low']), 2),
                    'body_ratio': self.calculate_body_ratio(candle),
                    'wick_analysis': self.analyze_wicks(candle)
                }
                analysis['prev_candles'].append(candle_data)
            
            # Institutional metrics for the big move
            analysis.update(self.calculate_institutional_metrics(df, big_candle_idx, prev_candles))
            
            return analysis
            
        except Exception as e:
            print(f"3-candle analysis error: {e}")
            return None
    
    def calculate_body_ratio(self, candle):
        """Calculate candle body to range ratio"""
        body_size = abs(float(candle['Close']) - float(candle['Open']))
        total_range = float(candle['High']) - float(candle['Low'])
        return round(body_size / total_range, 3) if total_range > 0 else 0
    
    def analyze_wicks(self, candle):
        """Analyze upper and lower wicks"""
        body_high = max(float(candle['Open']), float(candle['Close']))
        body_low = min(float(candle['Open']), float(candle['Close']))
        
        upper_wick = float(candle['High']) - body_high
        lower_wick = body_low - float(candle['Low'])
        total_range = float(candle['High']) - float(candle['Low'])
        
        return {
            'upper_wick_ratio': round(upper_wick / total_range, 3) if total_range > 0 else 0,
            'lower_wick_ratio': round(lower_wick / total_range, 3) if total_range > 0 else 0,
            'pressure': 'SELLING' if upper_wick > lower_wick else 'BUYING' if lower_wick > upper_wick else 'BALANCED'
        }
    
    def calculate_institutional_metrics(self, df, big_candle_idx, prev_candles):
        """Calculate institutional trading metrics"""
        try:
            current_candle = df.iloc[big_candle_idx]
            
            # Volume Analysis
            current_volume = float(current_candle['Volume'])
            prev_volumes = [float(c['Volume']) for c in prev_candles]
            avg_prev_volume = np.mean(prev_volumes)
            
            volume_surge_ratio = round(current_volume / max(1, avg_prev_volume), 2)
            volume_change_percent = round(((current_volume - avg_prev_volume) / max(1, avg_prev_volume)) * 100, 2)
            
            # Price Momentum Analysis
            prev_closes = [float(c['Close']) for c in prev_candles]
            price_momentum = (prev_closes[-1] - prev_closes[0]) / prev_closes[0] * 100
            
            # Volatility Analysis
            current_range_pct = (float(current_candle['High']) - float(current_candle['Low'])) / float(current_candle['Open']) * 100
            prev_ranges = []
            for candle in prev_candles:
                range_pct = (float(candle['High']) - float(candle['Low'])) / float(candle['Open']) * 100
                prev_ranges.append(range_pct)
            
            avg_prev_range = np.mean(prev_ranges)
            volatility_expansion = round(((current_range_pct - avg_prev_range) / max(0.1, avg_prev_range)) * 100, 2)
            
            # Order Flow Pressure
            green_candles = sum(1 for c in prev_candles if c['Close'] > c['Open'])
            buying_pressure_ratio = round(green_candles / 3, 2)
            
            # Institutional Probability Score
            score = 0
            if volume_surge_ratio > 1.5: score += 30
            if volatility_expansion > 50: score += 25
            if abs(price_momentum) > 0.1: score += 20
            if abs(current_candle['Close'] - current_candle['Open']) > 25: score += 25
            
            institutional_score = min(100, score)
            institutional_confidence = "HIGH" if score >= 70 else "MEDIUM" if score >= 50 else "LOW"
            
            # Aggressive Trading Detection
            aggressive_trading = self.detect_aggressive_trading(prev_candles, current_candle)
            
            return {
                'volume_surge_ratio': volume_surge_ratio,
                'volume_change_percent': volume_change_percent,
                'prev_momentum_percent': round(price_momentum, 2),
                'volatility_expansion': volatility_expansion,
                'buying_pressure_ratio': buying_pressure_ratio,
                'institutional_score': institutional_score,
                'institutional_confidence': institutional_confidence,
                'aggressive_trading': aggressive_trading,
                'what_happened': self.explain_what_happened(current_candle, prev_candles, volume_surge_ratio, volatility_expansion)
            }
            
        except Exception as e:
            print(f"Institutional metrics error: {e}")
            return {}
    
    def detect_aggressive_trading(self, prev_candles, current_candle):
        """Detect aggressive institutional trading"""
        try:
            # Check for consecutive same-direction candles
            directions = [1 if c['Close'] > c['Open'] else -1 for c in prev_candles]
            current_direction = 1 if current_candle['Close'] > current_candle['Open'] else -1
            
            if all(d == current_direction for d in directions):
                return "AGGRESSIVE_CONTINUATION"
            elif sum(directions) == 0 and current_direction != 0:
                return "AGGRESSIVE_REVERSAL"
            else:
                return "MIXED_SENTIMENT"
                
        except:
            return "UNKNOWN"
    
    def explain_what_happened(self, current_candle, prev_candles, volume_surge, volatility_expansion):
        """Explain what caused the big move"""
        direction = "GREEN" if current_candle['Close'] > current_candle['Open'] else "RED"
        points_moved = abs(float(current_candle['Close']) - float(current_candle['Open']))
        
        explanation = f"{direction} move of {points_moved} points caused by "
        
        if volume_surge > 2.0 and volatility_expansion > 50:
            explanation += "STRONG INSTITUTIONAL PARTICIPATION with high volume surge and volatility expansion"
        elif volume_surge > 1.5:
            explanation += "MODERATE INSTITUTIONAL BUYING/SELLING with significant volume increase"
        elif volatility_expansion > 30:
            explanation += "VOLATILITY EXPANSION indicating institutional order flow"
        else:
            explanation += "PRICE MOMENTUM with retail participation"
        
        # Add previous context
        prev_directions = ["GREEN" if c['Close'] > c['Open'] else "RED" for c in prev_candles]
        green_count = sum(1 for d in prev_directions if d == "GREEN")
        
        if green_count >= 2 and direction == "GREEN":
            explanation += " | Building on previous bullish momentum"
        elif green_count <= 1 and direction == "GREEN":
            explanation += " | Reversing previous bearish sentiment"
        elif green_count >= 2 and direction == "RED":
            explanation += " | Reversing previous bullish momentum"
        else:
            explanation += " | Continuing previous bearish sentiment"
        
        return explanation
    
    def find_all_big_candles(self, df, threshold=20):
        """Find ALL big candles >= threshold points"""
        big_candles = []
        try:
            if df is None or len(df) < 4:
                return big_candles
                
            for i in range(3, len(df)):
                candle_move = abs(float(df['Close'].iloc[i]) - float(df['Open'].iloc[i]))
                if candle_move >= threshold:
                    analysis = self.analyze_previous_3_candles_detailed(df, i)
                    if analysis:
                        big_candles.append(analysis)
                        
            return big_candles
            
        except Exception as e:
            print(f"Error finding big candles: {e}")
            return []
    
    def format_complete_analysis_message(self, index, timeframe, analysis):
        """Format COMPLETE analysis message with previous 3 candles"""
        
        # Format previous candles information
        prev_candles_text = ""
        for i, candle in enumerate(analysis['prev_candles'], 1):
            wick_info = candle['wick_analysis']
            prev_candles_text += f"""
    {i}. {candle['time']} - {candle['direction']} {candle['points_move']} points
       O: {candle['open']} | H: {candle['high']} | L: {candle['low']} | C: {candle['close']}
       Range: {candle['range']} pts | Volume: {candle['volume']:,}
       Body Ratio: {candle['body_ratio']} | Wick Pressure: {wick_info['pressure']}"""
    
        msg = f"""
🔴🟢 **BIG CANDLE DETECTED - {index} {timeframe}** 🔴🟢

⏰ **TIME**: {analysis['time_str']}
🎯 **DIRECTION**: {analysis['direction']}
📈 **POINTS MOVED**: {analysis['points_moved']} points
📊 **CANDLE RANGE**: {analysis['candle_range']} points  
📦 **VOLUME**: {analysis['volume']:,}

📋 **PREVIOUS 3 CANDLES ANALYSIS**:{prev_candles_text}

📊 **INSTITUTIONAL METRICS**:
• Volume Surge: {analysis['volume_surge_ratio']}x
• Volume Change: {analysis['volume_change_percent']}%
• Previous Momentum: {analysis['prev_momentum_percent']}%
• Volatility Expansion: {analysis['volatility_expansion']}%
• Buying Pressure: {analysis['buying_pressure_ratio']}
• Aggressive Trading: {analysis['aggressive_trading']}

🏛️ **INSTITUTIONAL ASSESSMENT**:
• Institutional Score: {analysis['institutional_score']}/100
• Confidence: {analysis['institutional_confidence']}

💡 **WHAT HAPPENED**:
{analysis['what_happened']}

🎯 **TRADING IMPLICATION**:
Consider {analysis['direction']} positions with {analysis['institutional_confidence'].lower()} confidence
Institutional activity probability: {analysis['institutional_score']}%

─────────────────────────────
"""
        return msg

# --------- MAIN MONITORING FUNCTION ---------
def monitor_all_indices_complete_analysis():
    """Monitor all indices for big candles with complete 3-candle analysis"""
    
    analyzer = CompleteInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["1m", "5m"]
    
    startup_msg = f"""
📊 **COMPLETE INSTITUTIONAL ANALYSIS STARTED**
📅 Date: {datetime.now().strftime('%d %b %Y')}
🎯 Target: {BIG_CANDLE_THRESHOLD}+ points moves
📈 Indices: NIFTY, BANKNIFTY, SENSEX
⏰ Timeframes: 1min + 5min
🔍 Analyzing ALL big moves with COMPLETE previous 3 candles context

**MONITORING STARTED...**
"""
    send_telegram(startup_msg)
    print("Starting complete institutional analysis...")
    
    total_big_moves_found = 0
    
    while True:
        try:
            if not is_market_open():
                print("Market closed. Waiting...")
                time.sleep(300)
                continue
            
            for index in indices:
                for timeframe in timeframes:
                    try:
                        # Fetch data
                        df = fetch_index_data_complete(index, timeframe)
                        
                        if df is not None and len(df) > 10:
                            # Find all big candles
                            big_candles = analyzer.find_all_big_candles(df, BIG_CANDLE_THRESHOLD)
                            
                            # Send analysis for each big candle
                            for analysis in big_candles:
                                candle_id = f"{index}_{timeframe}_{analysis['time_str']}"
                                
                                if candle_id not in analyzer.analyzed_candles:
                                    message = analyzer.format_complete_analysis_message(index, timeframe, analysis)
                                    
                                    if send_telegram(message):
                                        print(f"✅ Sent analysis for {index} {timeframe} at {analysis['time_str']}")
                                        analyzer.analyzed_candles.add(candle_id)
                                        total_big_moves_found += 1
                                    
                                    time.sleep(3)  # Avoid rate limiting
                        
                        time.sleep(2)
                        
                    except Exception as e:
                        print(f"Error analyzing {index} {timeframe}: {e}")
                        continue
            
            # Clean up old analyzed candles (keep only last 4 hours)
            current_time = datetime.now()
            analyzer.analyzed_candles = {
                candle_id for candle_id in analyzer.analyzed_candles
                if current_time.hour - int(candle_id.split('_')[-1].split(':')[0]) <= 4
            }
            
            print(f"🔄 Completed scan cycle. Total big moves found: {total_big_moves_found}")
            time.sleep(60)  # Wait 1 minute between scans
            
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(60)

# --------- START THE COMPLETE ANALYSIS ---------
if __name__ == "__main__":
    print("🚀 Starting Complete Institutional Analysis with 3-Candle Context...")
    monitor_all_indices_complete_analysis()
