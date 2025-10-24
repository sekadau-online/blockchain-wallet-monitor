import requests
import datetime
import time
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading

# === KONFIGURASI LENGKAP ===
CONFIG = {
    # Telegram Configuration
    "TELEGRAM_BOT_TOKEN": "8345066310:AAElrMezSmJZwWOWWRYLFMf2z5nyDCkTg0g",
    "TELEGRAM_CHAT_ID": "-4848345455",

    # Trading Symbols
    "SYMBOLS": ["BTCUSDT"],

    # Analysis Timing
    "INTERVAL": 60 * 1,  # 1 menit dalam detik
    "TIMEZONE": "Asia/Pontianak",  # Tetap menggunakan Asia/Pontianak

    # Risk Management
    "RISK_LEVEL": "MEDIUM",  # LOW, MEDIUM, HIGH
    "MAX_POSITION_SIZE": 0.05,  # 5% dari equity maksimal

    # Volume Analysis Parameters
    "VOLUME_THRESHOLD": 1.5,
    "MIN_VOLUME": 1000000,  # Minimum volume dalam USDT
    "VOLUME_HISTORY_LENGTH": 20,

    # API Configuration
    "MAX_RETRIES": 3,
    "REQUEST_DELAY": 1,
    "TIMEOUT": 20,

    # Technical Analysis Parameters
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "RSI_PERIOD": 14,

    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,

    "EMA_SHORT": 12,
    "EMA_LONG": 26,
    "SMA_SHORT": 10,
    "SMA_LONG": 20,

    # Price History
    "PRICE_HISTORY_LENGTH": 50,

    # Technical Analysis Timeframes
    "TIMEFRAMES": ["15m", "1h"],
    "KLINE_LIMIT": 100,
    "MAIN_TIMEFRAME": "15m",

    # Support Resistance Parameters
    "SR_PERIOD": 10,

    # Momentum Parameters
    "MOMENTUM_PERIOD": 5,

    # Trend Parameters
    "TREND_STRONG_THRESHOLD": 1.02,
    "TREND_WEAK_THRESHOLD": 1.01,

    # Volume Trend Parameters
    "VOLUME_TREND_INCREASE": 1.15,
    "VOLUME_TREND_DECREASE": 0.85,

    # Sentiment Scoring Weights
    "WEIGHT_FUNDING": 0.25,
    "WEIGHT_OI": 0.20,
    "WEIGHT_TAKER": 0.20,
    "WEIGHT_LONGSHORT": 0.15,
    "WEIGHT_DEPTH": 0.10,
    "WEIGHT_VOLUME": 0.10,

    # Signal Generation Thresholds
    "MIN_CONFIDENCE": 50,
    "SCORE_THRESHOLD_STRONG": 6,
    "SCORE_THRESHOLD_MEDIUM": 3,
    "SCORE_THRESHOLD_WEAK": 1,

    # Entry Signal Parameters
    "ENTRY_SCORE_DIFFERENCE": 3,
    "STOP_LOSS_PERCENT": 0.5,
    "TAKE_PROFIT_1_PERCENT": 1.0,
    "TAKE_PROFIT_2_PERCENT": 2.0,

    # Position Sizing berdasarkan Risk Level
    "POSITION_SIZING": {
        "LOW": {"MIN": 0.01, "MAX": 0.02, "TP1": 0.03, "TP2": 0.05, "SL": 0.02},
        "MEDIUM": {"MIN": 0.03, "MAX": 0.05, "TP1": 0.05, "TP2": 0.08, "SL": 0.03},
        "HIGH": {"MIN": 0.05, "MAX": 0.07, "TP1": 0.08, "TP2": 0.12, "SL": 0.04},
    },

    # Funding Rate Thresholds
    "FUNDING_STRONG_POSITIVE": 0.001,
    "FUNDING_WEAK_POSITIVE": 0.0001,
    "FUNDING_STRONG_NEGATIVE": -0.001,
    "FUNDING_WEAK_NEGATIVE": -0.0001,

    # Open Interest Thresholds
    "OI_STRONG_INCREASE": 2.0,
    "OI_WEAK_INCREASE": 0.5,
    "OI_STRONG_DECREASE": -2.0,
    "OI_WEAK_DECREASE": -0.5,

    # Taker Ratio Thresholds
    "TAKER_STRONG_BUY": 1.2,
    "TAKER_WEAK_BUY": 1.05,
    "TAKER_STRONG_SELL": 0.8,
    "TAKER_WEAK_SELL": 0.95,

    # Long/Short Ratio Thresholds
    "LONGSHORT_STRONG_DIFFERENCE": 0.15,
    "LONGSHORT_WEAK_DIFFERENCE": 0.05,

    # Market Depth Thresholds
    "DEPTH_STRONG_BID": 1.2,
    "DEPTH_WEAK_BID": 1.05,
    "DEPTH_STRONG_ASK": 0.8,
    "DEPTH_WEAK_ASK": 0.95,

    # Retry Configuration
    "RATE_LIMIT_WAIT": 10,
    "IP_BAN_WAIT": 300,

    # === FLASH CRASH DETECTION CONFIG ===
    "FLASH_CRASH_THRESHOLD": 0.03,  # 3% drop dalam 5 menit
    "VOLUME_SPIKE_THRESHOLD": 3.0,  # 3x volume rata-rata
    "PRICE_ALERT_INTERVAL": 30,  # Detik antara price check
    "LIQUIDATION_ALERT_THRESHOLD": 1000000,  # $1M liquidasi dalam 5m
    "REALTIME_SYMBOLS": ["BTCUSDT", "ETHUSDT"],
    "PRICE_MONITOR_INTERVAL": 10,  # Detik
}

# ✅ URL dasar
BASE_URL = "https://fapi.binance.com"
TELEGRAM_BASE_URL = "https://api.telegram.org/bot"


class FlashCrashDetector:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.price_history = {}
        self.alert_cooldown = {}
        self.monitoring_active = False
        
        for symbol in CONFIG["REALTIME_SYMBOLS"]:
            self.price_history[symbol] = {
                'prices': deque(maxlen=30),
                'timestamps': deque(maxlen=30),
                'volumes': deque(maxlen=30)
            }
            self.alert_cooldown[symbol] = 0

    def start_realtime_monitoring(self):
        """Memulai monitoring real-time untuk flash crash"""
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._realtime_monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        print("🚀 Flash Crash Detector Activated!")

    def _realtime_monitor_loop(self):
        """Loop monitoring real-time"""
        while self.monitoring_active:
            try:
                for symbol in CONFIG["REALTIME_SYMBOLS"]:
                    self._check_symbol_price(symbol)
                    time.sleep(2)
                
                time.sleep(CONFIG["PRICE_MONITOR_INTERVAL"])
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(10)

    def _check_symbol_price(self, symbol: str):
        """Cek pergerakan harga real-time untuk satu simbol"""
        try:
            url = f"{BASE_URL}/fapi/v1/ticker/24hr?symbol={symbol}"
            data = self.analyzer.safe_get(url, max_retries=1)
            
            if not data:
                return
                
            current_price = float(data['lastPrice'])
            current_volume = float(data.get('quoteVolume', 0))
            current_time = time.time()
            
            history = self.price_history[symbol]
            history['prices'].append(current_price)
            history['timestamps'].append(current_time)
            history['volumes'].append(current_volume)
            
            self._detect_flash_crash(symbol, history, current_price, current_volume)
            self._detect_volume_spike(symbol, history, current_volume)
            
        except Exception as e:
            print(f"⚠️ Price check error for {symbol}: {e}")

    def _detect_flash_crash(self, symbol: str, history: Dict, current_price: float, current_volume: float):
        """Deteksi flash crash berdasarkan pergerakan harga"""
        if len(history['prices']) < 6:
            return
            
        five_min_ago = time.time() - 300
        recent_prices = []
        
        for i, timestamp in enumerate(history['timestamps']):
            if timestamp >= five_min_ago:
                recent_prices.append(history['prices'][i])
        
        if len(recent_prices) < 2:
            return
            
        max_price = max(recent_prices)
        price_drop = (max_price - current_price) / max_price
        
        if price_drop >= CONFIG["FLASH_CRASH_THRESHOLD"]:
            if time.time() - self.alert_cooldown.get(symbol, 0) > 300:
                self._trigger_flash_crash_alert(symbol, price_drop, current_price, max_price, current_volume)
                self.alert_cooldown[symbol] = time.time()

    def _detect_volume_spike(self, symbol: str, history: Dict, current_volume: float):
        """Deteksi spike volume yang tidak normal"""
        if len(history['volumes']) < 10:
            return
            
        avg_volume = np.mean(list(history['volumes'])[:-1])
        if avg_volume <= 0:
            return
            
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio >= CONFIG["VOLUME_SPIKE_THRESHOLD"]:
            if time.time() - self.alert_cooldown.get(f"{symbol}_volume", 0) > 300:
                self._trigger_volume_alert(symbol, volume_ratio, current_volume, avg_volume)
                self.alert_cooldown[f"{symbol}_volume"] = time.time()

    def _trigger_flash_crash_alert(self, symbol: str, drop_percent: float, current_price: float, 
                                 max_price: float, volume: float):
        """Trigger alert untuk flash crash"""
        drop_pct = drop_percent * 100
        
        message = (
            f"🚨🚨 <b>FLASH CRASH DETECTED!</b> 🚨🚨\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Price Drop:</b> <code>{drop_pct:.2f}%</code>\n"
            f"<b>From:</b> <code>${max_price:,.2f}</code>\n"
            f"<b>To:</b> <code>${current_price:,.2f}</code>\n"
            f"<b>Volume:</b> <code>{volume:,.0f} USDT</code>\n"
            f"<b>Time:</b> {self.analyzer.get_local_time()}\n\n"
            f"<b>⚠️ ACTION REQUIRED:</b>\n"
            f"• Check position safety\n"
            f"• Review stop-loss levels\n"
            f"• Monitor for recovery\n"
            f"• Avoid FOMO buying\n"
        )
        
        self.analyzer.send_telegram(message, f"CRASH_{symbol}")
        print(f"🚨 Flash Crash Alert for {symbol}: {drop_pct:.2f}% drop")

    def _trigger_volume_alert(self, symbol: str, volume_ratio: float, current_volume: float, avg_volume: float):
        """Trigger alert untuk volume spike"""
        message = (
            f"📊 <b>VOLUME SPIKE DETECTED!</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Volume Ratio:</b> <code>{volume_ratio:.1f}x</code> average\n"
            f"<b>Current Volume:</b> <code>{current_volume:,.0f} USDT</code>\n"
            f"<b>Average Volume:</b> <code>{avg_volume:,.0f} USDT</code>\n"
            f"<b>Time:</b> {self.analyzer.get_local_time()}\n\n"
            f"<i>Possible large movement incoming</i>"
        )
        
        self.analyzer.send_telegram(message, f"VOLUME_{symbol}")
        print(f"📊 Volume Spike for {symbol}: {volume_ratio:.1f}x")


class EnhancedFuturesAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.analysis_history = {}
        self.volume_history = {}
        self.price_history = {}

        self.crash_detector = FlashCrashDetector(self)

        for symbol in config["SYMBOLS"]:
            self.volume_history[symbol] = deque(maxlen=config["VOLUME_HISTORY_LENGTH"])
            self.price_history[symbol] = deque(maxlen=config["PRICE_HISTORY_LENGTH"])

    def get_local_time(self):
        """Dapatkan waktu lokal Asia/Pontianak (UTC+7)"""
        utc_time = datetime.datetime.now(datetime.timezone.utc)
        # Asia/Pontianak = UTC+7
        local_time = utc_time + datetime.timedelta(hours=7)
        return local_time.strftime("%Y-%m-%d %H:%M:%S") + " (Asia/Pontianak)"

    def safe_get(self, url: str, max_retries: int = None):
        """Safe HTTP GET request dengan retry mechanism"""
        if max_retries is None:
            max_retries = self.config["MAX_RETRIES"]

        for attempt in range(max_retries):
            try:
                time.sleep(self.config["REQUEST_DELAY"])
                response = self.session.get(url, timeout=self.config["TIMEOUT"])

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) + self.config["RATE_LIMIT_WAIT"]
                    print(f"⚠️ Rate limit detected, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 418:
                    wait_time = self.config["IP_BAN_WAIT"]
                    print(f"🚫 IP Banned, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ HTTP {response.status_code}, retrying...")
                    time.sleep(5)

            except requests.exceptions.Timeout:
                print(f"⚠️ Timeout attempt {attempt + 1}, retrying...")
                time.sleep(10)
            except requests.exceptions.ConnectionError:
                print(f"⚠️ Connection error attempt {attempt + 1}, retrying...")
                time.sleep(10)
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                time.sleep(5)

        print(f"❌ Failed to fetch data from {url} after {max_retries} attempts")
        return None

    def send_telegram(self, message: str, symbol: str = "GLOBAL"):
        """Kirim pesan ke Telegram"""
        try:
            url = f"{TELEGRAM_BASE_URL}{self.config['TELEGRAM_BOT_TOKEN']}/sendMessage"
            payload = {
                "chat_id": self.config["TELEGRAM_CHAT_ID"],
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            response = self.session.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                print(f"✅ [{symbol}] Pesan terkirim ke Telegram")
            else:
                print(f"⚠️ [{symbol}] Gagal kirim: {response.status_code}")
        except Exception as e:
            print(f"⚠️ [{symbol}] Error Telegram: {e}")

    def get_funding_rate(self, symbol: str) -> float:
        """Dapatkan funding rate terbaru"""
        try:
            url = f"{BASE_URL}/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            data = self.safe_get(url)
            if data and len(data) > 0:
                return float(data[0]["fundingRate"])
            return 0.0
        except Exception as e:
            print(f"⚠️ Error getting funding rate for {symbol}: {e}")
            return 0.0

    def get_open_interest(self, symbol: str) -> tuple:
        """Dapatkan open interest data"""
        try:
            url = f"{BASE_URL}/fapi/v1/openInterest?symbol={symbol}"
            data = self.safe_get(url)
            if data:
                return [time.time()], [float(data.get("openInterest", 0))]
            return [], [0]
        except Exception as e:
            print(f"⚠️ Error getting OI for {symbol}: {e}")
            return [], [0]

    def get_taker_ratio(self, symbol: str) -> float:
        """Dapatkan taker buy/sell ratio"""
        try:
            url = f"{BASE_URL}/fapi/v1/ticker/24hr?symbol={symbol}"
            data = self.safe_get(url)
            if data:
                buy_volume = float(data.get("volume", 0))
                sell_volume = float(data.get("quoteVolume", 0))
                if sell_volume > 0:
                    return buy_volume / sell_volume
            return 1.0
        except Exception as e:
            print(f"⚠️ Error getting taker ratio for {symbol}: {e}")
            return 1.0

    def get_long_short_ratio(self, symbol: str) -> tuple:
        """Dapatkan long/short ratio"""
        try:
            return 0.5, 0.5
        except Exception as e:
            print(f"⚠️ Error getting long/short ratio for {symbol}: {e}")
            return 0.5, 0.5

    def get_price_data(self, symbol: str) -> Dict:
        """Dapatkan data harga terbaru"""
        try:
            url = f"{BASE_URL}/fapi/v1/ticker/24hr?symbol={symbol}"
            data = self.safe_get(url)
            if data:
                return {
                    "price": float(data["lastPrice"]),
                    "change_24h": float(data["priceChangePercent"]),
                    "high_24h": float(data["highPrice"]),
                    "low_24h": float(data["lowPrice"]),
                    "volume": float(data["volume"]),
                    "quote_volume": float(data["quoteVolume"]),
                    "price_change": float(data["priceChange"]),
                    "count": int(data["count"])
                }
        except Exception as e:
            print(f"⚠️ Error getting price data for {symbol}: {e}")
        
        return {
            "price": 0,
            "change_24h": 0,
            "high_24h": 0,
            "low_24h": 0,
            "volume": 0,
            "quote_volume": 0,
            "price_change": 0,
            "count": 0
        }

    def get_market_depth(self, symbol: str) -> Dict:
        """Dapatkan market depth data"""
        try:
            url = f"{BASE_URL}/fapi/v1/depth?symbol={symbol}&limit=10"
            data = self.safe_get(url, max_retries=2)

            if data and "bids" in data and "asks" in data:
                bids = sum(float(bid[1]) * float(bid[0]) for bid in data["bids"][:5])
                asks = sum(float(ask[1]) * float(ask[0]) for ask in data["asks"][:5])
                total_bids = sum(float(bid[1]) for bid in data["bids"][:5])
                total_asks = sum(float(ask[1]) for ask in data["asks"][:5])

                pressure = bids/asks if asks > 0 else 1
                volume_pressure = total_bids/total_asks if total_asks > 0 else 1

                return {
                    "bids_value": bids,
                    "asks_value": asks, 
                    "bids_volume": total_bids,
                    "asks_volume": total_asks,
                    "pressure": pressure,
                    "volume_pressure": volume_pressure
                }
        except Exception as e:
            print(f"⚠️ Depth data unavailable for {symbol}: {e}")
        
        return {
            "bids_value": 1,
            "asks_value": 1, 
            "bids_volume": 1,
            "asks_volume": 1,
            "pressure": 1,
            "volume_pressure": 1
        }

    def calculate_volume_analysis(self, symbol: str, current_volume: float) -> Dict:
        """Analisis volume trading"""
        volume_history = self.volume_history[symbol]
        volume_history.append(current_volume)

        if len(volume_history) < 3:
            return {"volume_spike": False, "volume_trend": "INSUFFICIENT_DATA", "volume_ratio": 1.0}

        avg_volume = np.mean(list(volume_history))
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        volume_spike = volume_ratio > self.config["VOLUME_THRESHOLD"]

        volume_trend = "STABLE"
        if len(volume_history) >= 6:
            recent_volumes = list(volume_history)[-3:]
            older_volumes = list(volume_history)[-6:-3]
            recent_avg = np.mean(recent_volumes)
            older_avg = np.mean(older_volumes)
            if recent_avg > older_avg * self.config["VOLUME_TREND_INCREASE"]:
                volume_trend = "INCREASING"
            elif recent_avg < older_avg * self.config["VOLUME_TREND_DECREASE"]:
                volume_trend = "DECREASING"

        return {
            "volume_spike": volume_spike,
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "avg_volume": avg_volume,
            "current_volume": current_volume
        }

    def calculate_ema(self, prices: np.array, period: int) -> float:
        """Hitung Exponential Moving Average"""
        if len(prices) < period:
            return float(prices[-1])
        try:
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            return np.convolve(prices, weights, mode='valid')[-1]
        except:
            return float(prices[-1])

    def calculate_rsi(self, prices: np.array, period: int = None) -> float:
        """Hitung Relative Strength Index"""
        if period is None:
            period = self.config["RSI_PERIOD"]
        if len(prices) < period + 1:
            return 50.0
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        except:
            return 50.0

    def calculate_advanced_technicals(self, symbol: str) -> Dict:
        """Hitung indikator teknikal advanced"""
        technicals = {}
        for tf in self.config["TIMEFRAMES"]:
            try:
                url = f"{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={tf}&limit={self.config['KLINE_LIMIT']}"
                data = self.safe_get(url, max_retries=2)
                
                if not data or len(data) < 20:
                    continue
                    
                closes = np.array([float(c[4]) for c in data])
                highs = np.array([float(h[2]) for h in data])
                lows = np.array([float(l[3]) for l in data])

                sma_short = np.mean(closes[-self.config["SMA_SHORT"]:])
                sma_long = np.mean(closes[-self.config["SMA_LONG"]:])
                
                ema_short = self.calculate_ema(closes, self.config["EMA_SHORT"])
                ema_long = self.calculate_ema(closes, self.config["EMA_LONG"])
                
                rsi = self.calculate_rsi(closes, self.config["RSI_PERIOD"])
                
                resistance = np.max(highs[-self.config["SR_PERIOD"]:])
                support = np.min(lows[-self.config["SR_PERIOD"]:])
                
                momentum = ((closes[-1] - closes[-self.config["MOMENTUM_PERIOD"]]) / 
                          closes[-self.config["MOMENTUM_PERIOD"]]) * 100 if closes[-self.config["MOMENTUM_PERIOD"]] > 0 else 0

                price_trend = "NEUTRAL"
                if ema_short > ema_long and closes[-1] > sma_long * self.config["TREND_STRONG_THRESHOLD"]:
                    price_trend = "STRONG_BULLISH"
                elif ema_short < ema_long and closes[-1] < sma_long * (2 - self.config["TREND_STRONG_THRESHOLD"]):
                    price_trend = "STRONG_BEARISH"
                elif ema_short > ema_long and closes[-1] > sma_long * self.config["TREND_WEAK_THRESHOLD"]:
                    price_trend = "MILD_BULLISH"
                elif ema_short < ema_long and closes[-1] < sma_long * (2 - self.config["TREND_WEAK_THRESHOLD"]):
                    price_trend = "MILD_BEARISH"
                elif ema_short > ema_long:
                    price_trend = "BULLISH"
                elif ema_short < ema_long:
                    price_trend = "BEARISH"

                technicals[tf] = {
                    "sma_short": sma_short,
                    "sma_long": sma_long,
                    "ema_short": ema_short,
                    "ema_long": ema_long,
                    "rsi": rsi,
                    "resistance": resistance,
                    "support": support,
                    "trend": price_trend,
                    "momentum": momentum,
                    "current_price": closes[-1]
                }
            except Exception as e:
                print(f"⚠️ Error calculating technicals for {symbol} {tf}: {e}")
                continue
                
        return technicals

    def get_multi_tf_consensus(self, technicals: Dict) -> str:
        """Dapatkan konsensus dari multiple timeframe"""
        if not technicals:
            return "NEUTRAL"
            
        bullish = sum(1 for tf in technicals.values() if "BULLISH" in tf.get("trend", ""))
        bearish = sum(1 for tf in technicals.values() if "BEARISH" in tf.get("trend", ""))
        total = len(technicals)
        
        if bullish >= total * 0.6:
            return "BULLISH"
        elif bearish >= total * 0.6:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def calculate_sentiment_score(self, symbol_data: Dict) -> Tuple[float, List[str]]:
        """Hitung sentiment score berdasarkan berbagai faktor"""
        score = 0.0
        details = []

        # Funding Rate Analysis
        fund = symbol_data["funding"]
        if fund > self.config["FUNDING_STRONG_POSITIVE"]:
            score += 10 * self.config["WEIGHT_FUNDING"]
            details.append("💰 Funding sangat positif")
        elif fund > self.config["FUNDING_WEAK_POSITIVE"]:
            score += 5 * self.config["WEIGHT_FUNDING"]
            details.append("💰 Funding positif")
        elif fund < self.config["FUNDING_STRONG_NEGATIVE"]:
            score -= 10 * self.config["WEIGHT_FUNDING"]
            details.append("💰 Funding sangat negatif")
        elif fund < self.config["FUNDING_WEAK_NEGATIVE"]:
            score -= 5 * self.config["WEIGHT_FUNDING"]
            details.append("💰 Funding negatif")

        # Open Interest Analysis
        oi_change = symbol_data["oi_change"]
        if oi_change > self.config["OI_STRONG_INCREASE"]:
            score += 10 * self.config["WEIGHT_OI"]
            details.append("📈 OI meningkat kuat")
        elif oi_change > self.config["OI_WEAK_INCREASE"]:
            score += 5 * self.config["WEIGHT_OI"]
            details.append("📈 OI meningkat")
        elif oi_change < self.config["OI_STRONG_DECREASE"]:
            score -= 10 * self.config["WEIGHT_OI"]
            details.append("📉 OI menurun kuat")
        elif oi_change < self.config["OI_WEAK_DECREASE"]:
            score -= 5 * self.config["WEIGHT_OI"]
            details.append("📉 OI menurun")

        # Taker Ratio Analysis
        taker = symbol_data["taker_ratio"]
        if taker > self.config["TAKER_STRONG_BUY"]:
            score += 10 * self.config["WEIGHT_TAKER"]
            details.append("⚔️ Buyer sangat agresif")
        elif taker > self.config["TAKER_WEAK_BUY"]:
            score += 5 * self.config["WEIGHT_TAKER"]
            details.append("⚔️ Buyer agresif")
        elif taker < self.config["TAKER_STRONG_SELL"]:
            score -= 10 * self.config["WEIGHT_TAKER"]
            details.append("⚔️ Seller sangat agresif")
        elif taker < self.config["TAKER_WEAK_SELL"]:
            score -= 5 * self.config["WEIGHT_TAKER"]
            details.append("⚔️ Seller agresif")

        # Long/Short Ratio Analysis
        long_ratio = symbol_data["long_ratio"]
        short_ratio = symbol_data["short_ratio"]
        ratio_diff = long_ratio - short_ratio
        
        if ratio_diff > self.config["LONGSHORT_STRONG_DIFFERENCE"]:
            score += 10 * self.config["WEIGHT_LONGSHORT"]
            details.append("👥 Long dominance kuat")
        elif ratio_diff > self.config["LONGSHORT_WEAK_DIFFERENCE"]:
            score += 5 * self.config["WEIGHT_LONGSHORT"]
            details.append("👥 Long dominance")
        elif ratio_diff < -self.config["LONGSHORT_STRONG_DIFFERENCE"]:
            score -= 10 * self.config["WEIGHT_LONGSHORT"]
            details.append("👥 Short dominance kuat")
        elif ratio_diff < -self.config["LONGSHORT_WEAK_DIFFERENCE"]:
            score -= 5 * self.config["WEIGHT_LONGSHORT"]
            details.append("👥 Short dominance")

        # Market Depth Analysis
        depth = symbol_data["depth"]
        if depth["pressure"] > self.config["DEPTH_STRONG_BID"]:
            score += 10 * self.config["WEIGHT_DEPTH"]
            details.append("🏊 Bid pressure kuat")
        elif depth["pressure"] > self.config["DEPTH_WEAK_BID"]:
            score += 5 * self.config["WEIGHT_DEPTH"]
            details.append("🏊 Bid pressure")
        elif depth["pressure"] < self.config["DEPTH_STRONG_ASK"]:
            score -= 10 * self.config["WEIGHT_DEPTH"]
            details.append("🏊 Ask pressure kuat")
        elif depth["pressure"] < self.config["DEPTH_WEAK_ASK"]:
            score -= 5 * self.config["WEIGHT_DEPTH"]
            details.append("🏊 Ask pressure")

        # Volume Analysis
        volume_analysis = symbol_data["volume_analysis"]
        if volume_analysis["volume_spike"]:
            if volume_analysis["volume_trend"] == "INCREASING":
                score += 10 * self.config["WEIGHT_VOLUME"]
                details.append("📊 Volume spike increasing")
            else:
                score += 5 * self.config["WEIGHT_VOLUME"]
                details.append("📊 Volume spike terdeteksi")

        normalized_score = np.clip(score / 10.0, -10, 10)
        
        return normalized_score, details

    def generate_entry_signals(self, symbol_data: Dict, technicals: Dict, volume_analysis: Dict) -> Dict:
        """Generate sinyal entry trading"""
        price_data = symbol_data["price_data"]
        current_price = price_data["price"]
        
        if current_price == 0:
            return {
                "action": "HOLD",
                "direction": "NEUTRAL",
                "confidence": 0,
                "entry_price": 0,
                "take_profit": [],
                "stop_loss": 0,
                "reason": ["Invalid price data"]
            }

        main_tf = technicals.get(self.config["MAIN_TIMEFRAME"], {})
        if not main_tf:
            return {
                "action": "HOLD",
                "direction": "NEUTRAL",
                "confidence": 0,
                "entry_price": current_price,
                "take_profit": [],
                "stop_loss": 0,
                "reason": ["Insufficient technical data"]
            }

        risk_adj = {"LOW": 1.5, "MEDIUM": 1.0, "HIGH": 0.7}
        adj = risk_adj[self.config["RISK_LEVEL"]]
        min_confidence = self.config["MIN_CONFIDENCE"] * adj
        entry_diff = self.config["ENTRY_SCORE_DIFFERENCE"] * adj

        bull_score = 0
        bear_score = 0
        reasons = []

        multi_tf = self.get_multi_tf_consensus(technicals)
        if multi_tf == "BULLISH":
            bull_score += 2
            reasons.append("✅ Multi-timeframe bullish")
        elif multi_tf == "BEARISH":
            bear_score += 2
            reasons.append("❌ Multi-timeframe bearish")

        trend = main_tf.get('trend', 'NEUTRAL')
        if 'BULLISH' in trend:
            bull_score += 3
            reasons.append(f"📈 Trend {trend}")
        elif 'BEARISH' in trend:
            bear_score += 3
            reasons.append(f"📉 Trend {trend}")

        rsi = main_tf.get('rsi', 50)
        if rsi < self.config["RSI_OVERSOLD"]:
            bull_score += 2
            reasons.append("🔻 RSI Oversold")
        elif rsi > self.config["RSI_OVERBOUGHT"]:
            bear_score += 2
            reasons.append("🔺 RSI Overbought")

        sma_long = main_tf.get('sma_long', current_price)
        if current_price > sma_long * self.config["TREND_WEAK_THRESHOLD"]:
            bull_score += 1
            reasons.append("💰 Price above SMA")
        elif current_price < sma_long * (2 - self.config["TREND_WEAK_THRESHOLD"]):
            bear_score += 1
            reasons.append("💰 Price below SMA")

        if volume_analysis.get('volume_spike'):
            if symbol_data["sentiment_score"] > 0:
                bull_score += 2
                reasons.append("📊 Volume Spike Bullish")
            else:
                bear_score += 2
                reasons.append("📊 Volume Spike Bearish")

        sentiment = symbol_data["sentiment_score"]
        if sentiment > 2:
            bull_score += 2
            reasons.append("😊 Strong Bullish Sentiment")
        elif sentiment < -2:
            bear_score += 2
            reasons.append("😰 Strong Bearish Sentiment")

        score_difference = bull_score - bear_score
        confidence = min(abs(score_difference) * 10, 100)

        support = main_tf.get('support', current_price * 0.98)
        resistance = main_tf.get('resistance', current_price * 1.02)

        signal = {
            "action": "HOLD",
            "direction": "NEUTRAL",
            "confidence": confidence,
            "entry_price": current_price,
            "take_profit": [],
            "stop_loss": 0,
            "reason": reasons
        }

        if score_difference >= entry_diff and confidence >= min_confidence:
            signal.update({
                "action": "LONG",
                "direction": "BULLISH",
                "stop_loss": support * (1 - self.config["STOP_LOSS_PERCENT"] / 100),
                "take_profit": [
                    current_price * (1 + self.config["TAKE_PROFIT_1_PERCENT"] / 100),
                    current_price * (1 + self.config["TAKE_PROFIT_2_PERCENT"] / 100)
                ]
            })
        elif score_difference <= -entry_diff and confidence >= min_confidence:
            signal.update({
                "action": "SHORT",
                "direction": "BEARISH",
                "stop_loss": resistance * (1 + self.config["STOP_LOSS_PERCENT"] / 100),
                "take_profit": [
                    current_price * (1 - self.config["TAKE_PROFIT_1_PERCENT"] / 100),
                    current_price * (1 - self.config["TAKE_PROFIT_2_PERCENT"] / 100)
                ]
            })

        return signal

    def get_risk_recommendation(self, symbol: str, score: float, price_data: Dict, entry_signal: Dict) -> str:
        """Dapatkan rekomendasi risk management"""
        risk_level = self.config["RISK_LEVEL"]
        ps = self.config["POSITION_SIZING"][risk_level]

        base = ""
        entry_details = ""

        if entry_signal["action"] != "HOLD" and entry_signal["confidence"] >= self.config["MIN_CONFIDENCE"]:
            emoji = "🟢" if entry_signal["action"] == "LONG" else "🔴"
            entry_details = (
                f"\n🎯 <b>ENTRY SIGNAL: {entry_signal['action']} ({entry_signal['confidence']:.0f}% confidence)</b>\n"
                f"💰 Entry: <code>${entry_signal['entry_price']:,.2f}</code>\n"
                f"🛡️ Stop Loss: <code>${entry_signal['stop_loss']:,.2f}</code>\n"
                f"🎯 Take Profit: <code>${entry_signal['take_profit'][0]:,.2f}</code> | <code>${entry_signal['take_profit'][1]:,.2f}</code>\n"
            )
            if entry_signal["reason"]:
                entry_details += "📋 Reasons:\n• " + "\n• ".join(entry_signal["reason"][:3]) + "\n"

        if score >= self.config["SCORE_THRESHOLD_STRONG"]:
            base = "🟢 STRONG BULLISH - Konfirmasi bullish kuat"
            return f"{base}\n💡 Risk: Position {ps['MIN']*100:.0f}-{ps['MAX']*100:.0f}%, TP {ps['TP1']*100:.0f}-{ps['TP2']*100:.0f}%, SL {ps['SL']*100:.0f}%{entry_details}"
        elif score >= self.config["SCORE_THRESHOLD_MEDIUM"]:
            base = "🟡 MILD BULLISH - Signal bullish moderat"
            return f"{base}\n💡 Risk: Position {ps['MIN']*100:.0f}-{ps['MAX']*100*0.7:.0f}%, TP {ps['TP1']*100:.0f}%, SL {ps['SL']*100:.0f}%{entry_details}"
        elif score >= self.config["SCORE_THRESHOLD_WEAK"]:
            base = "⚪️ NEUTRAL BULLISH - Sedikit bias bullish"
            return f"{base}\n💡 Risk: Wait confirmation atau position {ps['MIN']*100*0.5:.0f}-{ps['MIN']*100:.0f}%{entry_details}"
        elif score <= -self.config["SCORE_THRESHOLD_STRONG"]:
            base = "🔴 STRONG BEARISH - Konfirmasi bearish kuat"
            return f"{base}\n💡 Risk: Position {ps['MIN']*100:.0f}-{ps['MAX']*100:.0f}%, TP {ps['TP1']*100:.0f}-{ps['TP2']*100:.0f}%, SL {ps['SL']*100:.0f}%{entry_details}"
        elif score <= -self.config["SCORE_THRESHOLD_MEDIUM"]:
            base = "🟠 MILD BEARISH - Signal bearish moderat"
            return f"{base}\n💡 Risk: Position {ps['MIN']*100:.0f}-{ps['MAX']*100*0.7:.0f}%, TP {ps['TP1']*100:.0f}%, SL {ps['SL']*100:.0f}%{entry_details}"
        elif score <= -self.config["SCORE_THRESHOLD_WEAK"]:
            base = "⚪️ NEUTRAL BEARISH - Sedikit bias bearish"
            return f"{base}\n💡 Risk: Wait confirmation atau position {ps['MIN']*100*0.5:.0f}-{ps['MIN']*100:.0f}%{entry_details}"
        else:
            return f"⚖️ NEUTRAL - Market sideways{entry_details}"

    def analyze_symbol(self, symbol: str):
        """Analisis satu simbol trading"""
        try:
            print(f"📊 Analyzing {symbol}...")

            funding = self.get_funding_rate(symbol)
            times, oi = self.get_open_interest(symbol)
            taker_ratio = self.get_taker_ratio(symbol)
            long_ratio, short_ratio = self.get_long_short_ratio(symbol)
            price_data = self.get_price_data(symbol)

            if price_data["price"] == 0:
                print(f"⚠️ [{symbol}] Invalid price data, skipping...")
                return

            if price_data["quote_volume"] < self.config["MIN_VOLUME"]:
                print(f"⚠️ [{symbol}] Volume too low ({price_data['quote_volume']:,.0f}), skipping...")
                return

            depth = self.get_market_depth(symbol)
            volume_analysis = self.calculate_volume_analysis(symbol, price_data["quote_volume"])
            technicals = self.calculate_advanced_technicals(symbol)

            oi_change = 0
            if len(oi) >= 2 and oi[-2] > 0:
                oi_change = ((oi[-1] - oi[-2]) / oi[-2]) * 100

            symbol_data = {
                "funding": funding,
                "oi_change": oi_change,
                "taker_ratio": taker_ratio,
                "long_ratio": long_ratio,
                "short_ratio": short_ratio,
                "depth": depth,
                "volume_analysis": volume_analysis,
                "price_data": price_data
            }

            score, score_details = self.calculate_sentiment_score(symbol_data)
            symbol_data["sentiment_score"] = score

            entry_signal = self.generate_entry_signals(symbol_data, technicals, volume_analysis)
            recommendation = self.get_risk_recommendation(symbol, score, price_data, entry_signal)

            local_time = self.get_local_time()

            message = (
                f"<b>🎯 {symbol} FUTURES ANALYSIS</b>\n"
                f"⏰ Waktu: {local_time}\n\n"

                f"<b>📈 PRICE INFO:</b>\n"
                f"💰 Price: <code>${price_data['price']:,.2f}</code>\n"
                f"📊 24h Change: <code>{price_data['change_24h']:+.2f}%</code>\n"
                f"⬆️ High 24h: <code>${price_data['high_24h']:,.2f}</code>\n"
                f"⬇️ Low 24h: <code>${price_data['low_24h']:,.2f}</code>\n"
                f"💎 Volume: <code>{price_data['quote_volume']:,.0f} USDT</code>\n\n"

                f"<b>📊 MARKET SENTIMENT:</b>\n"
                f"💸 Funding: <code>{funding*100:+.4f}%</code>\n"
                f"📈 OI Change: <code>{oi_change:+.2f}%</code>\n"
                f"⚔️ Taker Ratio: <code>{taker_ratio:.2f}</code>\n"
                f"👥 Long/Short: <code>{long_ratio:.2f}</code>/<code>{short_ratio:.2f}</code>\n"
                f"🏊 Bid Pressure: <code>{depth['pressure']:.2f}</code>\n\n"
            )

            if volume_analysis and volume_analysis.get('volume_ratio', 1) != 1:
                vol_emoji = "🚀" if volume_analysis["volume_spike"] else "📊"
                message += (
                    f"<b>📊 VOLUME ANALYSIS:</b>\n"
                    f"{vol_emoji} Volume Ratio: <code>{volume_analysis['volume_ratio']:.2f}x</code>\n"
                    f"📈 Volume Trend: <code>{volume_analysis['volume_trend']}</code>\n\n"
                )

            main_tf = self.config["MAIN_TIMEFRAME"]
            if technicals and main_tf in technicals:
                tf = technicals[main_tf]
                rsi_val = tf.get('rsi', 50)
                rsi_emoji = "🔴" if rsi_val > self.config["RSI_OVERBOUGHT"] else "🟢" if rsi_val < self.config["RSI_OVERSOLD"] else "🟡"
                message += (
                    f"<b>🔧 TECHNICALS ({main_tf.upper()}):</b>\n"
                    f"📊 Trend: <code>{tf.get('trend', 'N/A')}</code>\n"
                    f"{rsi_emoji} RSI: <code>{rsi_val:.1f}</code>\n"
                    f"📈 Momentum: <code>{tf.get('momentum', 0):+.2f}%</code>\n"
                    f"🛡️ Support: <code>${tf.get('support', 0):,.2f}</code>\n"
                    f"🎯 Resistance: <code>${tf.get('resistance', 0):,.2f}</code>\n\n"
                )

            message += f"<b>🎲 SENTIMENT SCORE: <code>{score:.1f}/10</code></b>\n"
            if score_details:
                message += "📋 Key Factors:\n• " + "\n• ".join(score_details[:4]) + "\n\n"

            message += f"<b>💡 RECOMMENDATION:</b>\n{recommendation}\n\n"
            message += f"<i>Risk Level: {self.config['RISK_LEVEL']} | Update every {self.config['INTERVAL']//60} minutes</i>"

            print(f"✅ [{symbol}] Analysis completed - Score: {score:.1f}")
            self.send_telegram(message, symbol)

            self.analysis_history[symbol] = {
                "timestamp": time.time(),
                "score": score,
                "price": price_data["price"],
                "entry_signal": entry_signal
            }

        except Exception as e:
            print(f"❌ [{symbol}] Analysis error: {e}")

    def start_flash_crash_monitoring(self):
        """Memulai monitoring flash crash"""
        self.crash_detector.start_realtime_monitoring()

    def analyze_all_symbols(self):
        """Analisis semua simbol"""
        print(f"\n🔄 Starting analysis for {len(self.config['SYMBOLS'])} symbols...")
        print("=" * 60)
        successful = 0
        for symbol in self.config["SYMBOLS"]:
            try:
                self.analyze_symbol(symbol)
                successful += 1
            except Exception as e:
                print(f"❌ Failed to analyze {symbol}: {e}")
            time.sleep(3)
        print("=" * 60)
        print(f"✅ {successful}/{len(self.config['SYMBOLS'])} analyses completed")
        print(f"⏳ Waiting {self.config['INTERVAL']//60} minutes...")

    def run(self):
        """Jalankan analyzer utama"""
        print("🚀 ENHANCED FUTURES ANALYZER STARTED")
        print(f"📍 Timezone: {self.config['TIMEZONE']}")
        print(f"📊 Symbols: {', '.join(self.config['SYMBOLS'])}")
        print(f"⏰ Interval: {self.config['INTERVAL']//60} minutes")
        print(f"🎯 Risk Level: {self.config['RISK_LEVEL']}")
        print(f"📈 Main Timeframe: {self.config['MAIN_TIMEFRAME']}")
        print("=" * 60)

        self.start_flash_crash_monitoring()

        while True:
            try:
                self.analyze_all_symbols()
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user")
                self.crash_detector.monitoring_active = False
                break
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(60)
            time.sleep(self.config["INTERVAL"])


def main():
    """Main function"""
    analyzer = EnhancedFuturesAnalyzer(CONFIG)
    analyzer.run()


if __name__ == "__main__":
    main()
