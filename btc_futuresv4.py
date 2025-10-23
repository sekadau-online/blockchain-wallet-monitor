import requests, datetime, time, json, os, numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

# === KONFIGURASI ===
CONFIG = {
    "TELEGRAM_BOT_TOKEN": "8345066310:AAElrMezSmJZwWOWWRYLFMf2z5nyDCkTg0g",
    "TELEGRAM_CHAT_ID": "-4848345455",
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "INTERVAL": 60 * 5,  # 5 menit untuk avoid rate limit
    "TIMEZONE": "Asia/Pontianak",  # UTC+7
    "RISK_LEVEL": "MEDIUM",  # LOW, MEDIUM, HIGH
    "VOLUME_THRESHOLD": 1.5,  # Threshold volume spike
    "MIN_VOLUME": 1000000,  # Minimum volume dalam USDT
    "MAX_RETRIES": 3,
    "REQUEST_DELAY": 1,  # Delay antara requests
}

BASE = "https://fapi.binance.com"

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
        
        # Initialize history for each symbol
        for symbol in config["SYMBOLS"]:
            self.volume_history[symbol] = deque(maxlen=20)
            self.price_history[symbol] = deque(maxlen=50)

    def get_pontianak_time(self):
        """Waktu Pontianak (UTC+7)"""
        utc_time = datetime.datetime.now(datetime.timezone.utc)
        pontianak_time = utc_time + datetime.timedelta(hours=7)
        return pontianak_time.strftime("%Y-%m-%d %H:%M:%S")

    def safe_get(self, url: str, max_retries: int = None):
        """Enhanced safe_get dengan better error handling"""
        if max_retries is None:
            max_retries = self.config["MAX_RETRIES"]
            
        for attempt in range(max_retries):
            try:
                time.sleep(self.config["REQUEST_DELAY"])  # Jeda antara requests
                response = self.session.get(url, timeout=20)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) + 10  # Wait longer for rate limit
                    print(f"⚠️ Rate limit detected, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 418:  # IP banned
                    wait_time = 300  # Wait 5 minutes
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
                
        raise Exception(f"Failed to fetch data from {url} after {max_retries} attempts")

    def send_telegram(self, message: str, symbol: str = "GLOBAL"):
        """Kirim pesan ke Telegram dengan formatting yang lebih baik"""
        try:
            url = f"https://api.telegram.org/bot{self.config['TELEGRAM_BOT_TOKEN']}/sendMessage"
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
        """Dapatkan funding rate dengan fallback"""
        try:
            url = f"{BASE}/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            data = self.safe_get(url)
            return float(data[0]["fundingRate"])
        except:
            return 0.0

    def get_open_interest(self, symbol: str) -> tuple:
        """Dapatkan open interest data"""
        try:
            url = f"{BASE}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=10"
            data = self.safe_get(url)
            if not data:
                return [], [0]
            times = [datetime.datetime.fromtimestamp(x["timestamp"]/1000) for x in data]
            values = [float(x["sumOpenInterest"]) for x in data]
            return times, values
        except:
            return [], [0]

    def get_taker_ratio(self, symbol: str) -> float:
        """Dapatkan taker buy/sell ratio"""
        try:
            url = f"{BASE}/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=5"
            data = self.safe_get(url)
            return float(data[0]["buySellRatio"]) if data else 1.0
        except:
            return 1.0

    def get_long_short_ratio(self, symbol: str) -> tuple:
        """Dapatkan long/short ratio"""
        try:
            url = f"{BASE}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=5"
            data = self.safe_get(url)
            if data:
                long_ratio = float(data[0]["longAccount"])
                short_ratio = float(data[0]["shortAccount"])
                return long_ratio, short_ratio
        except:
            pass
        return 0.5, 0.5  # Default netral

    def get_price_data(self, symbol: str) -> Dict:
        """Dapatkan data harga 24h"""
        try:
            url = f"{BASE}/fapi/v1/ticker/24hr?symbol={symbol}"
            data = self.safe_get(url)
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
            # Return default data
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
        """Dapatkan data market depth (order book) dengan fallback"""
        try:
            url = f"{BASE}/fapi/v1/depth?symbol={symbol}&limit=10"  # Reduced limit
            data = self.safe_get(url, max_retries=2)  # Fewer retries for depth
            
            bids = sum(float(bid[1]) * float(bid[0]) for bid in data["bids"][:5])  # Only 5 levels
            asks = sum(float(ask[1]) * float(ask[0]) for ask in data["asks"][:5])
            total_bids = sum(float(bid[1]) for bid in data["bids"][:5])
            total_asks = sum(float(ask[1]) for ask in data["asks"][:5])
            
            return {
                "bids_value": bids,
                "asks_value": asks, 
                "bids_volume": total_bids,
                "asks_volume": total_asks,
                "pressure": bids/asks if asks > 0 else 1,
                "volume_pressure": total_bids/total_asks if total_asks > 0 else 1
            }
        except Exception as e:
            print(f"⚠️ Depth data unavailable for {symbol}, using defaults")
            return {
                "bids_value": 1,
                "asks_value": 1, 
                "bids_volume": 1,
                "asks_volume": 1,
                "pressure": 1,
                "volume_pressure": 1
            }

    def calculate_volume_analysis(self, symbol: str, current_volume: float) -> Dict:
        """Analisis volume dengan historical comparison"""
        volume_history = self.volume_history[symbol]
        
        # Tambahkan volume saat ini ke history
        volume_history.append(current_volume)
        
        if len(volume_history) < 3:  # Reduced minimum data requirement
            return {"volume_spike": False, "volume_trend": "INSUFFICIENT_DATA", "volume_ratio": 1.0}
        
        # Hitung volume average
        avg_volume = np.mean(list(volume_history))
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Deteksi volume spike
        volume_spike = volume_ratio > self.config["VOLUME_THRESHOLD"]
        
        # Tentukan trend volume
        if len(volume_history) >= 6:  # Reduced requirement
            recent_volumes = list(volume_history)[-3:]
            older_volumes = list(volume_history)[-6:-3]
            recent_avg = np.mean(recent_volumes)
            older_avg = np.mean(older_volumes)
            
            if recent_avg > older_avg * 1.15:  # Reduced threshold
                volume_trend = "INCREASING"
            elif recent_avg < older_avg * 0.85:
                volume_trend = "DECREASING"
            else:
                volume_trend = "STABLE"
        else:
            volume_trend = "STABLE"
            
        return {
            "volume_spike": volume_spike,
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "avg_volume": avg_volume,
            "current_volume": current_volume
        }

    def calculate_advanced_technicals(self, symbol: str) -> Dict:
        """Hitung indikator teknikal yang lebih advanced dengan fallback"""
        try:
            # Gunakan timeframe yang lebih sedikit untuk mengurangi requests
            timeframes = ['15m']  # Hanya 15m untuk mengurangi load
            
            technicals = {}
            
            for tf in timeframes:
                url = f"{BASE}/fapi/v1/klines?symbol={symbol}&interval={tf}&limit=30"  # Reduced limit
                data = self.safe_get(url, max_retries=2)
                
                if not data or len(data) < 20:
                    continue
                    
                closes = np.array([float(c[4]) for c in data])
                highs = np.array([float(h[2]) for h in data])
                lows = np.array([float(l[3]) for l in data])
                volumes = np.array([float(v[5]) for v in data])
                
                # Simple Moving Averages
                sma_20 = np.mean(closes[-20:])
                sma_10 = np.mean(closes[-10:])
                
                # Exponential Moving Average (approximated)
                ema_12 = self.calculate_ema(closes, 12)
                ema_26 = self.calculate_ema(closes, 26)
                
                # RSI
                rsi = self.calculate_rsi(closes)
                
                # Support Resistance sederhana
                resistance = np.max(highs[-10:])
                support = np.min(lows[-10:])
                
                # Trend sederhana
                price_trend = "BULLISH" if closes[-1] > sma_20 else "BEARISH"
                if ema_12 > ema_26 and closes[-1] > sma_20:
                    price_trend = "STRONG_BULLISH"
                elif ema_12 < ema_26 and closes[-1] < sma_20:
                    price_trend = "STRONG_BEARISH"
                
                technicals[tf] = {
                    "sma_10": sma_10,
                    "sma_20": sma_20,
                    "ema_12": ema_12,
                    "ema_26": ema_26,
                    "rsi": rsi,
                    "resistance": resistance,
                    "support": support,
                    "trend": price_trend,
                    "momentum": ((closes[-1] - closes[-5]) / closes[-5]) * 100
                }
                
            return technicals
            
        except Exception as e:
            print(f"⚠️ Technical indicator error for {symbol}: {e}")
            return {}

    def calculate_ema(self, prices: np.array, period: int) -> float:
        """Hitung Exponential Moving Average"""
        if len(prices) < period:
            return float(prices[-1])
        
        try:
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            
            return np.convolve(prices[-period:], weights, mode='valid')[-1]
        except:
            return float(prices[-1])

    def calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Hitung Relative Strength Index"""
        if len(prices) < period + 1:
            return 50
            
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.mean(gains[-period:])
            avg_losses = np.mean(losses[-period:])
            
            if avg_losses == 0:
                return 100
                
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return 50

    def generate_entry_signals(self, symbol_data: Dict, technicals: Dict, volume_analysis: Dict) -> Dict:
        """Generate sinyal entry long/short yang spesifik"""
        price_data = symbol_data["price_data"]
        current_price = price_data["price"]
        
        if current_price == 0:  # Invalid price data
            return {
                "action": "HOLD",
                "direction": "NEUTRAL", 
                "confidence": 0,
                "entry_price": 0,
                "take_profit": [],
                "stop_loss": 0,
                "reason": ["Invalid price data"]
            }
        
        # Default signal
        signal = {
            "action": "HOLD",
            "direction": "NEUTRAL", 
            "confidence": 0,
            "entry_price": current_price,
            "take_profit": [],
            "stop_loss": 0,
            "reason": []
        }
        
        # Analisis technical
        tf_15m = technicals.get('15m', {})
        if not tf_15m:
            signal["reason"].append("Insufficient technical data")
            return signal
        
        # Skor bullish/bearish
        bull_score = 0
        bear_score = 0
        reasons = []
        
        # 1. Trend Analysis
        trend = tf_15m.get('trend', 'NEUTRAL')
        if 'BULLISH' in trend:
            bull_score += 3
            reasons.append("📈 Trend Bullish")
        elif 'BEARISH' in trend:
            bear_score += 3
            reasons.append("📉 Trend Bearish")
            
        # 2. RSI Analysis
        rsi = tf_15m.get('rsi', 50)
        if rsi < 35:
            bull_score += 2
            reasons.append("🔻 RSI Oversold")
        elif rsi > 65:
            bear_score += 2
            reasons.append("🔺 RSI Overbought")
            
        # 3. Price vs Moving Averages
        sma_20 = tf_15m.get('sma_20', current_price)
        if current_price > sma_20 * 1.01:
            bull_score += 1
            reasons.append("💰 Price above SMA20")
        elif current_price < sma_20 * 0.99:
            bear_score += 1
            reasons.append("💰 Price below SMA20")
            
        # 4. Volume Analysis
        if volume_analysis.get('volume_spike'):
            if symbol_data["sentiment_score"] > 0:
                bull_score += 2
                reasons.append("📊 Volume Spike Bullish")
            else:
                bear_score += 2
                reasons.append("📊 Volume Spike Bearish")
                
        # 5. Market Sentiment
        sentiment = symbol_data["sentiment_score"]
        if sentiment > 2:
            bull_score += 2
            reasons.append("😊 Strong Bullish Sentiment")
        elif sentiment < -2:
            bear_score += 2
            reasons.append("😰 Strong Bearish Sentiment")
            
        # Determine final signal
        score_difference = bull_score - bear_score
        confidence = min(abs(score_difference) * 10, 100)  # Convert to percentage
        
        support = tf_15m.get('support', current_price * 0.98)
        resistance = tf_15m.get('resistance', current_price * 1.02)
        
        if score_difference >= 3 and confidence > 50:
            signal["action"] = "LONG"
            signal["direction"] = "BULLISH"
            signal["confidence"] = confidence
            signal["entry_price"] = current_price
            signal["stop_loss"] = support * 0.995
            signal["take_profit"] = [
                resistance * 0.99,
                resistance * 1.02
            ]
            signal["reason"] = reasons
            
        elif score_difference <= -3 and confidence > 50:
            signal["action"] = "SHORT" 
            signal["direction"] = "BEARISH"
            signal["confidence"] = confidence
            signal["entry_price"] = current_price
            signal["stop_loss"] = resistance * 1.005
            signal["take_profit"] = [
                support * 1.01,
                support * 0.98
            ]
            signal["reason"] = reasons
            
        return signal

    def calculate_sentiment_score(self, symbol_data: Dict) -> tuple:
        """Hitung skor sentimen dengan weighting yang lebih sophisticated"""
        score = 0
        details = []

        # Funding Rate (Weight: 25%)
        fund = symbol_data["funding"]
        if fund > 0.001:
            score += 2.5
            details.append("💰 Funding sangat positif")
        elif fund > 0.0001:
            score += 1.25
            details.append("💰 Funding positif")
        elif fund < -0.001:
            score -= 2.5
            details.append("💰 Funding sangat negatif")
        elif fund < -0.0001:
            score -= 1.25
            details.append("💰 Funding negatif")

        # Open Interest (Weight: 20%)
        oi_change = symbol_data["oi_change"]
        if oi_change > 2:
            score += 2.0
            details.append("📈 OI meningkat kuat")
        elif oi_change > 0.5:
            score += 1.0
            details.append("📈 OI meningkat")
        elif oi_change < -2:
            score -= 2.0
            details.append("📉 OI menurun kuat")
        elif oi_change < -0.5:
            score -= 1.0
            details.append("📉 OI menurun")

        # Taker Ratio (Weight: 20%)
        taker = symbol_data["taker_ratio"]
        if taker > 1.2:
            score += 2.0
            details.append("⚔️ Buyer sangat agresif")
        elif taker > 1.05:
            score += 1.0
            details.append("⚔️ Buyer agresif")
        elif taker < 0.8:
            score -= 2.0
            details.append("⚔️ Seller sangat agresif")
        elif taker < 0.95:
            score -= 1.0
            details.append("⚔️ Seller agresif")

        # Long/Short Ratio (Weight: 15%)
        long_ratio = symbol_data["long_ratio"]
        short_ratio = symbol_data["short_ratio"]
        if long_ratio > short_ratio + 0.15:
            score += 1.5
            details.append("👥 Long dominance kuat")
        elif long_ratio > short_ratio + 0.05:
            score += 0.75
            details.append("👥 Long dominance")
        elif short_ratio > long_ratio + 0.15:
            score -= 1.5
            details.append("👥 Short dominance kuat")
        elif short_ratio > long_ratio + 0.05:
            score -= 0.75
            details.append("👥 Short dominance")

        # Market Depth (Weight: 10%)
        depth = symbol_data["depth"]
        if depth["pressure"] > 1.2:
            score += 1.0
            details.append("🏊 Bid pressure kuat")
        elif depth["pressure"] > 1.05:
            score += 0.5
            details.append("🏊 Bid pressure")
        elif depth["pressure"] < 0.8:
            score -= 1.0
            details.append("🏊 Ask pressure kuat")
        elif depth["pressure"] < 0.95:
            score -= 0.5
            details.append("🏊 Ask pressure")

        # Volume Analysis (Weight: 10%)
        volume_analysis = symbol_data["volume_analysis"]
        if volume_analysis["volume_spike"]:
            if volume_analysis["volume_trend"] == "INCREASING":
                score += 1.0
                details.append("📊 Volume spike increasing")
            else:
                score += 0.5
                details.append("📊 Volume spike terdeteksi")

        return score, details

    def get_risk_recommendation(self, symbol: str, score: float, price_data: Dict, entry_signal: Dict) -> str:
        """Beri rekomendasi risk management berdasarkan skor dan sinyal entry"""
        risk_level = self.config["RISK_LEVEL"]
        price = price_data["price"]

        base_recommendation = ""
        entry_details = ""

        # Tambahkan detail entry signal jika ada
        if entry_signal["action"] != "HOLD" and entry_signal["confidence"] > 50:
            action_emoji = "🟢" if entry_signal["action"] == "LONG" else "🔴"
            entry_details = (
                f"\n🎯 <b>ENTRY SIGNAL: {entry_signal['action']} ({entry_signal['confidence']:.0f}% confidence)</b>\n"
                f"💰 Entry: <code>${entry_signal['entry_price']:,.2f}</code>\n"
                f"🛡️ Stop Loss: <code>${entry_signal['stop_loss']:,.2f}</code>\n"
                f"🎯 Take Profit: <code>${entry_signal['take_profit'][0]:,.2f}</code> | <code>${entry_signal['take_profit'][1]:,.2f}</code>\n"
            )
            
            # Tambahkan alasan
            if entry_signal["reason"]:
                entry_details += "📋 Reasons:\n• " + "\n• ".join(entry_signal["reason"][:3]) + "\n"

        if score >= 6:
            base_recommendation = "🟢 STRONG BULLISH - Konfirmasi bullish kuat"
            if risk_level == "LOW":
                return f"{base_recommendation}\n💡 Risk: Position 1-2%, TP 3-5%, SL 2%{entry_details}"
            elif risk_level == "MEDIUM":
                return f"{base_recommendation}\n💡 Risk: Position 3-5%, TP 5-8%, SL 3%{entry_details}"
            else:
                return f"{base_recommendation}\n💡 Risk: Position 5-7%, TP 8-12%, SL 4%{entry_details}"

        elif score >= 3:
            base_recommendation = "🟡 MILD BULLISH - Signal bullish moderat"
            return f"{base_recommendation}\n💡 Risk: Position 1-3%, TP 3-6%, SL 2.5%{entry_details}"

        elif score >= 1:
            base_recommendation = "⚪️ NEUTRAL BULLISH - Sedikit bias bullish"
            return f"{base_recommendation}\n💡 Risk: Wait confirmation atau position 0.5-1%{entry_details}"

        elif score <= -6:
            base_recommendation = "🔴 STRONG BEARISH - Konfirmasi bearish kuat"
            if risk_level == "LOW":
                return f"{base_recommendation}\n💡 Risk: Position 1-2%, TP 3-5%, SL 2%{entry_details}"
            elif risk_level == "MEDIUM":
                return f"{base_recommendation}\n💡 Risk: Position 3-5%, TP 5-8%, SL 3%{entry_details}"
            else:
                return f"{base_recommendation}\n💡 Risk: Position 5-7%, TP 8-12%, SL 4%{entry_details}"

        elif score <= -3:
            base_recommendation = "🟠 MILD BEARISH - Signal bearish moderat"
            return f"{base_recommendation}\n💡 Risk: Position 1-3%, TP 3-6%, SL 2.5%{entry_details}"

        elif score <= -1:
            base_recommendation = "⚪️ NEUTRAL BEARISH - Sedikit bias bearish"
            return f"{base_recommendation}\n💡 Risk: Wait confirmation atau position 0.5-1%{entry_details}"
        else:
            return f"⚖️ NEUTRAL - Market sideways{entry_details}"

    def analyze_symbol(self, symbol: str):
        """Analisis untuk satu simbol dengan comprehensive error handling"""
        try:
            print(f"📊 Analyzing {symbol}...")

            # Collect semua data dengan error handling
            funding = self.get_funding_rate(symbol)
            times, oi = self.get_open_interest(symbol)
            taker_ratio = self.get_taker_ratio(symbol)
            long_ratio, short_ratio = self.get_long_short_ratio(symbol)
            price_data = self.get_price_data(symbol)
            
            # Skip jika price data invalid
            if price_data["price"] == 0:
                print(f"⚠️ [{symbol}] Invalid price data, skipping...")
                return
                
            depth = self.get_market_depth(symbol)
            
            # Analisis volume
            volume_analysis = self.calculate_volume_analysis(symbol, price_data["quote_volume"])
            
            # Technical analysis advanced
            technicals = self.calculate_advanced_technicals(symbol)

            # Hitung OI change
            oi_change = 0
            if len(oi) >= 2:
                oi_change = ((oi[-1] - oi[-2]) / oi[-2]) * 100

            # Prepare data untuk scoring
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

            # Hitung skor sentimen
            score, score_details = self.calculate_sentiment_score(symbol_data)
            symbol_data["sentiment_score"] = score

            # Generate entry signals
            entry_signal = self.generate_entry_signals(symbol_data, technicals, volume_analysis)

            # Dapatkan rekomendasi
            recommendation = self.get_risk_recommendation(symbol, score, price_data, entry_signal)

            # Format message
            pontianak_time = self.get_pontianak_time()

            message = (
                f"<b>🎯 {symbol} FUTURES ANALYSIS</b>\n"
                f"⏰ Waktu: {pontianak_time} (Pontianak)\n\n"

                f"<b>📈 PRICE INFO:</b>\n"
                f"💰 Price: <code>${price_data['price']:,.2f}</code>\n"
                f"📊 24h Change: <code>{price_data['change_24h']:+.2f}%</code>\n"
                f"⬆️ High 24h: <code>${price_data['high_24h']:,.2f}</code>\n"
                f"⬇️ Low 24h: <code>${price_data['low_24h']:,.2f}</code>\n"
                f"💎 Volume: <code>{price_data['volume']:,.0f}</code>\n\n"

                f"<b>📊 MARKET SENTIMENT:</b>\n"
                f"💸 Funding: <code>{funding*100:+.4f}%</code>\n"
                f"📈 OI Change: <code>{oi_change:+.2f}%</code>\n"
                f"⚔️ Taker Ratio: <code>{taker_ratio:.2f}</code>\n"
                f"👥 Long/Short: <code>{long_ratio:.2f}</code>/<code>{short_ratio:.2f}</code>\n"
                f"🏊 Bid Pressure: <code>{depth['pressure']:.2f}</code>\n\n"
            )

            # Tambahkan volume analysis
            if volume_analysis and volume_analysis.get('volume_ratio', 1) != 1:
                volume_emoji = "🚀" if volume_analysis["volume_spike"] else "📊"
                message += (
                    f"<b>📊 VOLUME ANALYSIS:</b>\n"
                    f"{volume_emoji} Volume Ratio: <code>{volume_analysis['volume_ratio']:.2f}x</code>\n"
                    f"📈 Volume Trend: <code>{volume_analysis['volume_trend']}</code>\n\n"
                )

            # Tambahkan technical indicators jika ada
            if technicals and '15m' in technicals:
                tf = technicals['15m']
                rsi_emoji = "🔴" if tf.get('rsi', 50) > 70 else "🟢" if tf.get('rsi', 50) < 30 else "🟡"
                message += (
                    f"<b>🔧 TECHNICALS (15M):</b>\n"
                    f"📊 Trend: <code>{tf.get('trend', 'N/A')}</code>\n"
                    f"{rsi_emoji} RSI: <code>{tf.get('rsi', 0):.1f}</code>\n"
                    f"🛡️ Support: <code>${tf.get('support', 0):,.2f}</code>\n"
                    f"🎯 Resistance: <code>${tf.get('resistance', 0):,.2f}</code>\n\n"
                )

            # Tambahkan scoring details
            message += f"<b>🎲 SENTIMENT SCORE: <code>{score:.1f}/10</code></b>\n"
            if score_details:
                message += "📋 Key Factors:\n• " + "\n• ".join(score_details[:4]) + "\n\n"

            # Tambahkan rekomendasi
            message += f"<b>💡 RECOMMENDATION:</b>\n{recommendation}\n\n"

            message += f"<i>Risk Level: {self.config['RISK_LEVEL']} | Update setiap {self.config['INTERVAL']//60} menit</i>"

            print(f"✅ [{symbol}] Analysis completed - Score: {score:.1f}")
            self.send_telegram(message, symbol)

            # Simpan ke history
            self.analysis_history[symbol] = {
                "timestamp": pontianak_time,
                "score": score,
                "price": price_data["price"],
                "entry_signal": entry_signal
            }

        except Exception as e:
            error_msg = f"❌ [{symbol}] Error: {str(e)}"
            print(error_msg)
            # Jangan kirim error ke Telegram untuk avoid spam
            # self.send_telegram(f"❌ <b>Error analyzing {symbol}:</b>\n{str(e)}", symbol)

    def analyze_all_symbols(self):
        """Analisis semua simbol dalam konfigurasi"""
        print(f"\n🔄 Memulai analisis {len(self.config['SYMBOLS'])} simbol...")
        print("=" * 50)

        successful_analysis = 0
        for symbol in self.config["SYMBOLS"]:
            try:
                self.analyze_symbol(symbol)
                successful_analysis += 1
            except Exception as e:
                print(f"❌ Failed to analyze {symbol}: {e}")
            
            # Jeda lebih panjang antara simbol
            time.sleep(3)

        print("=" * 50)
        print(f"✅ {successful_analysis}/{len(self.config['SYMBOLS'])} analisis berhasil")
        print(f"⏳ Menunggu {self.config['INTERVAL']//60} menit...")

    def run(self):
        """Jalankan monitor secara kontinu"""
        print("🚀 ENHANCED FUTURES ANALYZER DIMULAI")
        print(f"📍 Timezone: {self.config['TIMEZONE']}")
        print(f"📊 Symbols: {', '.join(self.config['SYMBOLS'])}")
        print(f"⏰ Interval: {self.config['INTERVAL']//60} menit")
        print(f"🎯 Risk Level: {self.config['RISK_LEVEL']}")
        print(f"🛡️ Max Retries: {self.config['MAX_RETRIES']}")
        print("=" * 50)

        while True:
            try:
                self.analyze_all_symbols()
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                # Wait longer if there's a major error
                time.sleep(60)

            wait_minutes = self.config["INTERVAL"] // 60
            print(f"⏳ Menunggu {wait_minutes} menit hingga analisis berikutnya...")
            time.sleep(self.config["INTERVAL"])

def main():
    # Jalankan analyzer
    analyzer = EnhancedFuturesAnalyzer(CONFIG)
    analyzer.run()

if __name__ == "__main__":
    main()
