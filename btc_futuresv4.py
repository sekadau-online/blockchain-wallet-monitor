import requests, datetime, time, json, os, numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

# === KONFIGURASI ===
CONFIG = {
    "TELEGRAM_BOT_TOKEN": "8345066310:AAElrMezSmJZwWOWWRYLFMf2z5nyDCkTg0g",
    "TELEGRAM_CHAT_ID": "-4848345455",
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "INTERVAL": 60 * 3,  # 3 menit
    "TIMEZONE": "Asia/Pontianak",  # UTC+7
    "RISK_LEVEL": "MEDIUM",  # LOW, MEDIUM, HIGH
    "VOLUME_THRESHOLD": 1.5,  # Threshold volume spike
    "MIN_VOLUME": 1000000,  # Minimum volume dalam USDT
}

BASE = "https://fapi.binance.com"

class EnhancedFuturesAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
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

    def safe_get(self, url: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) + 5
                    print(f"⚠️ Rate limit, waiting {wait_time}s...")
                    time.sleep(wait_time)
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                time.sleep(3)
        raise Exception(f"Failed to fetch data from {url}")

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
            response = self.session.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"✅ [{symbol}] Pesan terkirim ke Telegram")
            else:
                print(f"⚠️ [{symbol}] Gagal kirim: {response.status_code}")
        except Exception as e:
            print(f"⚠️ [{symbol}] Error Telegram: {e}")

    def get_funding_rate(self, symbol: str) -> float:
        url = f"{BASE}/fapi/v1/fundingRate?symbol={symbol}&limit=3"
        data = self.safe_get(url)
        return float(data[0]["fundingRate"])

    def get_open_interest(self, symbol: str) -> tuple:
        url = f"{BASE}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=30"
        data = self.safe_get(url)
        times = [datetime.datetime.fromtimestamp(x["timestamp"]/1000) for x in data]
        values = [float(x["sumOpenInterest"]) for x in data]
        return times, values

    def get_taker_ratio(self, symbol: str) -> float:
        url = f"{BASE}/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=10"
        data = self.safe_get(url)
        return float(data[0]["buySellRatio"])

    def get_long_short_ratio(self, symbol: str) -> tuple:
        url = f"{BASE}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=10"
        data = self.safe_get(url)
        long_ratio = float(data[0]["longAccount"])
        short_ratio = float(data[0]["shortAccount"])
        return long_ratio, short_ratio

    def get_price_data(self, symbol: str) -> Dict:
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

    def get_market_depth(self, symbol: str) -> Dict:
        """Dapatkan data market depth (order book)"""
        url = f"{BASE}/fapi/v1/depth?symbol={symbol}&limit=15"
        data = self.safe_get(url)
        bids = sum(float(bid[1]) * float(bid[0]) for bid in data["bids"][:8])
        asks = sum(float(ask[1]) * float(ask[0]) for ask in data["asks"][:8])
        total_bids = sum(float(bid[1]) for bid in data["bids"][:8])
        total_asks = sum(float(ask[1]) for ask in data["asks"][:8])
        
        return {
            "bids_value": bids,
            "asks_value": asks, 
            "bids_volume": total_bids,
            "asks_volume": total_asks,
            "pressure": bids/asks if asks > 0 else 1,
            "volume_pressure": total_bids/total_asks if total_asks > 0 else 1
        }

    def calculate_volume_analysis(self, symbol: str, current_volume: float) -> Dict:
        """Analisis volume dengan historical comparison"""
        volume_history = self.volume_history[symbol]
        
        # Tambahkan volume saat ini ke history
        volume_history.append(current_volume)
        
        if len(volume_history) < 5:
            return {"volume_spike": False, "volume_trend": "INSUFFICIENT_DATA"}
        
        # Hitung volume average
        avg_volume = np.mean(list(volume_history))
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Deteksi volume spike
        volume_spike = volume_ratio > self.config["VOLUME_THRESHOLD"]
        
        # Tentukan trend volume
        if len(volume_history) >= 10:
            recent_volumes = list(volume_history)[-5:]
            older_volumes = list(volume_history)[-10:-5]
            recent_avg = np.mean(recent_volumes)
            older_avg = np.mean(older_volumes)
            
            if recent_avg > older_avg * 1.2:
                volume_trend = "INCREASING"
            elif recent_avg < older_avg * 0.8:
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
        """Hitung indikator teknikal yang lebih advanced"""
        try:
            # Ambil data kline dengan berbagai timeframe
            timeframes = ['5m', '15m', '1h']
            technicals = {}
            
            for tf in timeframes:
                url = f"{BASE}/fapi/v1/klines?symbol={symbol}&interval={tf}&limit=50"
                data = self.safe_get(url)
                
                if not data:
                    continue
                    
                closes = np.array([float(c[4]) for c in data])
                highs = np.array([float(h[2]) for h in data])
                lows = np.array([float(l[3]) for l in data])
                volumes = np.array([float(v[5]) for v in data])
                
                # Simple Moving Averages
                sma_20 = np.mean(closes[-20:])
                sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
                
                # Exponential Moving Average (approximated)
                ema_12 = self.calculate_ema(closes, 12)
                ema_26 = self.calculate_ema(closes, 26)
                
                # RSI
                rsi = self.calculate_rsi(closes)
                
                # MACD
                macd, macd_signal = self.calculate_macd(closes)
                
                # Support Resistance dengan metode swing points
                resistance, support = self.calculate_support_resistance(highs, lows)
                
                # Volume Weighted Average Price (VWAP)
                vwap = np.sum(volumes * closes) / np.sum(volumes) if np.sum(volumes) > 0 else closes[-1]
                
                technicals[tf] = {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "ema_12": ema_12,
                    "ema_26": ema_26,
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_signal,
                    "macd_histogram": macd - macd_signal,
                    "resistance": resistance,
                    "support": support,
                    "vwap": vwap,
                    "trend": "BULLISH" if ema_12 > ema_26 and sma_20 > sma_50 else "BEARISH",
                    "momentum": ((closes[-1] - closes[-10]) / closes[-10]) * 100
                }
                
            return technicals
            
        except Exception as e:
            print(f"⚠️ Technical indicator error for {symbol}: {e}")
            return {}

    def calculate_ema(self, prices: np.array, period: int) -> float:
        """Hitung Exponential Moving Average"""
        if len(prices) < period:
            return float(prices[-1])
        
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        
        return np.convolve(prices[-period:], weights, mode='valid')[-1]

    def calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Hitung Relative Strength Index"""
        if len(prices) < period + 1:
            return 50
            
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

    def calculate_macd(self, prices: np.array) -> Tuple[float, float]:
        """Hitung MACD"""
        if len(prices) < 26:
            return 0, 0
            
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        macd_signal = self.calculate_ema(prices[-9:], 9)  # Signal line EMA 9 dari MACD
        
        return macd, macd_signal

    def calculate_support_resistance(self, highs: np.array, lows: np.array) -> Tuple[float, float]:
        """Hitung support dan resistance levels"""
        if len(highs) < 20 or len(lows) < 20:
            return max(highs), min(lows)
        
        # Simple method: recent high/low sebagai S/R
        resistance = np.max(highs[-10:])
        support = np.min(lows[-10:])
        
        return resistance, support

    def generate_entry_signals(self, symbol_data: Dict, technicals: Dict, volume_analysis: Dict) -> Dict:
        """Generate sinyal entry long/short yang spesifik"""
        price_data = symbol_data["price_data"]
        current_price = price_data["price"]
        
        # Default signal
        signal = {
            "action": "HOLD",
            "direction": "NEUTRAL", 
            "confidence": 0,
            "entry_price": 0,
            "take_profit": [],
            "stop_loss": 0,
            "reason": []
        }
        
        # Analisis multi-timeframe
        tf_5m = technicals.get('5m', {})
        tf_15m = technicals.get('15m', {})
        tf_1h = technicals.get('1h', {})
        
        # Skor bullish/bearish
        bull_score = 0
        bear_score = 0
        reasons = []
        
        # 1. Trend Analysis (30%)
        if tf_1h.get('trend') == 'BULLISH':
            bull_score += 3
            reasons.append("📈 Trend 1H Bullish")
        elif tf_1h.get('trend') == 'BEARISH':
            bear_score += 3
            reasons.append("📉 Trend 1H Bearish")
            
        if tf_15m.get('trend') == 'BULLISH':
            bull_score += 2
            reasons.append("📈 Trend 15M Bullish")
        elif tf_15m.get('trend') == 'BEARISH':
            bear_score += 2
            reasons.append("📉 Trend 15M Bearish")
            
        # 2. Momentum Analysis (25%)
        rsi_5m = tf_5m.get('rsi', 50)
        if rsi_5m < 30:
            bull_score += 2.5
            reasons.append("🔻 RSI Oversold")
        elif rsi_5m > 70:
            bear_score += 2.5
            reasons.append("🔺 RSI Overbought")
            
        macd_hist_5m = tf_5m.get('macd_histogram', 0)
        if macd_hist_5m > 0:
            bull_score += 1.5
            reasons.append("📊 MACD Bullish")
        else:
            bear_score += 1.5
            reasons.append("📊 MACD Bearish")
            
        # 3. Volume Analysis (20%)
        if volume_analysis.get('volume_spike'):
            if symbol_data["sentiment_score"] > 0:
                bull_score += 2
                reasons.append("💰 Volume Spike Bullish")
            else:
                bear_score += 2
                reasons.append("💰 Volume Spike Bearish")
                
        # 4. Price Action (15%)
        support = tf_5m.get('support', 0)
        resistance = tf_5m.get('resistance', 0)
        
        if current_price > support * 1.01 and current_price < support * 1.02:
            bull_score += 1.5
            reasons.append("🛡️ Near Support Bounce")
        elif current_price < resistance * 0.99 and current_price > resistance * 0.98:
            bear_score += 1.5
            reasons.append("🎯 Near Resistance Rejection")
            
        # 5. Market Structure (10%)
        if tf_5m.get('ema_12', 0) > tf_5m.get('ema_26', 0) and tf_15m.get('ema_12', 0) > tf_15m.get('ema_26', 0):
            bull_score += 1
            reasons.append("⚡ EMA Alignment Bullish")
        elif tf_5m.get('ema_12', 0) < tf_5m.get('ema_26', 0) and tf_15m.get('ema_12', 0) < tf_15m.get('ema_26', 0):
            bear_score += 1
            reasons.append("⚡ EMA Alignment Bearish")
            
        # Determine final signal
        score_difference = bull_score - bear_score
        total_score = bull_score + bear_score
        confidence = min(abs(score_difference) / max(total_score, 1) * 100, 100)
        
        if score_difference >= 3 and confidence > 60:
            signal["action"] = "LONG"
            signal["direction"] = "BULLISH"
            signal["confidence"] = confidence
            signal["entry_price"] = current_price
            signal["stop_loss"] = support * 0.995  # 0.5% below support
            signal["take_profit"] = [
                resistance * 0.99,  # TP1 at resistance
                resistance * 1.02   # TP2 above resistance
            ]
            signal["reason"] = reasons
            
        elif score_difference <= -3 and confidence > 60:
            signal["action"] = "SHORT" 
            signal["direction"] = "BEARISH"
            signal["confidence"] = confidence
            signal["entry_price"] = current_price
            signal["stop_loss"] = resistance * 1.005  # 0.5% above resistance
            signal["take_profit"] = [
                support * 1.01,  # TP1 at support
                support * 0.98   # TP2 below support
            ]
            signal["reason"] = reasons
            
        return signal

    def calculate_sentiment_score(self, symbol_data: Dict) -> tuple:
        """Hitung skor sentimen dengan weighting yang lebih sophisticated"""
        score = 0
        details = []

        # Funding Rate (Weight: 20%)
        fund = symbol_data["funding"]
        if fund > 0.001:
            score += 2.0
            details.append("💰 Funding sangat positif")
        elif fund > 0.0001:
            score += 1.0
            details.append("💰 Funding positif")
        elif fund < -0.001:
            score -= 2.0
            details.append("💰 Funding sangat negatif")
        elif fund < -0.0001:
            score -= 1.0
            details.append("💰 Funding negatif")

        # Open Interest (Weight: 15%)
        oi_change = symbol_data["oi_change"]
        if oi_change > 2:
            score += 1.5
            details.append("📈 OI meningkat kuat")
        elif oi_change > 0.5:
            score += 0.75
            details.append("📈 OI meningkat")
        elif oi_change < -2:
            score -= 1.5
            details.append("📉 OI menurun kuat")
        elif oi_change < -0.5:
            score -= 0.75
            details.append("📉 OI menurun")

        # Taker Ratio (Weight: 15%)
        taker = symbol_data["taker_ratio"]
        if taker > 1.2:
            score += 1.5
            details.append("⚔️ Buyer sangat agresif")
        elif taker > 1.05:
            score += 0.75
            details.append("⚔️ Buyer agresif")
        elif taker < 0.8:
            score -= 1.5
            details.append("⚔️ Seller sangat agresif")
        elif taker < 0.95:
            score -= 0.75
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

        # Market Depth (Weight: 15%)
        depth = symbol_data["depth"]
        if depth["pressure"] > 1.2:
            score += 1.5
            details.append("🏊 Bid pressure kuat")
        elif depth["pressure"] > 1.05:
            score += 0.75
            details.append("🏊 Bid pressure")
        elif depth["pressure"] < 0.8:
            score -= 1.5
            details.append("🏊 Ask pressure kuat")
        elif depth["pressure"] < 0.95:
            score -= 0.75
            details.append("🏊 Ask pressure")

        # Volume Analysis (Weight: 20%)
        volume_analysis = symbol_data["volume_analysis"]
        if volume_analysis["volume_spike"]:
            if volume_analysis["volume_trend"] == "INCREASING":
                score += 2.0
                details.append("📊 Volume spike increasing")
            else:
                score += 1.0
                details.append("📊 Volume spike terdeteksi")

        return score, details

    def get_risk_recommendation(self, symbol: str, score: float, price_data: Dict, entry_signal: Dict) -> str:
        """Beri rekomendasi risk management berdasarkan skor dan sinyal entry"""
        risk_level = self.config["RISK_LEVEL"]
        price = price_data["price"]
        change_24h = price_data["change_24h"]

        base_recommendation = ""
        entry_details = ""

        # Tambahkan detail entry signal jika ada
        if entry_signal["action"] != "HOLD":
            action_emoji = "🟢" if entry_signal["action"] == "LONG" else "🔴"
            entry_details = (
                f"\n🎯 <b>ENTRY SIGNAL: {entry_signal['action']}</b>\n"
                f"📊 Confidence: <code>{entry_signal['confidence']:.1f}%</code>\n"
                f"💰 Entry: <code>${entry_signal['entry_price']:,.2f}</code>\n"
                f"🛡️ Stop Loss: <code>${entry_signal['stop_loss']:,.2f}</code>\n"
                f"🎯 Take Profit: <code>${entry_signal['take_profit'][0]:,.2f}</code> | <code>${entry_signal['take_profit'][1]:,.2f}</code>\n"
            )
            
            # Tambahkan alasan
            if entry_signal["reason"]:
                entry_details += "📋 Reasons:\n• " + "\n• ".join(entry_signal["reason"][:5]) + "\n"

        if score >= 6:
            base_recommendation = "🟢 STRONG BULLISH - Konfirmasi bullish kuat dari multiple indikator"
            if risk_level == "LOW":
                return f"{base_recommendation}\n💡 Risk: Position kecil 1-2%, TP 3-5%, SL 2%{entry_details}"
            elif risk_level == "MEDIUM":
                return f"{base_recommendation}\n💡 Risk: Position medium 3-5%, TP 5-8%, SL 3%{entry_details}"
            else:
                return f"{base_recommendation}\n💡 Risk: Position besar 5-7%, TP 8-12%, SL 4%{entry_details}"

        elif score >= 3:
            base_recommendation = "🟡 MILD BULLISH - Signal bullish moderat"
            return f"{base_recommendation}\n💡 Risk: Position kecil 1-3%, TP 3-6%, SL 2.5%{entry_details}"

        elif score >= 1:
            base_recommendation = "⚪️ NEUTRAL BULLISH - Sedikit bias bullish"
            return f"{base_recommendation}\n💡 Risk: Wait for confirmation atau position sangat kecil 0.5-1%{entry_details}"

        elif score <= -6:
            base_recommendation = "🔴 STRONG BEARISH - Konfirmasi bearish kuat"
            if risk_level == "LOW":
                return f"{base_recommendation}\n💡 Risk: Position kecil 1-2%, TP 3-5%, SL 2%{entry_details}"
            elif risk_level == "MEDIUM":
                return f"{base_recommendation}\n💡 Risk: Position medium 3-5%, TP 5-8%, SL 3%{entry_details}"
            else:
                return f"{base_recommendation}\n💡 Risk: Position besar 5-7%, TP 8-12%, SL 4%{entry_details}"

        elif score <= -3:
            base_recommendation = "🟠 MILD BEARISH - Signal bearish moderat"
            return f"{base_recommendation}\n💡 Risk: Position kecil 1-3%, TP 3-6%, SL 2.5%{entry_details}"

        elif score <= -1:
            base_recommendation = "⚪️ NEUTRAL BEARISH - Sedikit bias bearish"
            return f"{base_recommendation}\n💡 Risk: Wait for confirmation atau position sangat kecil 0.5-1%{entry_details}"
        else:
            return f"⚖️ NEUTRAL - Market sideways, tunggu breakout\n💡 Risk: No position atau sangat kecil 0.5%{entry_details}"

    def analyze_symbol(self, symbol: str):
        """Analisis untuk satu simbol"""
        try:
            print(f"📊 Analyzing {symbol}...")

            # Collect semua data
            funding = self.get_funding_rate(symbol)
            times, oi = self.get_open_interest(symbol)
            taker_ratio = self.get_taker_ratio(symbol)
            long_ratio, short_ratio = self.get_long_short_ratio(symbol)
            price_data = self.get_price_data(symbol)
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
                f"💎 Volume: <code>{price_data['volume']:,.0f}</code>\n"
                f"💰 Quote Volume: <code>${price_data['quote_volume']:,.0f}</code>\n\n"

                f"<b>📊 MARKET SENTIMENT:</b>\n"
                f"💸 Funding: <code>{funding*100:+.4f}%</code>\n"
                f"📈 OI Change: <code>{oi_change:+.3f}%</code>\n"
                f"⚔️ Taker Ratio: <code>{taker_ratio:.3f}</code>\n"
                f"👥 Long/Short: <code>{long_ratio:.3f}</code>/<code>{short_ratio:.3f}</code>\n"
                f"🏊 Bid/Ask Pressure: <code>{depth['pressure']:.2f}</code>\n\n"
            )

            # Tambahkan volume analysis
            if volume_analysis:
                volume_emoji = "🚀" if volume_analysis["volume_spike"] else "📊"
                message += (
                    f"<b>📊 VOLUME ANALYSIS:</b>\n"
                    f"{volume_emoji} Volume Ratio: <code>{volume_analysis['volume_ratio']:.2f}x</code>\n"
                    f"📈 Volume Trend: <code>{volume_analysis['volume_trend']}</code>\n"
                    f"💰 Avg Volume: <code>${volume_analysis['avg_volume']:,.0f}</code>\n\n"
                )

            # Tambahkan technical indicators jika ada
            if technicals and '15m' in technicals:
                tf = technicals['15m']
                message += (
                    f"<b>🔧 TECHNICALS (15M):</b>\n"
                    f"📊 Trend: <code>{tf.get('trend', 'N/A')}</code>\n"
                    f"📈 RSI: <code>{tf.get('rsi', 0):.1f}</code>\n"
                    f"📉 MACD: <code>{tf.get('macd', 0):.4f}</code>\n"
                    f"🛡️ Support: <code>${tf.get('support', 0):,.2f}</code>\n"
                    f"🎯 Resistance: <code>${tf.get('resistance', 0):,.2f}</code>\n\n"
                )

            # Tambahkan scoring details
            message += f"<b>🎲 SENTIMENT SCORE: <code>{score:.1f}/10</code></b>\n"
            if score_details:
                message += "📋 Details:\n• " + "\n• ".join(score_details) + "\n\n"

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
            self.send_telegram(f"❌ <b>Error analyzing {symbol}:</b>\n{str(e)}", symbol)

    def analyze_all_symbols(self):
        """Analisis semua simbol dalam konfigurasi"""
        print(f"\n🔄 Memulai analisis {len(self.config['SYMBOLS'])} simbol...")
        print("=" * 60)

        for symbol in self.config["SYMBOLS"]:
            self.analyze_symbol(symbol)
            # Jeda antara simbol untuk avoid rate limit
            time.sleep(2)

        print("=" * 60)
        print(f"✅ Semua analisis selesai - Menunggu {self.config['INTERVAL']//60} menit")

    def run(self):
        """Jalankan monitor secara kontinu"""
        print("🚀 ENHANCED FUTURES MULTI-ASSET MONITOR DIMULAI")
        print(f"📍 Timezone: {self.config['TIMEZONE']}")
        print(f"📊 Symbols: {', '.join(self.config['SYMBOLS'])}")
        print(f"⏰ Interval: {self.config['INTERVAL']} detik")
        print(f"🎯 Risk Level: {self.config['RISK_LEVEL']}")
        print(f"📈 Volume Threshold: {self.config['VOLUME_THRESHOLD']}x")
        print("=" * 60)

        while True:
            try:
                self.analyze_all_symbols()
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                self.send_telegram(f"❌ <b>System Error:</b>\n{str(e)}", "SYSTEM")

            wait_minutes = self.config["INTERVAL"] // 60
            print(f"⏳ Menunggu {wait_minutes} menit hingga analisis berikutnya...")
            print("=" * 60)
            time.sleep(self.config["INTERVAL"])

def main():
    # Jalankan analyzer
    analyzer = EnhancedFuturesAnalyzer(CONFIG)
    analyzer.run()

if __name__ == "__main__":
    main()
