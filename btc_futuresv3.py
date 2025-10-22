import requests, datetime, time, json, os
from typing import Dict, List, Optional

# === KONFIGURASI ===
CONFIG = {
    "TELEGRAM_BOT_TOKEN": "8345066310:AAElrMezSmJZwWOWWRYLFMf2z5nyDCkTg0g",
    "TELEGRAM_CHAT_ID": "623072599",
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "SOLUSDT"],
    "INTERVAL": 60 * 5,  # 5 menit
    "TIMEZONE": "Asia/Pontianak",  # UTC+7
    "RISK_LEVEL": "MEDIUM"  # LOW, MEDIUM, HIGH
}

BASE = "https://fapi.binance.com"

class FuturesAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.analysis_history = {}
        
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
            "quote_volume": float(data["quoteVolume"])
        }

    def get_market_depth(self, symbol: str) -> Dict:
        """Dapatkan data market depth (order book)"""
        url = f"{BASE}/fapi/v1/depth?symbol={symbol}&limit=10"
        data = self.safe_get(url)
        bids = sum(float(bid[1]) for bid in data["bids"][:5])
        asks = sum(float(ask[1]) for ask in data["asks"][:5])
        return {"bids": bids, "asks": asks, "pressure": bids/asks if asks > 0 else 1}

    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """Hitung indikator teknikal sederhana"""
        try:
            url = f"{BASE}/fapi/v1/klines?symbol={symbol}&interval=15m&limit=20"
            data = self.safe_get(url)
            closes = [float(c[4]) for c in data]
            
            # Simple Moving Average
            sma_short = sum(closes[-5:]) / 5
            sma_long = sum(closes[-15:]) / 15
            
            # Price momentum
            momentum = ((closes[-1] - closes[-5]) / closes[-5]) * 100
            
            # Support Resistance sederhana
            resistance = max(closes[-10:])
            support = min(closes[-10:])
            
            return {
                "sma_short": sma_short,
                "sma_long": sma_long,
                "momentum": momentum,
                "resistance": resistance,
                "support": support,
                "trend": "BULLISH" if sma_short > sma_long else "BEARISH"
            }
        except:
            return {}

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

        # Price Momentum (Weight: 10%)
        tech = symbol_data.get("technical", {})
        momentum = tech.get("momentum", 0)
        if momentum > 1:
            score += 1.0
            details.append("🚀 Momentum positif")
        elif momentum < -1:
            score -= 1.0
            details.append("📉 Momentum negatif")

        return score, details

    def get_risk_recommendation(self, symbol: str, score: float, price_data: Dict) -> str:
        """Beri rekomendasi risk management berdasarkan skor"""
        risk_level = self.config["RISK_LEVEL"]
        price = price_data["price"]
        change_24h = price_data["change_24h"]
        
        base_recommendation = ""
        
        if score >= 6:
            base_recommendation = "🟢 STRONG BUY - Konfirmasi bullish kuat dari multiple indikator"
            if risk_level == "LOW":
                return f"{base_recommendation}\n💡 Risk: Position kecil 1-2%, TP 3-5%, SL 2%"
            elif risk_level == "MEDIUM":
                return f"{base_recommendation}\n💡 Risk: Position medium 3-5%, TP 5-8%, SL 3%"
            else:
                return f"{base_recommendation}\n💡 Risk: Position besar 5-7%, TP 8-12%, SL 4%"
                
        elif score >= 3:
            base_recommendation = "🟡 MILD BUY - Signal bullish moderat"
            return f"{base_recommendation}\n💡 Risk: Position kecil 1-3%, TP 3-6%, SL 2.5%"
            
        elif score >= 1:
            base_recommendation = "⚪️ NEUTRAL BULLISH - Sedikit bias bullish"
            return f"{base_recommendation}\n💡 Risk: Wait for confirmation atau position sangat kecil 0.5-1%"
            
        elif score <= -6:
            base_recommendation = "🔴 STRONG SELL - Konfirmasi bearish kuat"
            if risk_level == "LOW":
                return f"{base_recommendation}\n💡 Risk: Position kecil 1-2%, TP 3-5%, SL 2%"
            elif risk_level == "MEDIUM":
                return f"{base_recommendation}\n💡 Risk: Position medium 3-5%, TP 5-8%, SL 3%"
            else:
                return f"{base_recommendation}\n💡 Risk: Position besar 5-7%, TP 8-12%, SL 4%"
                
        elif score <= -3:
            base_recommendation = "🟠 MILD SELL - Signal bearish moderat"
            return f"{base_recommendation}\n💡 Risk: Position kecil 1-3%, TP 3-6%, SL 2.5%"
            
        elif score <= -1:
            base_recommendation = "⚪️ NEUTRAL BEARISH - Sedikit bias bearish"
            return f"{base_recommendation}\n💡 Risk: Wait for confirmation atau position sangat kecil 0.5-1%"
        else:
            return "⚖️ NEUTRAL - Market sideways, tunggu breakout\n💡 Risk: No position atau sangat kecil 0.5%"

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
            technicals = self.calculate_technical_indicators(symbol)
            
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
                "technical": technicals
            }

            # Hitung skor sentimen
            score, score_details = self.calculate_sentiment_score(symbol_data)
            
            # Dapatkan rekomendasi
            recommendation = self.get_risk_recommendation(symbol, score, price_data)
            
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
                f"📈 OI Change: <code>{oi_change:+.3f}%</code>\n"
                f"⚔️ Taker Ratio: <code>{taker_ratio:.3f}</code>\n"
                f"👥 Long/Short: <code>{long_ratio:.3f}</code>/<code>{short_ratio:.3f}</code>\n"
                f"🏊 Bid/Ask Pressure: <code>{depth['pressure']:.2f}</code>\n\n"
            )
            
            # Tambahkan technical indicators jika ada
            if technicals:
                message += (
                    f"<b>🔧 TECHNICALS:</b>\n"
                    f"📊 Trend: <code>{technicals.get('trend', 'N/A')}</code>\n"
                    f"🚀 Momentum: <code>{technicals.get('momentum', 0):+.2f}%</code>\n"
                    f"🛡️ Support: <code>${technicals.get('support', 0):,.2f}</code>\n"
                    f"🎯 Resistance: <code>${technicals.get('resistance', 0):,.2f}</code>\n\n"
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
                "price": price_data["price"]
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
        print("🚀 FUTURES MULTI-ASSET MONITOR DIMULAI")
        print(f"📍 Timezone: {self.config['TIMEZONE']}")
        print(f"📊 Symbols: {', '.join(self.config['SYMBOLS'])}")
        print(f"⏰ Interval: {self.config['INTERVAL']} detik")
        print(f"🎯 Risk Level: {self.config['RISK_LEVEL']}")
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
    analyzer = FuturesAnalyzer(CONFIG)
    analyzer.run()

if __name__ == "__main__":
    main()