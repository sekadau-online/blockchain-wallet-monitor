import requests, datetime, time

# === KONFIGURASI ===
TELEGRAM_BOT_TOKEN = "8345066310:AAElrMezSmJZwWOWWRYLFMf2z5nyDCkTg0g"
TELEGRAM_CHAT_ID   = "623072599"
SYMBOL = "BTCUSDT"
INTERVAL = 60 * 5

BASE = "https://fapi.binance.com"

def safe_get(url):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print("⚠️ Gagal konek:", e)
            time.sleep(3)
    raise Exception("Gagal konek API Binance")

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})
        print("✅ Pesan terkirim ke Telegram")
    except Exception as e:
        print("⚠️ Gagal kirim Telegram:", e)

def get_funding(symbol):
    url = f"{BASE}/fapi/v1/fundingRate?symbol={symbol}&limit=1"
    data = safe_get(url)[0]
    return float(data["fundingRate"])

def get_oi(symbol):
    url = f"{BASE}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=20"
    data = safe_get(url)
    values = [float(x["sumOpenInterest"]) for x in data]
    return values

def get_taker_ratio(symbol):
    url = f"{BASE}/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=1"
    data = safe_get(url)[0]
    return float(data["buySellRatio"])

def get_liquidations(symbol):
    url = f"{BASE}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
    data = safe_get(url)[0]
    long_ratio = float(data["longAccount"])
    short_ratio = float(data["shortAccount"])
    return long_ratio, short_ratio

def get_price(symbol):
    url = f"{BASE}/fapi/v1/ticker/price?symbol={symbol}"
    return float(safe_get(url)["price"])

def get_price_change(symbol):
    url = f"{BASE}/fapi/v1/ticker/24hr?symbol={symbol}"
    data = safe_get(url)
    price_change = float(data["priceChangePercent"])
    return price_change

def analyze():
    try:
        print("📊 Mengambil data dari Binance...")
        
        # Ambil data secara paralel (dalam sequence)
        fund = get_funding(SYMBOL)
        print(f"✅ Funding rate: {fund*100:.4f}%")
        
        oi = get_oi(SYMBOL)
        taker = get_taker_ratio(SYMBOL)
        long_ratio, short_ratio = get_liquidations(SYMBOL)
        price = get_price(SYMBOL)
        price_change_24h = get_price_change(SYMBOL)
        
        # Hitung perubahan OI
        if len(oi) >= 2:
            oi_change = (oi[-1] - oi[-2]) / oi[-2] * 100
            oi_trend = "📈" if oi_change > 0 else "📉"
        else:
            oi_change = 0
            oi_trend = "➡️"

        # --- Skoring sederhana ---
        score = 0
        
        # Funding rate scoring
        if fund > 0.0001: 
            score += 1
            fund_emoji = "🟢"
        elif fund < -0.0001: 
            score -= 1
            fund_emoji = "🔴"
        else:
            fund_emoji = "🟡"

        # OI change scoring
        if oi_change > 0.5: 
            score += 1
        elif oi_change < -0.5: 
            score -= 1

        # Taker ratio scoring
        if taker > 1.05: 
            score += 1
            taker_emoji = "🟢"
        elif taker < 0.95: 
            score -= 1
            taker_emoji = "🔴"
        else:
            taker_emoji = "🟡"

        # Long/Short ratio scoring
        if long_ratio > short_ratio + 0.1: 
            score += 1
            ls_emoji = "🟢"
        elif short_ratio > long_ratio + 0.1: 
            score -= 1
            ls_emoji = "🔴"
        else:
            ls_emoji = "🟡"

        # --- Kesimpulan ---
        if score >= 3:
            trend = "🔥 STRONG BULLISH"
            trend_emoji = "🚀"
        elif score >= 1:
            trend = "📗 MILD BULLISH"
            trend_emoji = "↗️"
        elif score <= -3:
            trend = "🐻 STRONG BEARISH" 
            trend_emoji = "📉"
        elif score <= -1:
            trend = "📘 MILD BEARISH"
            trend_emoji = "↘️"
        else:
            trend = "⚖️ NEUTRAL"
            trend_emoji = "➡️"

        # Fix deprecated datetime.utcnow()
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format pesan dengan emoji dan layout yang lebih baik
        msg = (
            f"<b>₿ BTC Futures Analysis</b> {trend_emoji}\n"
            f"⏰ {now} UTC\n\n"
            
            f"<b>Price Data:</b>\n"
            f"💰 Price: <code>{price:,.2f}</code> USDT\n"
            f"📊 24h Change: <code>{price_change_24h:+.2f}%</code>\n\n"
            
            f"<b>Sentiment Indicators:</b>\n"
            f"{fund_emoji} Funding: <code>{fund*100:+.4f}%</code>\n"
            f"{oi_trend} OI Change: <code>{oi_change:+.3f}%</code>\n"
            f"{taker_emoji} Taker Ratio: <code>{taker:.3f}</code>\n"
            f"{ls_emoji} Long/Short: <code>{long_ratio:.3f}</code>/<code>{short_ratio:.3f}</code>\n\n"
            
            f"<b>Summary:</b>\n"
            f"📈 Score: <code>{score:+d}</code>/4\n"
            f"🎯 Trend: <b>{trend}</b>\n\n"
            
            f"<i>Update setiap {INTERVAL//60} menit</i>"
        )
        
        print(f"✅ Analysis completed - Score: {score} - Trend: {trend}")
        send_telegram(msg)
        
    except Exception as e:
        error_msg = f"❌ Error dalam analisis: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)

if __name__ == "__main__":
    print(f"🚀 BTC Futures Monitor dimulai...")
    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL} detik ({INTERVAL//60} menit)")
    print("-" * 50)
    
    while True:
        try:
            analyze()
        except Exception as e:
            print(f"❌ Main loop error: {e}")
        
        print(f"⏳ Menunggu {INTERVAL//60} menit hingga analisis berikutnya...")
        print("-" * 50)
        time.sleep(INTERVAL)