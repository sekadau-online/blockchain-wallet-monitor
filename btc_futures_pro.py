import requests, datetime, time, io
import matplotlib.pyplot as plt

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

def send_telegram(msg, img=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})
    if img:
        f = {"photo": ("chart.png", img)}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                      data={"chat_id": TELEGRAM_CHAT_ID}, files=f)

def get_funding(symbol):
    url = f"{BASE}/fapi/v1/fundingRate?symbol={symbol}&limit=1"
    data = safe_get(url)[0]
    return float(data["fundingRate"])

def get_oi(symbol):
    url = f"{BASE}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=20"
    data = safe_get(url)
    times = [datetime.datetime.fromtimestamp(x["timestamp"]/1000) for x in data]
    values = [float(x["sumOpenInterest"]) for x in data]
    return times, values

def get_taker_ratio(symbol):
    url = f"{BASE}/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=1"
    data = safe_get(url)[0]
    return float(data["buySellRatio"])  # >1 berarti buyer agresif

def get_liquidations(symbol):
    url = f"{BASE}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
    data = safe_get(url)[0]
    long_ratio = float(data["longAccount"])
    short_ratio = float(data["shortAccount"])
    return long_ratio, short_ratio

def get_price(symbol):
    url = f"{BASE}/fapi/v1/ticker/price?symbol={symbol}"
    return float(safe_get(url)["price"])

def analyze():
    fund = get_funding(SYMBOL)
    times, oi = get_oi(SYMBOL)
    taker = get_taker_ratio(SYMBOL)
    long_ratio, short_ratio = get_liquidations(SYMBOL)
    price = get_price(SYMBOL)
    oi_change = (oi[-1] - oi[-2]) / oi[-2] * 100

    # --- Skoring sederhana ---
    score = 0
    if fund > 0: score += 1
    elif fund < 0: score -= 1

    if oi_change > 0: score += 1
    elif oi_change < 0: score -= 1

    if taker > 1: score += 1
    elif taker < 1: score -= 1

    if long_ratio > short_ratio: score += 1
    else: score -= 1

    # --- Kesimpulan ---
    if score >= 3:
        trend = "🔥 STRONG LONG"
    elif score <= -3:
        trend = "🐻 STRONG SHORT"
    elif -2 <= score <= 2:
        trend = "⚖️ NEUTRAL / SIDEWAYS"
    else:
        trend = "❓ Ambiguous"

    # --- Plot grafik OI ---
    plt.figure(figsize=(6, 3))
    plt.plot(times, oi, marker="o", linewidth=2)
    plt.title(f"Open Interest {SYMBOL}")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"<b>BTC Futures Multi-Parameter</b>\n"
        f"⏰ {now} UTC\n"
        f"💰 Price: {price:.2f} USDT\n"
        f"💸 Funding: {fund*100:.4f}%\n"
        f"📈 OI Δ: {oi_change:.3f}%\n"
        f"⚔️ Taker Ratio: {taker:.2f}\n"
        f"👥 Long/Short: {long_ratio:.2f}/{short_ratio:.2f}\n"
        f"📊 Score: {score}\n"
        f"🧭 Sentiment: {trend}"
    )
    print(msg)
    send_telegram(msg, img=buf)

if __name__ == "__main__":
    print("🚀 BTC Futures Pro Monitor dimulai...")
    while True:
        try:
            analyze()
        except Exception as e:
            print("❌ Error:", e)
        time.sleep(INTERVAL)