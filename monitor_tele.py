import requests
import time
import telegram

# === KONFIGURASI ===
wallet_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"  # contoh whale, ganti dengan wallet institusi
etherscan_api_key = "YOUR_ETHERSCAN_API_KEY"  # ganti API key etherscan
telegram_token = "YOUR_TELEGRAM_BOT_TOKEN"    # dari BotFather
chat_id = "YOUR_CHAT_ID"                      # chat id kamu

bot = telegram.Bot(token=telegram_token)

def get_transactions(address):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&sort=desc&apikey={etherscan_api_key}"
    response = requests.get(url).json()
    if response["status"] == "1":
        return response["result"]
    return []

def track_wallet(address):
    print(f"Tracking wallet: {address}\n")
    last_tx = None
    while True:
        txs = get_transactions(address)
        if txs:
            latest = txs[0]
            if last_tx is None:
                last_tx = latest["hash"]
            elif latest["hash"] != last_tx:
                message = (
                    f"🚨 Transaksi Baru Terdeteksi 🚨\n\n"
                    f"🔗 Hash: {latest['hash']}\n"
                    f"📤 Dari : {latest['from']}\n"
                    f"📥 Ke   : {latest['to']}\n"
                    f"💰 Jumlah : {int(latest['value'])/1e18:.4f} ETH\n"
                    f"⛓️ Block  : {latest['blockNumber']}\n"
                )
                bot.send_message(chat_id=chat_id, text=message)
                print(message)
                last_tx = latest["hash"]
        time.sleep(15)  # cek tiap 15 detik

# Jalankan
track_wallet(wallet_address)