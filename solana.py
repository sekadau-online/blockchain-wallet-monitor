import requests
import asyncio
import aiohttp
from telegram import Bot
from datetime import datetime

# Konfigurasi
SOLANA_RPC_URL = "https://api.mainnet.solana.com"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
MIN_SOL_AMOUNT = 1000  # Minimum jumlah SOL untuk dianggap sebagai whale transaction

async def get_recent_blockhash():
    """Mendapatkan blockhash terbaru dari jaringan Solana"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getLatestBlockhash",
        "params": [{"commitment": "finalized"}]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            result = await response.json()
            return result['result']['value']['blockhash']

async def get_block_transactions(blockhash):
    """Mendapatkan transaksi dari block tertentu"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBlock",
        "params": [
            blockhash,
            {"encoding": "json", "transactionDetails": "full", "rewards": False}
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            result = await response.json()
            return result['result']['transactions'] if 'result' in result else []

async def check_whale_transactions(transactions):
    """Mengecek transaksi whale berdasarkan jumlah SOL"""
    whale_transactions = []
    
    for tx in transactions:
        try:
            # Mendapatkan informasi transaksi
            tx_info = tx['transaction']
            pre_balances = tx['meta']['preBalances']
            post_balances = tx['meta']['postBalances']
            
            # Menghitung perubahan balance untuk setiap account
            for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
                balance_change = (post - pre) / 1e9  # Convert lamports to SOL
                
                if abs(balance_change) >= MIN_SOL_AMOUNT:
                    whale_transactions.append({
                        'signature': tx_info['signatures'][0],
                        'balance_change': balance_change,
                        'from_account': tx_info['message']['accountKeys'][i],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        except (KeyError, TypeError):
            continue
    
    return whale_transactions

async def send_telegram_alert(transaction):
    """Mengirim notifikasi ke Telegram"""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    message = f"🚨 WHALE ALERT 🚨\n\n"
    message += f"Signature: {transaction['signature'][:10]}...\n"
    message += f"Amount: {transaction['balance_change']:,.2f} SOL\n"
    message += f"Account: {transaction['from_account'][:10]}...\n"
    message += f"Time: {transaction['timestamp']}"
    
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

async def monitor_whale_transactions():
    """Monitoring utama untuk transaksi whale"""
    print("Memulai monitoring whale transactions di Solana...")
    last_blockhash = None
    
    while True:
        try:
            current_blockhash = await get_recent_blockhash()
            
            if current_blockhash != last_blockhash:
                transactions = await get_block_transactions(current_blockhash)
                whale_txs = await check_whale_transactions(transactions)
                
                for tx in whale_txs:
                    await send_telegram_alert(tx)
                
                last_blockhash = current_blockhash
            
            await asyncio.sleep(5)  # Check setiap 5 detik
            
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    # Pastikan untuk mengganti dengan token dan chat ID Telegram Anda
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
        print("ERROR: Silakan set TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID yang valid")
    else:
        asyncio.run(monitor_whale_transactions())