🔔 Blockchain Wallet Monitor




Real-time monitoring of cryptocurrency wallet activity with instant email alerts.
Perfect for detecting unauthorized transfers and keeping your assets secure—24/7 protection.

🛡️ Sleep Peacefully, We’ve Got Your Wallet Covered 😴
✨ Features
🔗 Multi-chain Support
Ethereum, BNB Smart Chain, Polygon, Optimism, Arbitrum, and more.

📬 Instant Email Alerts
Immediate notification when funds leave your wallet.

⏱️ Real-time Monitoring
Continuously scans for new outgoing transactions.

📊 Detailed Transaction Info
Includes amount, timestamp, gas fees, and direct explorer links.

⚙️ Easy Configuration
Just update a simple .env file.

🔒 Security Focused
No private keys required – read-only via public blockchain APIs.

⚙️ Prerequisites
Python 3.8+

Etherscan API Key (Free tier supported)

SMTP Email Credentials (Gmail or others)

🚀 Quick Installation
bash
Salin
Edit
# 1. Clone the repository
git clone https://github.com/sekadau-online/blockchain-wallet-monitor.git

# 2. Navigate into the directory
cd blockchain-wallet-monitor

# 3. Install dependencies
pip install -r requirements.txt
⚡ Quick Start
Create a .env file in the root directory with your configuration:

ini
Salin
Edit
# Wallet settings
DEPLOYER_WALLET=0xYourWalletAddress
CHAIN_ID=56
ETHERSCAN_API_KEY=your_etherscan_api_key

# Email alert settings
EMAIL_USER=your@email.com
EMAIL_PASS=your_app_password
EMAIL_TO=recipient@email.com
Start the monitor:

bash
Salin
Edit
python3 wallet_monitor.py
🛠️ Configuration Options
Variable	Description	Default
DEPLOYER_WALLET	Wallet address to monitor	(Required)
CHAIN_ID	Blockchain network ID	1 (Ethereum)
ETHERSCAN_API_KEY	Your Etherscan API key	(Required)
CHECK_INTERVAL	Seconds between checks	300 (5 min)
EMAIL_USER	SMTP email username	(Required)
EMAIL_PASS	SMTP email password/app pass	(Required)
EMAIL_TO	Recipient email address	(Required)
SMTP_SERVER	SMTP server address	smtp.gmail.com
SMTP_PORT	SMTP server port	587

🌐 Supported Blockchains
Chain ID	Network	Currency
1	Ethereum	ETH
56	BNB Smart Chain	BNB
137	Polygon	MATIC
10	Optimism	ETH
42161	Arbitrum	ETH
8453	Base	ETH
5	Goerli Testnet	ETH
11155111	Sepolia Testnet	ETH

📬 Sample Alert Email
yaml
Salin
Edit
🚨 CRITICAL: Funds movement detected from monitored wallet!

Transaction Hash: 0x515d2e21fee87a01252192c8d9303ab4e18c68b4dc521af118dd1c364e3ff358
Chain: Ethereum Mainnet
From: 0x95e9123ccca26aae1de8cb52563802c84a8d2636
To: 0x47b13583fa53663e610bbd1d856487081164c89a
Amount: 18.586522 ETH
Date: 2025-06-23 10:48:23

🔗 View Transaction:  
https://etherscan.io/tx/0x515d2e21fee87a01252192c8d9303ab4e18c68b4dc521af118dd1c364e3ff358
☕ Support My Work
If this project helps you secure your crypto, consider supporting me:

💸 Crypto Donations

Coin	Address
Verus	RPMu8QpUxvevPuTX2baVeVmt9PvYfWjURN
Bitcoin	1F5i3twCN6rMKu6KZRbNYySgP8TwzKrWgh
Ethereum	0x1F491f5d86b78865cD20379FC47FaA04E4f5ceB3
Litecoin	LZJfK7F2Sm6QahnUjZafpzWSbLqE7mp2NK

📜 License
This project is licensed under the MIT License.

⚠️ Disclaimer
This software is provided “as is” without warranty. Use at your own risk.
The author is not responsible for any financial loss or security breach resulting from the use of this tool.
Always secure your wallets and API keys properly.

