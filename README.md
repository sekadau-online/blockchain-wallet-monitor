Blockchain Wallet Monitor 🔔
https://img.shields.io/badge/python-3.8%252B-blue
https://img.shields.io/badge/license-MIT-green
https://img.shields.io/badge/status-active-brightgreen

Monitor real-time cryptocurrency wallet transactions and receive instant email alerts for outgoing transfers. Perfect for detecting suspicious activity and protecting your assets.

Monitor your wallets 24/7 and sleep peacefully 😴

✨ Features
Multi-chain Support: Ethereum, BSC, Polygon, Optimism, Arbitrum, and more

Instant Email Alerts: Get notified immediately when funds leave your wallet

Real-time Monitoring: Continuously checks for new transactions

Detailed Transaction Info: Amount, timestamp, gas fees, and direct explorer links

Easy Configuration: Simple .env file setup

Security Focused: No wallet keys required - read-only API access

⚙️ Prerequisites
Python 3.8+

Etherscan API key (free tier available)

SMTP email credentials (Gmail or other provider)

🚀 Installation
bash
# Clone the repository
git clone https://github.com/sekadau-online/blockchain-wallet-monitor.git

# Navigate to project directory
cd blockchain-wallet-monitor

# Install dependencies
pip install -r requirements.txt
⚡ Quick Start
Create a .env file with your configuration:

ini
# For BNB Smart Chain
DEPLOYER_WALLET=0xYourWalletAddress
CHAIN_ID=56
ETHERSCAN_API_KEY=YourApiKey

# Email configuration
EMAIL_USER=your@email.com
EMAIL_PASS=your_app_password
EMAIL_TO=recipient@email.com
Run the monitor:

bash
python3 wallet_monitor.py
🛠 Configuration Options
Variable	Description	Default Value
DEPLOYER_WALLET	Wallet address to monitor	- (Required)
CHAIN_ID	Blockchain network ID	1 (Ethereum)
ETHERSCAN_API_KEY	Your Etherscan API key	- (Required)
CHECK_INTERVAL	Seconds between checks	300 (5 minutes)
EMAIL_USER	SMTP email username	- (Required)
EMAIL_PASS	SMTP email password/app password	- (Required)
EMAIL_TO	Alert recipient email	- (Required)
SMTP_SERVER	SMTP server address	smtp.gmail.com
SMTP_PORT	SMTP server port	587
📋 Supported Blockchains
Chain ID	Blockchain	Native Currency
1	Ethereum	ETH
56	BNB Smart Chain	BNB
137	Polygon	MATIC
10	Optimism	ETH
42161	Arbitrum	ETH
8453	Base	ETH
5	Goerli Testnet	ETH
11155111	Sepolia Testnet	ETH
📬 Sample Alert Email
text
CRITICAL: Funds movement detected from monitored wallet!

Transaction Hash: 0x515d2e21fee87a01252192c8d9303ab4e18c68b4dc521af118dd1c364e3ff358
Chain: Ethereum Mainnet
From: 0x95e9123ccca26aae1de8cb52563802c84a8d2636
To: 0x47b13583fa53663e610bbd1d856487081164c89a
Amount: 18.586522 ETH
Date: 2025-06-23 10:48:23

Verify transaction: https://etherscan.io/tx/0x515d2e21fee87a01252192c8d9303ab4e18c68b4dc521af118dd1c364e3ff358
☕ Support My Work
If this project helped you secure your crypto assets, consider buying me a coffee!

Cryptocurrency Donations:

Coin	Wallet Address
Verus	RPMu8QpUxvevPuTX2baVeVmt9PvYfWjURN
Bitcoin	1F5i3twCN6rMKu6KZRbNYySgP8TwzKrWgh
Ethereum	0x1F491f5d86b78865cD20379FC47FaA04E4f5ceB3
Litecoin	LZJfK7F2Sm6QahnUjZafpzWSbLqE7mp2NK
Every donation helps me maintain and improve this project!

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

⚠️ Disclaimer
This software is provided "as is" without warranty of any kind. Use at your own risk. The author is not responsible for any financial losses or security breaches resulting from the use of this tool. Always conduct your own security audits and maintain proper wallet security practices.
