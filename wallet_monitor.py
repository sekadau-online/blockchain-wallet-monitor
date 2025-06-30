import requests
import time
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import logging
import json
import ssl

# === SETUP LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# === LOAD ENV ===
load_dotenv()

DEPLOYER_WALLET = os.getenv('DEPLOYER_WALLET')
CHAIN_ID = os.getenv('CHAIN_ID', '1')  # Default to Ethereum mainnet
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 300))  # Default 300 seconds

SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
EMAIL_TO = os.getenv('EMAIL_TO')

# Single API endpoint for all chains
ETHERSCAN_V2_ENDPOINT = "https://api.etherscan.io/v2/api"

# Chain configuration for display and value conversion
CHAIN_CONFIG = {
    '1': {
        'name': 'Ethereum Mainnet',
        'symbol': 'ETH',
        'explorer': 'https://etherscan.io',
        'value_divisor': 10**18
    },
    '56': {
        'name': 'BNB Smart Chain',
        'symbol': 'BNB',
        'explorer': 'https://bscscan.com',
        'value_divisor': 10**18
    },
    '137': {
        'name': 'Polygon',
        'symbol': 'MATIC',
        'explorer': 'https://polygonscan.com',
        'value_divisor': 10**18
    },
    '10': {
        'name': 'Optimism',
        'symbol': 'ETH',
        'explorer': 'https://optimistic.etherscan.io',
        'value_divisor': 10**18
    },
    '42161': {
        'name': 'Arbitrum',
        'symbol': 'ETH',
        'explorer': 'https://arbiscan.io',
        'value_divisor': 10**18
    },
    '8453': {
        'name': 'Base',
        'symbol': 'ETH',
        'explorer': 'https://basescan.org',
        'value_divisor': 10**18
    },
    '5': {
        'name': 'Goerli Testnet',
        'symbol': 'ETH',
        'explorer': 'https://goerli.etherscan.io',
        'value_divisor': 10**18
    },
    '11155111': {
        'name': 'Sepolia Testnet',
        'symbol': 'ETH',
        'explorer': 'https://sepolia.etherscan.io',
        'value_divisor': 10**18
    }
}

# Validate chain configuration
chain_cfg = CHAIN_CONFIG.get(CHAIN_ID)
if not chain_cfg:
    logger.error(f"Unsupported CHAIN_ID: {CHAIN_ID}")
    exit(1)

# Use Etherscan API key for all chains
API_KEY = os.getenv('ETHERSCAN_API_KEY')
if not API_KEY:
    logger.error(f"Missing ETHERSCAN_API_KEY in .env")
    exit(1)

# Validate required settings
if not DEPLOYER_WALLET:
    logger.error("DEPLOYER_WALLET is not set in .env")
    exit(1)

if not all([EMAIL_USER, EMAIL_PASS, EMAIL_TO]):
    logger.error("Missing email configuration in .env")
    exit(1)

ALREADY_ALERTED = set()

def send_email_alert(tx_data, chain_cfg):
    try:
        # Convert values
        value_wei = int(tx_data.get('value', 0))
        value_main = value_wei / chain_cfg['value_divisor']
        
        # Format date
        timestamp = int(tx_data.get('timeStamp', time.time()))
        tx_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        
        subject = f'ALERT: Outgoing Transaction on {chain_cfg["name"]}!'
        body = (
            f'CRITICAL: Funds movement detected from monitored wallet!\n\n'
            f'Transaction Hash: {tx_data.get("hash", "Unknown")}\n'
            f'Chain: {chain_cfg["name"]}\n'
            f'From: {tx_data.get("from", "Unknown")}\n'
            f'To: {tx_data.get("to", "Unknown")}\n'
            f'Amount: {value_main:.6f} {chain_cfg["symbol"]}\n'
            f'Date: {tx_date}\n\n'
            f'Verify transaction: {chain_cfg["explorer"]}/tx/{tx_data.get("hash", "")}'
        )
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_TO

        context = ssl.create_default_context()
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, [EMAIL_TO], msg.as_string())
        
        logger.info(f"Email alert sent for TX: {tx_data.get('hash', 'unknown')}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication failed: {e.smtp_error.decode()}")
        return False
    except Exception as e:
        logger.error(f"Failed to send email alert: {str(e)}")
        return False

def get_transactions():
    """Fetch transactions using Etherscan V2 API for all chains"""
    params = {
        'chainid': CHAIN_ID,
        'module': 'account',
        'action': 'txlist',
        'address': DEPLOYER_WALLET,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'desc',
        'apikey': API_KEY
    }
    
    try:
        logger.info(f"Requesting transactions from Etherscan V2 API for chain {CHAIN_ID}")
        response = requests.get(ETHERSCAN_V2_ENDPOINT, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Log raw response for debugging
        logger.debug(f"API response: {json.dumps(data, indent=2)}")
        
        # Handle API response
        if str(data.get('status')) != '1' or data.get('message') != 'OK':
            error_msg = data.get('message', 'No error message')
            result_msg = data.get('result', 'No additional info')
            logger.error(f"API Error: {error_msg} - {result_msg}")
            return []
                
        transactions = data.get('result', [])
        
        if not isinstance(transactions, list):
            logger.error(f"Unexpected API response format: {type(transactions)}")
            return []
            
        logger.info(f"Received {len(transactions)} transactions")
        return transactions
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    
    return []

def check_transactions():
    try:
        transactions = get_transactions()
        new_alerts = 0
        
        for tx in transactions:
            tx_hash = tx.get('hash', '')
            if not tx_hash or tx_hash in ALREADY_ALERTED:
                continue
                
            # Check if outgoing transaction with value
            if (tx.get('from', '').lower() == DEPLOYER_WALLET.lower() and 
                int(tx.get('value', 0)) > 0):
                logger.warning(f"OUTGOING TX DETECTED: {tx_hash}")
                if send_email_alert(tx, chain_cfg):
                    ALREADY_ALERTED.add(tx_hash)
                    new_alerts += 1
        
        logger.info(f"Checked {len(transactions)} transactions. New alerts: {new_alerts}")
        return new_alerts
        
    except Exception as e:
        logger.error(f"Error checking transactions: {str(e)}")
        return 0

def main():
    logger.info(f"Starting Blockchain Monitor (Etherscan V2 Multichain API)")
    logger.info(f"Chain: {chain_cfg['name']} (ID: {CHAIN_ID})")
    logger.info(f"Wallet: {DEPLOYER_WALLET}")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
    logger.info(f"API Endpoint: {ETHERSCAN_V2_ENDPOINT}")
    
    try:
        while True:
            start_time = time.time()
            alerts = check_transactions()
            
            elapsed = time.time() - start_time
            sleep_time = max(1, CHECK_INTERVAL - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")

if __name__ == '__main__':
    main()
