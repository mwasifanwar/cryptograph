# data/blockchain_loader.py
import pandas as pd
import numpy as np
from web3 import Web3
import requests
import time

class BlockchainDataLoader:
    def __init__(self, rpc_url):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
    def get_transaction_data(self, tx_hash):
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            tx_data = {
                'hash': tx_hash,
                'from': tx['from'],
                'to': tx['to'],
                'value': tx['value'],
                'gas': tx['gas'],
                'gas_price': tx['gasPrice'],
                'input': tx['input'],
                'block_number': tx['blockNumber'],
                'timestamp': self.w3.eth.get_block(tx['blockNumber'])['timestamp'],
                'status': receipt['status'],
                'gas_used': receipt['gasUsed']
            }
            return tx_data
        except Exception as e:
            return None

    def get_address_transactions(self, address, limit=1000):
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset={limit}&sort=asc&apikey=YourApiKeyToken"
        
        try:
            response = requests.get(url)
            data = response.json()
            if data['status'] == '1':
                return data['result']
            return []
        except Exception as e:
            return []

    def generate_synthetic_data(self, num_transactions=10000):
        np.random.seed(42)
        
        transactions = []
        addresses = [f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}" for _ in range(1000)]
        
        for i in range(num_transactions):
            tx = {
                'hash': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}",
                'from': np.random.choice(addresses),
                'to': np.random.choice(addresses),
                'value': np.random.exponential(1000000000000000000),
                'gas': np.random.randint(21000, 1000000),
                'gas_price': np.random.exponential(1000000000),
                'block_number': np.random.randint(15000000, 17000000),
                'timestamp': np.random.randint(1600000000, 1700000000),
                'status': 1,
                'gas_used': np.random.randint(21000, 500000)
            }
            
            if np.random.random() < 0.05:
                tx['value'] *= 100
                tx['gas_price'] *= 10
                
            transactions.append(tx)
            
        return transactions

    def create_transaction_dataframe(self, transactions):
        df = pd.DataFrame(transactions)
        
        df['value_eth'] = df['value'] / 1e18
        df['gas_price_gwei'] = df['gas_price'] / 1e9
        df['input_length'] = df['input'].apply(len)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df