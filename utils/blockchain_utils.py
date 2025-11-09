# utils/blockchain_utils.py
import hashlib
import json
from web3 import Web3
import datetime

class BlockchainUtils:
    @staticmethod
    def hash_address(address):
        return hashlib.sha256(address.encode()).hexdigest()
    
    @staticmethod
    def validate_ethereum_address(address):
        return Web3.is_address(address)
    
    @staticmethod
    def wei_to_ether(wei_value):
        return wei_value / 1e18
    
    @staticmethod
    def ether_to_wei(ether_value):
        return int(ether_value * 1e18)
    
    @staticmethod
    def calculate_transaction_fee(gas_used, gas_price):
        return gas_used * gas_price
    
    @staticmethod
    def timestamp_to_datetime(timestamp):
        return datetime.datetime.fromtimestamp(timestamp)
    
    @staticmethod
    def generate_address_fingerprint(address, transactions):
        fingerprint_data = {
            'address': address,
            'transaction_count': len(transactions),
            'total_volume': sum(tx['value'] for tx in transactions),
            'avg_transaction_value': np.mean([tx['value'] for tx in transactions]) if transactions else 0,
            'first_seen': min(tx['timestamp'] for tx in transactions) if transactions else 0,
            'last_seen': max(tx['timestamp'] for tx in transactions) if transactions else 0
        }
        return hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()