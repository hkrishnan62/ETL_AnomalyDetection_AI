import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

def add_regulatory_anomalies(db_path, table_name):
    """
    Add records with various regulatory anomalies to the database.
    """
    conn = sqlite3.connect(db_path)
    
    try:
        # Get the current max ID
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(id) FROM {table_name}")
        max_id = cursor.fetchone()[0] or 0
        
        # Create anomalous records
        anomalous_records = []
        
        # 1. Money Laundering patterns - large round number transactions
        for i in range(10):
            anomalous_records.append({
                'id': max_id + i + 1,
                'report_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_amount': random.choice([10000, 25000, 50000, 100000]),  # Large round numbers
                'account_balance': random.randint(100000, 500000),
                'risk_score': random.randint(80, 100),
                'account_age': random.randint(1, 12),
                'account_type': random.choice(['Retail', 'Corporate']),
                'region': 'HighRisk'  # Invalid region
            })
        
        # 2. Structuring - transactions just under reporting threshold
        for i in range(5):
            anomalous_records.append({
                'id': max_id + i + 11,
                'report_date': (datetime.now() - timedelta(hours=random.randint(1, 24))).strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_amount': 9900 + random.randint(1, 99),  # Just under $10k
                'account_balance': random.randint(50000, 200000),
                'risk_score': random.randint(60, 85),
                'account_age': random.randint(1, 24),
                'account_type': 'Suspicious',  # Invalid account type
                'region': random.choice(['APAC', 'EU', 'US'])
            })
        
        # 3. Negative amounts and zero balances (accounting errors)
        for i in range(3):
            anomalous_records.append({
                'id': max_id + i + 16,
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_amount': -random.randint(1000, 10000),  # Negative amounts
                'account_balance': 0,  # Zero balance
                'risk_score': random.randint(90, 100),
                'account_age': random.randint(1, 60),
                'account_type': random.choice(['Retail', 'Corporate', 'Investment']),
                'region': random.choice(['APAC', 'EU', 'US'])
            })
        
        # 4. High-risk region transactions with large amounts
        high_risk_regions = ['SanctionsLand', 'RestrictedZone', 'WatchList']
        for i in range(5):
            anomalous_records.append({
                'id': max_id + i + 19,
                'report_date': (datetime.now() - timedelta(days=random.randint(1, 7))).strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_amount': random.randint(50000, 200000),
                'account_balance': random.randint(100000, 1000000),
                'risk_score': random.randint(95, 100),
                'account_age': random.randint(1, 6),  # Very new accounts
                'account_type': random.choice(['Retail', 'Corporate']),
                'region': random.choice(high_risk_regions)
            })
        
        # 5. Records with missing required fields
        for i in range(3):
            record = {
                'id': max_id + i + 24,
                'report_date': None,  # Missing date
                'transaction_amount': random.randint(1000, 50000),
                'account_balance': None,  # Missing balance
                'risk_score': random.randint(1, 100),
                'account_age': random.randint(1, 120),
                'account_type': None,  # Missing account type
                'region': random.choice(['APAC', 'EU', 'US'])
            }
            anomalous_records.append(record)
        
        # 6. Unusual patterns - same amount multiple times (potential scripting)
        base_amount = 7777  # Unusual repeating amount
        for i in range(8):
            anomalous_records.append({
                'id': max_id + i + 27,
                'report_date': (datetime.now() - timedelta(minutes=i*30)).strftime('%Y-%m-%d %H:%M:%S'),
                'transaction_amount': base_amount,
                'account_balance': random.randint(10000, 100000),
                'risk_score': random.randint(70, 90),
                'account_age': random.randint(1, 36),
                'account_type': random.choice(['Retail', 'Corporate', 'Investment']),
                'region': random.choice(['APAC', 'EU', 'US'])
            })
        
        # Insert records
        df_anomalous = pd.DataFrame(anomalous_records)
        df_anomalous.to_sql(table_name, conn, if_exists='append', index=False)
        
        print(f"Added {len(anomalous_records)} anomalous records to {table_name}")
        print("Anomalies include:")
        print("- Large round number transactions (money laundering)")
        print("- Structuring (under reporting thresholds)")
        print("- Negative amounts and zero balances")
        print("- High-risk region transactions")
        print("- Missing required fields")
        print("- Unusual repeating amounts")
        
    finally:
        conn.close()

if __name__ == "__main__":
    add_regulatory_anomalies(
        db_path="../data/transactions.db",
        table_name="transactions"
    )