import sqlite3
import pandas as pd
import os

def create_database_from_csv(csv_path, db_path, table_name):
    """
    Create a SQLite database and populate it with data from CSV.
    """
    # Read CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} records from {csv_path}")
    
    # Create database and table
    conn = sqlite3.connect(db_path)
    try:
        # Convert column names to match expected format if needed
        column_mapping = {
            'transaction_amount': 'transaction_amount',
            'account_balance': 'account_balance',
            'account_type': 'account_type',
            'region': 'region',
            'report_date': 'report_date'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['id', 'report_date', 'transaction_amount', 'account_type', 'account_balance', 'region']
        for col in required_cols:
            if col not in df.columns:
                if col == 'id' and 'id' not in df.columns:
                    df['id'] = range(1, len(df) + 1)
                elif col == 'report_date' and 'report_date' not in df.columns:
                    df['report_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                elif col == 'region' and 'region' not in df.columns:
                    df['region'] = 'Unknown'
                else:
                    df[col] = None
        
        # Save to database
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Created database '{db_path}' with table '{table_name}' containing {len(df)} records")
        
    finally:
        conn.close()

if __name__ == "__main__":
    # Create database from synthetic data
    create_database_from_csv(
        csv_path="../data/synthetic_data.csv",
        db_path="../data/transactions.db",
        table_name="transactions"
    )