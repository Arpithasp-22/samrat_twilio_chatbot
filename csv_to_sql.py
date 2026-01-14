import pandas as pd
import sqlite3

# Paths
CSV_FILE = "data/sales.csv"
DB_FILE = "data/sales.db"
TABLE_NAME = "sales"

# Load CSV
df = pd.read_csv(CSV_FILE, low_memory=False)

# Create SQLite DB
conn = sqlite3.connect(DB_FILE)

# Write to SQL
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

conn.close()

print("✅ CSV converted to SQLite successfully")
