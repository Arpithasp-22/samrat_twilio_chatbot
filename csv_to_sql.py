"""
csv_to_sql.py
-------------
Utility script to load sales data from a CSV file into a SQLite database.
This script is typically run once during setup or whenever the sales dataset
needs to be refreshed.

Main responsibilities:
- Read sales data from CSV
- Create SQLite tables if not present
- Insert cleaned sales records into the database
"""


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
