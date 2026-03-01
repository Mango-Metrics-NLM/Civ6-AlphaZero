import sqlite3
import os

db_path = r"c:\Users\iansh\OneDrive\Documents\Civ6 AlphaZero AI agent\Civ6-AlphaZero\data\tournament.db"

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("--- Tables ---")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(table[0])

for table in tables:
    table_name = table[0]
    print(f"\n--- Schema: {table_name} ---")
    cursor.execute(f"PRAGMA table_info({table_name});")
    for col in cursor.fetchall():
        print(col)
    
    print(f"\n--- Count: {table_name} ---")
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    print(cursor.fetchone()[0])

print("\n--- Recent Matches ---")
try:
    cursor.execute("SELECT * FROM matches ORDER BY timestamp DESC LIMIT 5;")
    for row in cursor.fetchall():
        print(row)
except:
    print("Could not query matches table")

conn.close()
