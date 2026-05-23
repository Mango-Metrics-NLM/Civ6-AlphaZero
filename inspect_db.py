"""Inspect tournament database structure and contents.

Usage:
    python inspect_db.py [path/to/tournament.db]
"""

from __future__ import annotations

import os
import sqlite3
import sys


def inspect_database(db_path: str) -> None:
    """Inspect a tournament SQLite database.

    Args:
        db_path: Path to the SQLite database file.
    """
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        sys.exit(1)

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
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")  # noqa: S608
        print(cursor.fetchone()[0])

    print("\n--- Recent Matches ---")
    try:
        cursor.execute("SELECT * FROM matches ORDER BY timestamp DESC LIMIT 5;")
        for row in cursor.fetchall():
            print(row)
    except sqlite3.OperationalError:
        print("Could not query matches table")

    conn.close()


if __name__ == "__main__":
    _default_path = os.path.join("data", "tournament.db")
    _db_path = sys.argv[1] if len(sys.argv) > 1 else _default_path
    inspect_database(_db_path)
