import sqlite3

DB_PATH = "instance/recruitment.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check existing columns
cursor.execute("PRAGMA table_info(job)")
columns = [row[1] for row in cursor.fetchall()]

if "latitude" not in columns:
    cursor.execute("ALTER TABLE job ADD COLUMN latitude REAL")
    print("Added latitude column")

if "longitude" not in columns:
    cursor.execute("ALTER TABLE job ADD COLUMN longitude REAL")
    print("Added longitude column")

conn.commit()
conn.close()

print("GPS columns added successfully!")