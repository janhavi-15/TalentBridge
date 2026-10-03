import sqlite3

conn = sqlite3.connect("recruitment.db")
cursor = conn.cursor()

try:
    cursor.execute(
        "ALTER TABLE job ADD COLUMN latitude REAL"
    )
    print("latitude column added")
except sqlite3.OperationalError:
    print("latitude column already exists")

try:
    cursor.execute(
        "ALTER TABLE job ADD COLUMN longitude REAL"
    )
    print("longitude column added")
except sqlite3.OperationalError:
    print("longitude column already exists")

conn.commit()
conn.close()

print("Database updated successfully!")