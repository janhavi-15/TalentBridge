import sqlite3

db_path = "instance/recruitment.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check whether field already exists
cursor.execute("PRAGMA table_info(job)")
columns = [row[1] for row in cursor.fetchall()]

if "field" not in columns:
    cursor.execute("ALTER TABLE job ADD COLUMN field TEXT")
    conn.commit()
    print("✅ field column added successfully!")
else:
    print("✅ field column already exists!")

conn.close()