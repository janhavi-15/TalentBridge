import sqlite3

DB_PATH = "instance/recruitment.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

columns = {
    "working_days": "TEXT",
    "timing": "TEXT",
    "user_type": "TEXT",
    "quick_apply": "INTEGER DEFAULT 0",
    "posted_at": "TEXT"
}

existing_columns = [
    row[1]
    for row in cursor.execute("PRAGMA table_info(job)").fetchall()
]

for column, column_type in columns.items():

    if column not in existing_columns:

        cursor.execute(
            f"ALTER TABLE job ADD COLUMN {column} {column_type}"
        )

        print(f"✅ Added column: {column}")

    else:

        print(f"✓ Already exists: {column}")

conn.commit()
conn.close()

print("\n✅ Job filter database update completed!")