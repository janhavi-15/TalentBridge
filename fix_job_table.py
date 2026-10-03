import sqlite3
import os

# =========================================================
# DATABASE
# =========================================================

db_path = os.path.join(
    os.path.dirname(__file__),
    "instance",
    "recruitment.db"
)

print("Database:")
print(db_path)
print()

# =========================================================
# CONNECT
# =========================================================

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# =========================================================
# CHECK EXISTING COLUMNS
# =========================================================

cursor.execute("PRAGMA table_info(job)")

columns = [row[1] for row in cursor.fetchall()]

print("Current columns:")
print(columns)
print()

# =========================================================
# ROLE
# =========================================================

if "role" not in columns:

    cursor.execute("""
        ALTER TABLE job
        ADD COLUMN role TEXT
    """)

    print("✅ Added role")

else:

    print("✅ role already exists")


# =========================================================
# DATE POSTED
# =========================================================

if "date_posted" not in columns:

    cursor.execute("""
        ALTER TABLE job
        ADD COLUMN date_posted DATETIME
    """)

    print("✅ Added date_posted")

else:

    print("✅ date_posted already exists")


# =========================================================
# QUICK APPLY
# =========================================================

if "quick_apply" not in columns:

    cursor.execute("""
        ALTER TABLE job
        ADD COLUMN quick_apply INTEGER DEFAULT 0
    """)

    print("✅ Added quick_apply")

else:

    print("✅ quick_apply already exists")


# =========================================================
# COMMIT
# =========================================================

conn.commit()

# =========================================================
# FINAL CHECK
# =========================================================

cursor.execute("PRAGMA table_info(job)")

print()
print("==========================================")
print("FINAL JOB TABLE")
print("==========================================")

for row in cursor.fetchall():

    print(
        row[0],
        row[1],
        row[2]
    )

conn.close()

print()
print("==========================================")
print("✅ JOB TABLE FIXED SUCCESSFULLY")
print("==========================================")