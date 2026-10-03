import sqlite3
import os

# =========================================================
# DATABASE PATH
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
# GET EXISTING COLUMNS
# =========================================================

cursor.execute("PRAGMA table_info(job)")

columns = [row[1] for row in cursor.fetchall()]

print("Existing job columns:")
print(columns)
print()

# =========================================================
# ADD ROLE
# =========================================================

if "role" not in columns:

    print("Adding role column...")

    cursor.execute("""
        ALTER TABLE job
        ADD COLUMN role TEXT
    """)

    print("✅ role added")

else:

    print("✅ role already exists")


# =========================================================
# ADD DATE_POSTED
# =========================================================

if "date_posted" not in columns:

    print("Adding date_posted column...")

    cursor.execute("""
        ALTER TABLE job
        ADD COLUMN date_posted TEXT
    """)

    print("✅ date_posted added")

else:

    print("✅ date_posted already exists")


# =========================================================
# COPY OLD posted_at DATA
# =========================================================

cursor.execute("""
    UPDATE job
    SET date_posted = posted_at
    WHERE date_posted IS NULL
      AND posted_at IS NOT NULL
""")

print("✅ Existing posted_at values copied to date_posted")


# =========================================================
# COMMIT
# =========================================================

conn.commit()

# =========================================================
# CHECK FINAL TABLE
# =========================================================

cursor.execute("PRAGMA table_info(job)")

print()
print("FINAL JOB TABLE:")
print()

for column in cursor.fetchall():

    print(
        column[0],
        column[1],
        column[2]
    )

conn.close()

print()
print("====================================")
print("✅ DATABASE MIGRATION COMPLETED")
print("====================================")