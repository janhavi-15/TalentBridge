import sqlite3
import os

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

# =========================================================
# DATABASE PATH
# =========================================================
# Flask SQLite database is normally inside:
# instance/recruitment.db
#
# If your database is there, this is correct.

DB_PATH = os.path.join(
    PROJECT_ROOT,
    "instance",
    "recruitment.db"
)

print("====================================")
print("PROJECT ROOT:")
print(PROJECT_ROOT)

print("\nDATABASE PATH:")
print(DB_PATH)

print("\nDATABASE EXISTS:")
print(os.path.exists(DB_PATH))

print("====================================")


# =========================================================
# CHECK DATABASE
# =========================================================

if not os.path.exists(DB_PATH):

    print("\n❌ Database not found!")
    print("\nExpected database location:")
    print(DB_PATH)

    print("\nPlease check where recruitment.db actually exists.")

    exit()


# =========================================================
# CONNECT DATABASE
# =========================================================

connection = sqlite3.connect(DB_PATH)

cursor = connection.cursor()


# =========================================================
# CHECK USER TABLE
# =========================================================

cursor.execute("PRAGMA table_info(user)")

columns = cursor.fetchall()

print("\nCurrent USER table columns:")

for column in columns:
    print(column)


# =========================================================
# CHECK PROFILE PHOTO COLUMN
# =========================================================

column_names = [
    column[1]
    for column in columns
]


if "profile_photo" not in column_names:

    print("\nAdding profile_photo column...")

    cursor.execute("""
        ALTER TABLE user
        ADD COLUMN profile_photo TEXT
    """)

    connection.commit()

    print("✅ profile_photo column added successfully.")

else:

    print("\n✅ profile_photo column already exists.")


# =========================================================
# CLOSE
# =========================================================

connection.close()

print("\n====================================")
print("DONE")
print("====================================")