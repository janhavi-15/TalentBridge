from app import app
from models import db


with app.app_context():

    print("=" * 60)
    print("FIXING JOB DATE_POSTED VALUES")
    print("=" * 60)

    # Find all existing values
    rows = db.session.execute(
        db.text("""
            SELECT id, date_posted, typeof(date_posted)
            FROM job
        """)
    ).fetchall()

    print(f"Total jobs found: {len(rows)}")

    for row in rows:
        print(
            f"Job ID: {row[0]} | "
            f"date_posted: {row[1]} | "
            f"type: {row[2]}"
        )

    print("=" * 60)

    # SQLite date_posted values that are not stored as text
    db.session.execute(
        db.text("""
            UPDATE job
            SET date_posted = NULL
            WHERE date_posted IS NOT NULL
              AND typeof(date_posted) != 'text'
        """)
    )

    db.session.commit()

    print("✅ Invalid date_posted values repaired.")

    print("=" * 60)