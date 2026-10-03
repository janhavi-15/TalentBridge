import sqlite3
import pandas as pd

# =========================================================
# CONNECT TO TALENTBRIDGE DATABASE
# =========================================================

DB_PATH = "instance/recruitment.db"

conn = sqlite3.connect(DB_PATH)

# =========================================================
# READ DATA FROM DATABASE
# =========================================================

users = pd.read_sql_query("""
    SELECT
        id,
        name,
        role,
        company,
        skills,
        education,
        experience,
        location
    FROM user
""", conn)

jobs = pd.read_sql_query("""
    SELECT
        id,
        title,
        company,
        location,
        salary,
        vacancies,
        job_type,
        experience,
        education,
        skills,
        deadline,
        hr_id,
        status,
        field,
        working_days,
        timing,
        user_type,
        quick_apply,
        role,
        date_posted,
        work_mode
    FROM job
""", conn)

applications = pd.read_sql_query("""
    SELECT
        id,
        candidate_id,
        job_id,
        applied_date,
        status
    FROM application
""", conn)

conn.close()

# =========================================================
# SEPARATE CANDIDATES AND HR USERS
# =========================================================

candidates = users[users["role"] == "candidate"].copy()

hr_users = users[users["role"] == "hr"].copy()

# =========================================================
# CREATE EXCEL FILE
# =========================================================

output_file = "TalentBridge_PowerBI_Data.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    candidates.to_excel(
        writer,
        sheet_name="Candidates",
        index=False
    )

    hr_users.to_excel(
        writer,
        sheet_name="HR_Users",
        index=False
    )

    jobs.to_excel(
        writer,
        sheet_name="Jobs",
        index=False
    )

    applications.to_excel(
        writer,
        sheet_name="Applications",
        index=False
    )

print()
print("==========================================")
print("TalentBridge Power BI export completed!")
print("==========================================")
print()
print("Candidates :", len(candidates))
print("HR Users   :", len(hr_users))
print("Jobs       :", len(jobs))
print("Applications:", len(applications))
print()
print("Excel file created:")
print(output_file)