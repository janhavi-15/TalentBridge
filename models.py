from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime


db = SQLAlchemy()


# =========================================================
# USER MODEL
# =========================================================

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        nullable=False
    )

    mobile = db.Column(
        db.String(20)
    )

    company = db.Column(
        db.String(150)
    )

    skills = db.Column(
        db.String(300)
    )

    education = db.Column(
        db.String(200)
    )

    experience = db.Column(
        db.String(100)
    )

    location = db.Column(
        db.String(100)
    )

    # Candidate resume
    resume = db.Column(
        db.String(300),
        nullable=True
    )

    # Candidate profile photo
    profile_photo = db.Column(
        db.String(255),
        nullable=True
    )


# =========================================================
# JOB MODEL
# =========================================================

class Job(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # -----------------------------------------------------
    # BASIC JOB INFORMATION
    # -----------------------------------------------------

    title = db.Column(
        db.String(200),
        nullable=False
    )

    company = db.Column(
        db.String(200),
        nullable=False
    )

    location = db.Column(
        db.String(200),
        nullable=False
    )

    latitude = db.Column(
        db.Float,
        nullable=True
    )

    longitude = db.Column(
        db.Float,
        nullable=True
    )

    salary = db.Column(
        db.String(100),
        nullable=False
    )

    vacancies = db.Column(
        db.Integer,
        nullable=False
    )

    job_type = db.Column(
        db.String(50),
        nullable=False
    )

    experience = db.Column(
        db.String(100),
        nullable=False
    )

    education = db.Column(
        db.String(150),
        nullable=False
    )

    skills = db.Column(
        db.Text,
        nullable=False
    )

    incentives = db.Column(
        db.Text
    )

    deadline = db.Column(
        db.String(50),
        nullable=False
    )

    description = db.Column(
        db.Text,
        nullable=False
    )

    # -----------------------------------------------------
    # ML PREDICTED FIELD
    # -----------------------------------------------------

    field = db.Column(
        db.String(100)
    )

    # -----------------------------------------------------
    # JOB FILTER FIELDS
    # -----------------------------------------------------

    role = db.Column(
        db.String(100)
    )

    working_days = db.Column(
        db.String(100)
    )

    timing = db.Column(
        db.String(100)
    )

    user_type = db.Column(
        db.String(100),
        default="All"
    )

    quick_apply = db.Column(
        db.Boolean,
        default=False
    )

    work_mode = db.Column(
        db.String(50),
        nullable=True
    )

    # -----------------------------------------------------
    # DATE POSTED
    # -----------------------------------------------------

    date_posted = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    # -----------------------------------------------------
    # HR
    # -----------------------------------------------------

    hr_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id")
    )

    # -----------------------------------------------------
    # JOB STATUS
    # -----------------------------------------------------

    status = db.Column(
        db.String(30),
        default="Approved",
        nullable=False
    )


# =========================================================
# APPLICATION MODEL
# =========================================================

class Application(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    candidate_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id')
    )

    job_id = db.Column(
        db.Integer,
        db.ForeignKey('job.id')
    )

    resume = db.Column(
        db.String(300)
    )

    cover_letter = db.Column(
        db.Text
    )

    applied_date = db.Column(
        db.DateTime,
        default=db.func.now()
    )

    status = db.Column(
        db.String(30),
        default="Applied"
    )


# =========================================================
# SAVED JOB MODEL
# =========================================================

class SavedJob(db.Model):

    __tablename__ = "saved_job"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    candidate_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "user.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    job_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "job.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    candidate = db.relationship(
        "User",
        backref=db.backref(
            "saved_jobs",
            cascade="all, delete-orphan"
        )
    )

    job = db.relationship(
        "Job",
        backref=db.backref(
            "saved_jobs",
            cascade="all, delete-orphan"
        )
    )


# =========================================================
# ADMIN SETTINGS MODEL
# =========================================================

class AdminSettings(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # -----------------------------------------------------
    # ACCOUNT SETTINGS
    # -----------------------------------------------------

    admin_name = db.Column(
        db.String(100),
        default="Administrator"
    )

    admin_email = db.Column(
        db.String(150),
        default="admin@gmail.com"
    )

    admin_password = db.Column(
        db.String(255),
        default="admin123"
    )

    # -----------------------------------------------------
    # SECURITY SETTINGS
    # -----------------------------------------------------

    two_factor = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

    login_alerts = db.Column(
        db.Boolean,
        default=True
    )

    session_security = db.Column(
        db.Boolean,
        default=True
    )

    # -----------------------------------------------------
    # NOTIFICATION SETTINGS
    # -----------------------------------------------------

    candidate_notifications = db.Column(
        db.Boolean,
        default=True
    )

    hr_notifications = db.Column(
        db.Boolean,
        default=True
    )

    job_notifications = db.Column(
        db.Boolean,
        default=True
    )

    application_notifications = db.Column(
        db.Boolean,
        default=True
    )

    # -----------------------------------------------------
    # RECRUITMENT CONTROLS
    # -----------------------------------------------------

    allow_applications = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )

    allow_job_posting = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )

    job_approval = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

    hr_approval = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

    # -----------------------------------------------------
    # PLATFORM SETTINGS
    # -----------------------------------------------------

    website_name = db.Column(
        db.String(150),
        default="TalentBridge"
    )

    support_email = db.Column(
        db.String(150),
        default="support@talentbridge.com"
    )

    contact_number = db.Column(
        db.String(50)
    )

    location = db.Column(
        db.String(200)
    )

    description = db.Column(
        db.Text
    )


