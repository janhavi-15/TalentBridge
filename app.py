from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import LoginManager
from models import db, User, Job, Application, SavedJob, AdminSettings
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from itsdangerous import URLSafeTimedSerializer
from werkzeug.security import generate_password_hash


import joblib
import os
import random
import uuid
import time
import re
import math
from datetime import datetime, timedelta

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity





MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "ml",
    "job_field_model.pkl"
)

job_field_model = joblib.load(MODEL_PATH)






def predict_job_field(title, description, skills):

    text = (
        str(title) + " " +
        str(description) + " " +
        str(skills)
    )

    prediction = job_field_model.predict([text])[0]

    return prediction



def extract_resume_text(file_path):

    text = ""

    try:

        with open(file_path, "rb") as file:

            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + " "

    except Exception as e:

        print("Resume extraction error:", e)

    return text








# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

def normalize_text(text):
    """Convert text into clean lowercase searchable text."""
    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_skill_list(text):
    """Convert comma/semicolon/pipe separated skills into a list."""

    if not text:
        return []

    text = normalize_text(text)

    skills = re.split(r"[,;/|]+", text)

    return [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]


def calculate_skill_score(candidate_skills, job_skills):
    """
    Calculate percentage of job skills matched by candidate.
    """

    candidate_text = normalize_text(candidate_skills)

    job_skill_list = get_skill_list(job_skills)

    if not candidate_text or not job_skill_list:
        return 0.0

    matched = 0

    for skill in job_skill_list:

        # Exact phrase match
        if skill in candidate_text:
            matched += 1
            continue

        # Word-level matching
        skill_words = skill.split()

        if all(word in candidate_text for word in skill_words):
            matched += 1

    score = (matched / len(job_skill_list)) * 100

    return round(score, 2)


def calculate_text_similarity(resume_text, job):
    """
    TF-IDF similarity between resume and job information.
    """

    resume_text = normalize_text(resume_text)

    job_text = " ".join([
        normalize_text(job.title),
        normalize_text(job.skills),
        normalize_text(job.description),
        normalize_text(job.role),
        normalize_text(job.field),
        normalize_text(job.education),
        normalize_text(job.experience)
    ])

    if not resume_text or not job_text:
        return 0.0

    try:

        documents = [
            resume_text,
            job_text
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(float(similarity * 100), 2)

    except Exception as e:

        print("TF-IDF ERROR:", e)

        return 0.0


def calculate_field_score(candidate_field, job_field):
    """
    Compare predicted candidate field with job field.
    """

    if not candidate_field or not job_field:
        return 0.0

    candidate_field = normalize_text(candidate_field)
    job_field = normalize_text(job_field)

    if not candidate_field or not job_field:
        return 0.0

    if candidate_field == job_field:
        return 100.0

    # Partial field match
    if candidate_field in job_field:
        return 75.0

    if job_field in candidate_field:
        return 75.0

    return 0.0


def calculate_education_score(candidate_education, job_education):
    """
    Basic education compatibility score.
    """

    candidate_education = normalize_text(candidate_education)
    job_education = normalize_text(job_education)

    if not candidate_education or not job_education:
        return 0.0

    # Direct match
    if candidate_education in job_education:
        return 100.0

    if job_education in candidate_education:
        return 100.0

    education_groups = {

        "10th": [
            "10th",
            "ssc"
        ],

        "12th": [
            "12th",
            "hsc"
        ],

        "diploma": [
            "diploma"
        ],

        "graduate": [
            "graduate",
            "b.e",
            "be",
            "b.tech",
            "btech",
            "b.sc",
            "bsc",
            "b.com",
            "bcom",
            "bca",
            "bba"
        ],

        "post graduate": [
            "post graduate",
            "postgraduate",
            "mca",
            "mba",
            "m.tech",
            "mtech",
            "m.sc",
            "msc",
            "m.com",
            "mcom"
        ]
    }

    candidate_level = None
    job_level = None

    for level, keywords in education_groups.items():

        for keyword in keywords:

            if keyword in candidate_education:
                candidate_level = level
                break

        if candidate_level:
            break

    for level, keywords in education_groups.items():

        for keyword in keywords:

            if keyword in job_education:
                job_level = level
                break

        if job_level:
            break

    levels = {
        "10th": 1,
        "12th": 2,
        "diploma": 3,
        "graduate": 4,
        "post graduate": 5
    }

    if candidate_level and job_level:

        if levels[candidate_level] >= levels[job_level]:
            return 100.0

        return 0.0

    return 0.0


def calculate_experience_score(
    candidate_experience,
    job_experience
):
    """
    Basic experience compatibility score.
    """

    candidate_experience = normalize_text(
        candidate_experience
    )

    job_experience = normalize_text(
        job_experience
    )

    if not candidate_experience or not job_experience:
        return 0.0

    if "fresher" in job_experience:

        if "fresher" in candidate_experience:
            return 100.0

        if "0" in candidate_experience:
            return 100.0

    if candidate_experience in job_experience:
        return 100.0

    # Extract numbers from candidate experience
    candidate_numbers = re.findall(
        r"\d+(?:\.\d+)?",
        candidate_experience
    )

    job_numbers = re.findall(
        r"\d+(?:\.\d+)?",
        job_experience
    )

    if candidate_numbers and job_numbers:

        try:

            candidate_years = float(
                candidate_numbers[0]
            )

            job_years = float(
                job_numbers[0]
            )

            if candidate_years >= job_years:
                return 100.0

        except (ValueError, TypeError):
            pass

    return 0.0








# =========================================================
# JOB RECOMMENDATION / MATCHING ENGINE
# =========================================================

def normalize_skill(skill):
    """
    Convert skill into a standard format.
    """

    if not skill:
        return ""

    skill = str(skill).lower().strip()

    # Replace common symbols with spaces
    skill = skill.replace("-", " ")
    skill = skill.replace("_", " ")

    # Remove extra spaces
    skill = re.sub(r"\s+", " ", skill)

    return skill.strip()


def extract_resume_skills(resume_text):
    """
    Extract important skills/keywords from resume text.

    This does not depend on candidate.skills from database.
    It directly uses the uploaded resume.
    """

    if not resume_text:
        return set()

    text = normalize_skill(resume_text)

    # Important finance/accounting skills
    known_skills = [

        # Finance / Accounting
        "accounting",
        "financial accounting",
        "finance",
        "financial analysis",
        "bookkeeping",
        "book keeping",
        "tally",
        "tally prime",
        "tally erp",
        "gst",
        "taxation",
        "income tax",
        "accounts payable",
        "accounts receivable",
        "bank reconciliation",
        "financial statements",
        "balance sheet",
        "profit and loss",
        "profit loss",
        "ledger",
        "trial balance",
        "payroll",
        "invoicing",
        "invoice",
        "audit",
        "auditing",
        "ms excel",
        "excel",

        # Programming
        "python",
        "java",
        "javascript",
        "c",
        "c++",
        "sql",
        "mysql",
        "mongodb",
        "flask",
        "django",
        "html",
        "css",
        "react",
        "node",
        "nodejs",

        # Data / ML
        "machine learning",
        "deep learning",
        "data analysis",
        "data analytics",
        "pandas",
        "numpy",
        "scikit learn",
        "sklearn",
        "power bi",
        "tableau",
        "statistics",

        # Sales / Marketing
        "sales",
        "marketing",
        "digital marketing",
        "customer service",
        "communication",
        "business development",
        "lead generation",
        "crm",

        # HR
        "recruitment",
        "human resources",
        "hr",
        "talent acquisition",
        "payroll",
        "employee relations",

        # Healthcare
        "nursing",
        "patient care",
        "clinical",
        "healthcare",
        "medical",
        "hospital",

        # Education
        "teaching",
        "training",
        "lesson planning",
        "classroom management",

        # Design
        "graphic design",
        "photoshop",
        "illustrator",
        "figma",
        "canva",
        "ui design",
        "ux design",

        # Logistics
        "logistics",
        "supply chain",
        "inventory",
        "warehouse",
        "transportation"
    ]

    found_skills = set()

    for skill in known_skills:

        normalized = normalize_skill(skill)

        if normalized in text:
            found_skills.add(normalized)

    return found_skills


def get_job_skills(job_skills):
    """
    Convert Job.skills into a clean set.
    """

    if not job_skills:
        return set()

    skills = re.split(
        r"[,;/|]+",
        str(job_skills)
    )

    cleaned = set()

    for skill in skills:

        skill = normalize_skill(skill)

        if skill:
            cleaned.add(skill)

    return cleaned


def calculate_skill_match(resume_text, job_skills):
    """
    Compare resume skills with job required skills.

    Example:

    Resume:
        tally, gst, accounting, excel

    Job:
        tally, gst, accounting, excel

    Result:
        100%
    """

    resume_skills = extract_resume_skills(
        resume_text
    )

    required_skills = get_job_skills(
        job_skills
    )

    if not required_skills:
        return 0.0, set(), set()

    matched_skills = set()

    for job_skill in required_skills:

        # Exact match
        if job_skill in resume_skills:

            matched_skills.add(
                job_skill
            )

            continue

        # Partial matching
        for resume_skill in resume_skills:

            if (
                job_skill in resume_skill
                or resume_skill in job_skill
            ):

                matched_skills.add(
                    job_skill
                )

                break

    score = (
        len(matched_skills)
        /
        len(required_skills)
    ) * 100

    return (
        round(score, 2),
        resume_skills,
        matched_skills
    )


def calculate_job_match(
    resume_text,
    job,
    candidate_field=None
):
    """
    FINAL JOB RECOMMENDATION SCORE.

    IMPORTANT:
    The argument order is intentionally:

        resume_text
        job
        candidate_field

    because the rest of your existing app already
    calls the function this way.
    """

    try:

        # =================================================
        # 1. SKILL MATCH
        # =================================================

        skill_score, resume_skills, matched_skills = (
            calculate_skill_match(
                resume_text,
                job.skills
            )
        )

        # =================================================
        # 2. FIELD MATCH
        # =================================================

        field_score = 0.0

        if candidate_field and job.field:

            candidate_field_clean = normalize_skill(
                candidate_field
            )

            job_field_clean = normalize_skill(
                job.field
            )

            if (
                candidate_field_clean ==
                job_field_clean
            ):

                field_score = 100.0

            elif (
                candidate_field_clean in
                job_field_clean
            ) or (
                job_field_clean in
                candidate_field_clean
            ):

                field_score = 75.0

        # =================================================
        # 3. TITLE MATCH
        # =================================================

        title_score = 0.0

        resume_lower = normalize_skill(
            resume_text
        )

        job_title = normalize_skill(
            job.title
        )

        if job_title:

            title_words = job_title.split()

            matched_title_words = 0

            for word in title_words:

                if len(word) >= 3 and word in resume_lower:

                    matched_title_words += 1

            if title_words:

                title_score = (
                    matched_title_words /
                    len(title_words)
                ) * 100

        # =================================================
        # 4. FINAL SCORE
        # =================================================

        # Skills are the MOST IMPORTANT requirement.
        #
        # Skills = 70%
        # Field  = 20%
        # Title  = 10%

        final_score = (

            skill_score * 0.70 +

            field_score * 0.20 +

            title_score * 0.10

        )

        final_score = max(
            0.0,
            min(100.0, final_score)
        )

        # =================================================
        # DEBUG
        # =================================================

        print(
            "------------------------------------------"
        )

        print(
            "JOB:",
            job.title
        )

        print(
            "JOB FIELD:",
            job.field
        )

        print(
            "JOB SKILLS:",
            job.skills
        )

        print(
            "RESUME SKILLS:",
            sorted(resume_skills)
        )

        print(
            "MATCHED SKILLS:",
            sorted(matched_skills)
        )

        print(
            "SKILL SCORE:",
            round(skill_score, 2)
        )

        print(
            "FIELD SCORE:",
            round(field_score, 2)
        )

        print(
            "TITLE SCORE:",
            round(title_score, 2)
        )

        print(
            "FINAL SCORE:",
            round(final_score, 2)
        )

        print(
            "------------------------------------------"
        )

        return round(
            final_score,
            2
        )

    except Exception as e:

        print(
            "RECOMMENDATION ERROR:",
            e
        )

        return 0.0





def predict_candidate_field(resume_text):

    if not resume_text:
        return None

    try:

        prediction = job_field_model.predict(
            [resume_text]
        )[0]

        return str(prediction).strip()

    except Exception as e:

        print(
            "Candidate field prediction error:",
            e
        )

        return None



# ====================================
# Resume Upload Configuration
# ====================================

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS




ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_image(filename):
    return "." in filename and \
           filename.rsplit(".",1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# ===========================
# Brevo Imports
# ===========================
from sib_api_v3_sdk import Configuration
from sib_api_v3_sdk import ApiClient
from sib_api_v3_sdk import TransactionalEmailsApi
from sib_api_v3_sdk import SendSmtpEmail

app = Flask(__name__)

app.config.from_pyfile("config.py")

app.config["SECRET_KEY"] = "talentbridge-reset-secret-2026"

serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"])





# =========================================================
# GOOGLE ANALYTICS 4
# =========================================================

GA4_MEASUREMENT_ID = "G-DQEN64H2L3"


@app.after_request
def add_google_analytics(response):

    if response.mimetype == "text/html":

        html = response.get_data(as_text=True)

        ga4_code = f"""
<!-- Google Analytics 4 -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA4_MEASUREMENT_ID}"></script>
<script>
    window.dataLayer = window.dataLayer || [];

    function gtag() {{
        dataLayer.push(arguments);
    }}

    gtag('js', new Date());

    gtag('config', '{GA4_MEASUREMENT_ID}');
</script>
"""

        head_end = html.lower().find("</head>")

        if head_end != -1 and GA4_MEASUREMENT_ID not in html:

            html = (
                html[:head_end]
                + ga4_code
                + html[head_end:]
            )

            response.set_data(html)

    return response







app.secret_key = "your-secret-key"

# Session lifetime
app.permanent_session_lifetime = 20 * 60

# ===========================
# Configuration
# ===========================



# ===========================
# Resume Upload Configuration
# ===========================

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ===========================
# Check Allowed File
# ===========================

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


import os

print("=" * 60)
print("DB URI:", app.config["SQLALCHEMY_DATABASE_URI"])
print("Instance path:", app.instance_path)
print("DB absolute path:", os.path.abspath(os.path.join(app.instance_path, "recruitment.db")))
print("=" * 60)

# ===========================
# Initialize Database
# ===========================
db.init_app(app)

# ===========================
# Send OTP Email (Brevo)
# ===========================
def send_otp_email(user_email, otp):

    configuration = Configuration()
    configuration.api_key["api-key"] = app.config["BREVO_API_KEY"]

    api_instance = TransactionalEmailsApi(ApiClient(configuration))

    email = SendSmtpEmail(

        sender={
            "name": app.config["TalentBridge Recruitment"],
            "email": app.config["your-verified-sender@email.com"]
        },

        to=[
            {
                "email": user_email
            }
        ],

        subject="Password Reset OTP",

        html_content=f"""
        <html>

        <body>

        <h2>TalentBridge Recruitment</h2>

        <p>Hello,</p>

        <p>Your Password Reset OTP is:</p>

        <h1 style="color:#27ae60;">{otp}</h1>

        <p>This OTP will expire in 10 minutes.</p>

        <br>

        <p>TalentBridge Team</p>

        </body>

        </html>
        """

    )

    api_instance.send_transac_email(email)


# ===========================
# Login Manager
# ===========================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "candidate_login"








@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))








@app.before_request
def check_admin_session_security():

    # Only check logged-in admin
    if "admin" not in session:
        return

    # Get current admin settings
    settings = AdminSettings.query.first()

    # If settings don't exist, do nothing
    if not settings:
        return

    # If Session Security is OFF
    if not settings.session_security:
        return

    # Current time
    current_time = time.time()

    # Last activity
    last_activity = session.get("admin_last_activity")

    # If there is a timestamp
    if last_activity:

        # 20 minutes = 1200 seconds
        inactive_time = current_time - last_activity

        if inactive_time > 1200:

            print("====================================")
            print("🔐 ADMIN SESSION EXPIRED")
            print("INACTIVE TIME:", inactive_time)
            print("====================================")

            # Clear admin session
            session.pop("admin", None)
            session.pop("admin_last_activity", None)

            # Remove OTP information
            session.pop("admin_otp", None)
            session.pop("admin_otp_email", None)

            flash("Your session expired due to inactivity. Please login again.")

            return redirect(url_for("admin_login"))

    # Update activity timestamp
    session["admin_last_activity"] = current_time







# ===========================
# Home
# ===========================
@app.route("/")
def home():

    settings = AdminSettings.query.first()

    return render_template(
        "index.html",
        settings=settings
    )





@app.route("/about")
def about():
    return render_template("about.html")






@app.route('/salary-guide')
def salary_guide():
    return render_template('salary_guide.html')












# ==================================================
# Candidate
# ==================================================

@app.route("/candidate/register", methods=["GET", "POST"])
def candidate_register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        mobile = request.form["mobile"]
        location = request.form["location"]
        education = request.form["education"]
        skills = request.form["skills"]
        experience = request.form["experience"]

        resume = request.files["resume"]

        password = request.form["password"]
        confirm = request.form["confirm"]

        # Check if uploaded file is PDF
        if not allowed_file(resume.filename):
            flash("Only PDF files are allowed.")
            return redirect(url_for("candidate_register"))

        # Generate unique filename
        
        filename = secure_filename(resume.filename)
        # Save resume
        resume.save(
            os.path.join(
                app.root_path,
                "static",
                "uploads",
                "resumes",
                filename
            )
        )

        # Check passwords
        if password != confirm:
            flash("Passwords do not match.")
            return redirect(url_for("candidate_register"))

        # Check if email already exists
        existing = User.query.filter_by(email=email).first()

        if existing:
            flash("Email already registered.")
            return redirect(url_for("candidate_register"))

        # Encrypt password
        hashed_password = generate_password_hash(password)

        # Create Candidate
        candidate = User(
            name=name,
            email=email,
            mobile=mobile,
            location=location,
            education=education,
            skills=skills,
            experience=experience,
            resume=filename,
            password=hashed_password,
            role="candidate"
        )

        db.session.add(candidate)
        db.session.commit()

        flash("Registration Successful!")
        return redirect(url_for("candidate_login"))

    return render_template("auth/candidate_register.html")




@app.route("/candidate/login", methods=["GET", "POST"])
def candidate_login():

    job_id = request.args.get("job_id", type=int)

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(
            email=email,
            role="candidate"
        ).first()

        if user and check_password_hash(user.password, password):

            session["candidate_id"] = user.id
            session["candidate_name"] = user.name

            # Remember the job the candidate wanted to apply for
            pending_job_id = session.get("pending_job_id")

            if pending_job_id:

                session.pop("pending_job_id", None)

                return redirect(
                    url_for(
                        "job_details",
                        job_id=pending_job_id
                    )
                )

            return redirect(url_for("candidate_dashboard"))

        flash("Invalid email or password.")

    # Remember selected job before showing login page
    if job_id:
        session["pending_job_id"] = job_id

    return render_template("auth/candidate_login.html")







@app.route("/candidate/forgot-password", methods=["GET", "POST"])
def candidate_forgot_password():

     if request.method == "POST":

        email = request.form.get("email", "").strip().lower()

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("No account found with this email address.", "danger")
            return redirect(url_for("candidate_forgot_password"))

        # Generate secure reset token
        token = serializer.dumps(
            email,
            salt="talentbridge-password-reset"
        )

        reset_link = url_for(
            "candidate_reset_password",
            token=token,
            _external=True
        )

        return render_template(
            "candidate/forgot_password.html",
            reset_link=reset_link
        )

     return render_template(
        "candidate/forgot_password.html"
    )








@app.route("/candidate/reset-password/<token>", methods=["GET", "POST"])
def candidate_reset_password(token):

    try:
        email = serializer.loads(
            token,
            salt="talentbridge-password-reset",
            max_age=1800
        )

    except Exception:
        flash("This password reset link is invalid or has expired.", "danger")
        return redirect(url_for("candidate_forgot_password"))

    user = User.query.filter_by(email=email).first()

    if not user:
        flash("User account not found.", "danger")
        return redirect(url_for("candidate_login"))

    if request.method == "POST":

        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not password or not confirm_password:
            flash("Please enter both passwords.", "danger")
            return redirect(
                url_for("candidate_reset_password", token=token)
            )

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(
                url_for("candidate_reset_password", token=token)
            )

        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "danger")
            return redirect(
                url_for("candidate_reset_password", token=token)
            )

        user.password = generate_password_hash(password)

        db.session.commit()

        flash(
            "Password reset successfully. You can now login.",
            "success"
        )

        return redirect(url_for("candidate_login"))

    return render_template(
        "candidate/reset_password.html"
    )







@app.route("/verify-otp", methods=["GET","POST"])
def verify_otp():

    if request.method == "POST":

        entered_otp = request.form["otp"]

        if entered_otp == session.get("otp"):

            flash("OTP Verified")

            return redirect(url_for("reset_password"))

        else:

            flash("Invalid OTP")

    return render_template("auth/verify_otp.html")


@app.route("/candidate/dashboard")
def candidate_dashboard():

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    user = User.query.get(session["candidate_id"])

    total_fields = 6
    completed = 0

    if user.name:
        completed += 1
    if user.email:
        completed += 1
    if user.mobile:
        completed += 1
    if user.education:
        completed += 1
    if user.skills:
        completed += 1
    if user.experience:
        completed += 1

    profile_percentage = int((completed / total_fields) * 100)

    # Dashboard Data
    available_jobs = Job.query.count()

    applied_jobs = Application.query.filter_by(
        candidate_id=user.id
    ).count()

    # If SavedJob table doesn't exist yet
    saved_jobs = SavedJob.query.filter_by(
    candidate_id=user.id
).count()

    recent_applications = (
        db.session.query(Application, Job)
        .join(Job, Application.job_id == Job.id)
        .filter(Application.candidate_id == user.id)
        .order_by(Application.id.desc())
        .limit(5)
        .all()
    )

    return render_template(
    "candidate/dashboard.html",
    user=user,
    available_jobs=available_jobs,
    applied_jobs=applied_jobs,
    saved_jobs=saved_jobs,
    profile_percentage=profile_percentage,
    recent_applications=recent_applications
)


@app.route("/candidate/profile", methods=["GET", "POST"])
def candidate_profile():

    # =====================================================
    # CHECK LOGIN
    # =====================================================

    candidate_id = session.get("candidate_id")

    if not candidate_id:

        flash(
            "Please login first.",
            "error"
        )

        return redirect(
            url_for("candidate_login")
        )


    # =====================================================
    # GET CANDIDATE
    # =====================================================

    candidate = User.query.get(candidate_id)

    if not candidate:

        flash(
            "Candidate not found.",
            "error"
        )

        return redirect(
            url_for("candidate_login")
        )


    # =====================================================
    # UPDATE PROFILE
    # =====================================================

    if request.method == "POST":

        candidate.name = request.form.get(
            "name",
            ""
        ).strip()


        candidate.mobile = request.form.get(
            "mobile",
            ""
        ).strip()


        candidate.location = request.form.get(
            "location",
            ""
        ).strip()


        candidate.education = request.form.get(
            "education",
            ""
        ).strip()


        candidate.skills = request.form.get(
            "skills",
            ""
        ).strip()


        candidate.experience = request.form.get(
            "experience",
            ""
        ).strip()


        # =================================================
        # PROFILE PHOTO
        # =================================================

        photo = request.files.get(
            "profile_photo"
        )


        if photo and photo.filename:

            allowed_extensions = {
                "png",
                "jpg",
                "jpeg",
                "webp"
            }


            original_filename = secure_filename(
                photo.filename
            )


            extension = (
                original_filename
               .rsplit(".", 1)[-1]
               .lower()
            )


            if extension not in allowed_extensions:

                flash(
                    "Please upload JPG, JPEG, PNG or WEBP image.",
                    "error"
                )

                return redirect(
                    url_for("candidate_profile")
                )


            # Create upload directory

            upload_folder = os.path.join(
                app.root_path,
                "static",
                "uploads",
                "profile_photos"
            )


            os.makedirs(
                upload_folder,
                exist_ok=True
            )


            # Generate unique filename

            filename = (
                str(uuid.uuid4())
                + "."
                + extension
            )


            file_path = os.path.join(
                upload_folder,
                filename
            )


            # Save image

            photo.save(file_path)


            # Save filename in database

            candidate.profile_photo = filename


        # =================================================
        # SAVE DATABASE
        # =================================================

        try:

            db.session.commit()

            flash(
                "Profile updated successfully!",
                "success"
            )

        except Exception as e:

            db.session.rollback()

            print(
                "PROFILE UPDATE ERROR:",
                e
            )

            flash(
                "Unable to update profile.",
                "error"
            )


        return redirect(
            url_for("candidate_profile")
        )


    # =====================================================
    # PROFILE COMPLETION
    # =====================================================

    total_fields = 7

    completed_fields = 0


    if candidate.name:
        completed_fields += 1

    if candidate.email:
        completed_fields += 1

    if candidate.mobile:
        completed_fields += 1

    if candidate.location:
        completed_fields += 1

    if candidate.education:
        completed_fields += 1

    if candidate.skills:
        completed_fields += 1

    if candidate.experience:
        completed_fields += 1


    profile_completion = int(
        (completed_fields / total_fields) * 100
    )


    return render_template(
        "candidate/profile.html",

        candidate=candidate,

        profile_completion=profile_completion
    )




@app.route("/candidate/jobs")
def candidate_jobs():
    """Render the candidate jobs page.

    Filtering, GPS and Best Match sorting are performed by
    /candidate/filter-jobs.
    """
    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    sort_by = request.args.get("sort_by", "match").strip().lower()

    return render_template(
        "candidate/jobs.html",
        jobs=[],
        match_scores={},
        candidate_field=None,
        sort_by=sort_by
    )


@app.route("/jobs")
def jobs():

    field = request.args.get("field", "").strip()

    if field:
        jobs = Job.query.filter(
            Job.field.ilike(field)
        ).order_by(Job.id.desc()).all()
    else:
        jobs = Job.query.order_by(Job.id.desc()).all()

    return render_template(
        "jobs.html",
        jobs=jobs,
        selected_field=field
    )








def calculate_distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometres between two GPS coordinates."""
    earth_radius_km = 6371.0
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


@app.route("/candidate/filter-jobs", methods=["GET", "POST"])
def filter_jobs():
    """Return filtered jobs with the real TalentBridge Best Match score.

    Existing matching engine weights:
        Skills = 70%
        Field  = 20%
        Title  = 10%

    Both GET and POST are accepted so the current frontend will work either way.
    """

    # =========================================================
    # 1. LOGIN
    # =========================================================
    if "candidate_id" not in session:
        return jsonify({
            "success": False,
            "count": 0,
            "jobs": [],
            "error": "Please login first."
        }), 401

    # =========================================================
    # 2. GET FILTER DATA
    # =========================================================
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
    else:
        data = request.args.to_dict()

    def get_value(*names, default=""):
        for name in names:
            value = data.get(name)
            if value is not None:
                return str(value).strip()
        return default

    keyword = get_value("keyword").lower()
    location = get_value("location").lower()
    field = get_value("field", "category").lower()
    job_type = get_value("job_type").lower()
    work_mode = get_value("work_mode").lower()
    education = get_value("education").lower()
    experience = get_value("experience").lower()
    date_posted = get_value("date_posted").lower()
    sort_value = get_value("sort", "sort_by", default="match").lower()

    sort_map = {
        "match": "best_match",
        "best match": "best_match",
        "best_match": "best_match",
        "recent": "recent",
        "latest": "recent",
        "salary": "salary",
        "distance": "distance",
        "nearest": "distance",
        "nearest first": "distance"
    }
    sort_by = sort_map.get(sort_value, "best_match")

    latitude = data.get("latitude")
    longitude = data.get("longitude")
    radius = data.get("radius")

    try:
        latitude = float(latitude) if latitude not in (None, "", "null") else None
        longitude = float(longitude) if longitude not in (None, "", "null") else None
        radius = float(radius) if radius not in (None, "", "null") else None
    except (ValueError, TypeError):
        latitude = longitude = radius = None

    # =========================================================
    # 3. CANDIDATE
    # =========================================================
    candidate = User.query.get(session["candidate_id"])
    if not candidate:
        return jsonify({
            "success": False,
            "count": 0,
            "jobs": [],
            "error": "Candidate not found."
        }), 404

    # =========================================================
    # 4. APPROVED JOBS
    # =========================================================
    jobs = Job.query.filter_by(status="Approved").order_by(Job.id.desc()).all()
    if not jobs:
        jobs = Job.query.order_by(Job.id.desc()).all()

    filtered_jobs = []

    education_levels = {
        "10th": 1,
        "12th": 2,
        "diploma": 3,
        "graduate": 4,
        "post graduate": 5,
        "postgraduate": 5,
        "phd": 6
    }
    experience_levels = {
        "fresher": 0,
        "0-1 years": 1,
        "1-3 years": 2,
        "3-5 years": 3,
        "5+ years": 4
    }

    def clean_filter(value):
        value = str(value or "").lower().strip()
        value = value.replace("–", "-").replace("—", "-")
        return re.sub(r"\s+", " ", value)

    for job in jobs:
        # KEYWORD
        if keyword:
            search_text = " ".join([
                str(job.title or ""),
                str(job.company or ""),
                str(job.skills or ""),
                str(job.description or ""),
                str(getattr(job, "role", "") or ""),
                str(job.field or "")
            ]).lower()
            if not all(word in search_text for word in keyword.split()):
                continue

        # LOCATION TEXT FILTER
        if location and location not in clean_filter(job.location):
            continue

        # FIELD / CATEGORY
        if field and field != "any":
            selected = clean_filter(field)
            actual = clean_filter(job.field)
            aliases = {
                "it & software": ["it & software", "it", "software"],
                "healthcare": ["healthcare", "health care", "medical"],
                "finance": ["finance", "financial"],
                "education": ["education", "teaching"],
                "sales": ["sales"],
                "marketing": ["marketing"],
                "human resources": ["human resources", "hr"],
                "engineering": ["engineering"],
                "data & analytics": ["data & analytics", "data", "analytics"],
                "design & creative": ["design & creative", "design", "creative"],
                "logistics": ["logistics"]
            }
            allowed = aliases.get(selected, [selected])
            if not any(alias in actual for alias in allowed):
                continue

        # JOB TYPE
        if job_type and job_type != "any":
            if job_type not in clean_filter(job.job_type):
                continue

        # WORK MODE
        if work_mode and work_mode != "any":
            if work_mode not in clean_filter(getattr(job, "work_mode", "")):
                continue

        # EDUCATION
        if education and education != "any":
            selected_level = education_levels.get(education)
            job_education = clean_filter(job.education)
            required_level = None
            for name, value in education_levels.items():
                if name in job_education:
                    required_level = value
                    break
            if selected_level is not None and required_level is not None:
                if selected_level < required_level:
                    continue
            elif education not in job_education:
                continue

        # EXPERIENCE
        if experience and experience != "any":
            selected_level = experience_levels.get(experience)
            job_experience = clean_filter(job.experience)
            required_level = None
            for name, value in experience_levels.items():
                if name in job_experience:
                    required_level = value
                    break
            if selected_level is not None and required_level is not None:
                if selected_level < required_level:
                    continue
            elif experience not in job_experience:
                continue

        # DATE POSTED
        if date_posted and date_posted != "any":
            if not job.date_posted:
                continue
            now = datetime.now()
            if date_posted == "last 24 hours":
                cutoff = now - timedelta(hours=24)
            elif date_posted == "last 7 days":
                cutoff = now - timedelta(days=7)
            elif date_posted == "last 30 days":
                cutoff = now - timedelta(days=30)
            else:
                cutoff = None
            if cutoff:
                job_date = job.date_posted
                if hasattr(job_date, "tzinfo") and job_date.tzinfo:
                    job_date = job_date.replace(tzinfo=None)
                if job_date < cutoff:
                    continue

        # GPS DISTANCE
        # ---------------------------------------------------------
        # IMPORTANT: Calculate distance whenever GPS coordinates
        # are available. The radius is OPTIONAL.
        #
        # This is what makes:
        #   - Nearest First  -> work without depending on radius
        #   - Jobs Near Me   -> still respect 5/10/25/50 km radius
        # ---------------------------------------------------------
        distance = None

        if latitude is not None and longitude is not None:
            job_lat = getattr(job, "latitude", None)
            job_lng = getattr(job, "longitude", None)

            if job_lat is not None and job_lng is not None:
                try:
                    distance = calculate_distance(
                        float(latitude),
                        float(longitude),
                        float(job_lat),
                        float(job_lng)
                    )
                except (ValueError, TypeError):
                    distance = None

                # Radius is only a filter when the user selected one.
                if radius is not None and distance is not None:
                    if distance > float(radius):
                        continue

        filtered_jobs.append((job, distance))

    # =========================================================
    # 5. RESUME + CANDIDATE FIELD
    # =========================================================
    candidate_field = None
    resume_text = ""

    if candidate.resume:
        resume_path = os.path.join(
            app.root_path,
            "static",
            "uploads",
            "resumes",
            candidate.resume
        )
        if os.path.exists(resume_path):
            resume_text = extract_resume_text(resume_path)
            if resume_text.strip():
                candidate_field = predict_candidate_field(resume_text)

    # Fallback when no resume is available.
    if not resume_text.strip():
        resume_text = " ".join([
            str(candidate.name or ""),
            str(candidate.skills or ""),
            str(candidate.education or ""),
            str(candidate.experience or ""),
            str(candidate.location or "")
        ]).strip()
        if resume_text and not candidate_field:
            candidate_field = predict_candidate_field(resume_text)

    # =========================================================
    # 6. REAL MATCH SCORE
    # =========================================================
    result = []

    print("\n==========================================")
    print("       TALENTBRIDGE BEST MATCH")
    print("==========================================")
    print("Candidate:", candidate.name)
    print("Predicted Candidate Field:", candidate_field)
    print("Jobs after filters:", len(filtered_jobs))

    for job, distance in filtered_jobs:
        try:
            score = float(calculate_job_match(
                resume_text,
                job,
                candidate_field
            ))
        except Exception as e:
            print("Match calculation error for", job.title, ":", e)
            score = 0.0

        score = max(0.0, min(100.0, score))

        print(f"{job.title} | {job.field} | SCORE: {score:.2f}")

        result.append({
            "id": job.id,
            "title": job.title or "",
            "company": job.company or "",
            "location": job.location or "",
            "salary": job.salary or "",
            "job_type": job.job_type or "",
            "work_mode": getattr(job, "work_mode", "") or "",
            "experience": job.experience or "",
            "education": job.education or "",
            "skills": job.skills or "",
            "field": job.field or "",
            "role": getattr(job, "role", "") or "",
            "description": getattr(job, "description", "") or "",
            "deadline": str(job.deadline) if job.deadline else "",
            "latitude": getattr(job, "latitude", None),
            "longitude": getattr(job, "longitude", None),
            "distance": round(distance, 2) if distance is not None else None,
            "score": round(score, 2)
        })

    # =========================================================
    # 7. SORT
    # =========================================================
    if sort_by == "best_match":
        result.sort(
            key=lambda item: float(item.get("score", 0)),
            reverse=True
        )
    elif sort_by == "recent":
        result.sort(key=lambda item: int(item.get("id", 0)), reverse=True)
    elif sort_by == "salary":
        def salary_value(item):
            text = str(item.get("salary") or "").replace(",", "")
            numbers = re.findall(r"\d+(?:\.\d+)?", text)
            try:
                return max(float(n) for n in numbers) if numbers else 0.0
            except (ValueError, TypeError):
                return 0.0
        result.sort(key=salary_value, reverse=True)
    elif sort_by == "distance":
        # Nearest first: smallest calculated GPS distance wins.
        # Jobs without coordinates are placed at the end.
        result.sort(
            key=lambda item: (
                item.get("distance")
                if item.get("distance") is not None
                else float("inf"),
                -int(item.get("id", 0))
            )
        )

    print("------------------------------------------")
    print("FINAL JOB ORDER")
    print("------------------------------------------")
    for item in result:
        print(
            item["id"],
            "|", item["title"],
            "|", item["field"],
            "| SCORE:", item["score"],
            "| DISTANCE:", item.get("distance")
        )
    print("==========================================\n")

    return jsonify({
        "success": True,
        "count": len(result),
        "jobs": result,
        "candidate_field": candidate_field,
        "sort": sort_by
    })


@app.route("/fields")
def fields():
    return render_template("fields.html")





@app.route("/candidate/job/<int:job_id>")
def job_details(job_id):

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    job = Job.query.get_or_404(job_id)

    return render_template("candidate/job_details.html", job=job)



@app.route("/candidate/apply/<int:job_id>")
def apply_job(job_id):

    # ==========================================
    # CANDIDATE LOGIN CHECK
    # ==========================================

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    # ==========================================
    # CHECK ADMIN RECRUITMENT SETTING
    # ==========================================

    settings = AdminSettings.query.first()

    if settings and not settings.allow_applications:

        flash(
            "❌ Candidate applications are currently disabled by the administrator.",
            "warning"
        )

        return redirect(url_for("candidate_jobs"))

    # ==========================================
    # CHECK JOB EXISTS
    # ==========================================

    job = Job.query.get(job_id)

    if not job:

        flash(
            "❌ Job not found.",
            "danger"
        )

        return redirect(url_for("candidate_jobs"))

    # ==========================================
    # CHECK IF ALREADY APPLIED
    # ==========================================

    existing_application = Application.query.filter_by(
        candidate_id=session["candidate_id"],
        job_id=job_id
    ).first()

    if existing_application:

        flash(
            "⚠️ You have already applied for this job.",
            "warning"
        )

        return redirect(url_for("candidate_jobs"))

    # ==========================================
    # CREATE APPLICATION
    # ==========================================

    application = Application(

        candidate_id=session["candidate_id"],

        job_id=job_id,

        status="Applied"

    )

    db.session.add(application)

    db.session.commit()

    # ==========================================
    # SUCCESS
    # ==========================================

    flash(
        "✅ Application Submitted Successfully!",
        "success"
    )

    return redirect(url_for("candidate_jobs"))
   




@app.route("/candidate/applied")
def applied_jobs():

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    applications = (
        db.session.query(Application, Job)
        .join(Job, Application.job_id == Job.id)
        .filter(Application.candidate_id == session["candidate_id"])
        .all()
    )

    return render_template(
        "candidate/applied_jobs.html",
        applications=applications
    )




@app.route("/candidate/saved")
def saved_jobs():

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    saved_jobs = SavedJob.query.filter_by(
        candidate_id=session["candidate_id"]
    ).all()

    return render_template(
        "candidate/saved_jobs.html",
        saved_jobs=saved_jobs
    )



@app.route("/candidate/save/<int:job_id>")
def save_job(job_id):

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    existing = SavedJob.query.filter_by(
        candidate_id=session["candidate_id"],
        job_id=job_id
    ).first()

    if existing:
        flash("Job already saved.")
    else:
        save = SavedJob(
            candidate_id=session["candidate_id"],
            job_id=job_id
        )

        db.session.add(save)
        db.session.commit()

        flash("Job saved successfully.")

    return redirect(url_for("candidate_jobs"))




@app.route("/candidate/delete-saved/<int:saved_id>")
def delete_saved_job(saved_id):

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    saved = SavedJob.query.get_or_404(saved_id)

    if saved.candidate_id != session["candidate_id"]:
        flash("Unauthorized!")
        return redirect(url_for("saved_jobs"))

    db.session.delete(saved)
    db.session.commit()

    flash("Saved job removed.")

    return redirect(url_for("saved_jobs"))





@app.route("/candidate/delete-application/<int:application_id>")
def delete_application(application_id):

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    application = Application.query.get_or_404(application_id)

    if application.candidate_id != session["candidate_id"]:
        flash("Unauthorized!")
        return redirect(url_for("applied_jobs"))

    db.session.delete(application)
    db.session.commit()

    flash("Application deleted successfully.")

    return redirect(url_for("applied_jobs"))




@app.route("/candidate/resume")
def resume():
    return render_template("candidate/resume.html")


@app.route("/candidate/notifications")
def notifications():

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    notifications = (
        db.session.query(Application, Job)
        .join(Job, Application.job_id == Job.id)
        .filter(Application.candidate_id == session["candidate_id"])
        .order_by(Application.applied_date.desc())
        .all()
    )

    return render_template(
        "candidate/notifications.html",
        notifications=notifications
    )





@app.route("/candidate/settings", methods=["GET", "POST"])
def settings():

    if "candidate_id" not in session:
        return redirect(url_for("candidate_login"))

    user = User.query.get(session["candidate_id"])

    if request.method == "POST":

        # -----------------------------
        # Update Profile Picture
        # -----------------------------
        profile = request.files.get("profile_pic")

        if profile and profile.filename != "":

            filename = secure_filename(profile.filename)

            folder = os.path.join(
                app.root_path,
                "static",
                "uploads",
                "profile_pics"
            )

            os.makedirs(folder, exist_ok=True)

            profile.save(os.path.join(folder, filename))

            user.profile_pic = filename


        # -----------------------------
        # Replace Resume
        # -----------------------------
        resume = request.files.get("resume")

        if resume and resume.filename != "":

            filename = secure_filename(resume.filename)

            folder = os.path.join(
                app.root_path,
                "static",
                "uploads",
                "resumes"
            )

            os.makedirs(folder, exist_ok=True)

            resume.save(os.path.join(folder, filename))

            user.resume = filename

        db.session.commit()

        flash("Settings updated successfully!")

        return redirect(url_for("settings"))

    return render_template("candidate/settings.html", user=user)





@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():

    # Prevent direct access if the user hasn't verified an OTP
    if "reset_email" not in session:
        flash("Session expired. Please request a new OTP.")
        return redirect(url_for("candidate_forgot_password"))

    if request.method == "POST":

        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match.")
            return redirect(url_for("reset_password"))

        # Find the user
        user = User.query.filter_by(email=session["reset_email"]).first()

        if user:

            # Encrypt password
            hashed_password = generate_password_hash(password)

            # Update password
            user.password = hashed_password

            db.session.commit()

            # Clear session
            session.pop("otp", None)
            session.pop("reset_email", None)

            flash("Password updated successfully!")
            return redirect(url_for("candidate_login"))

        else:
            flash("User not found.")
            return redirect(url_for("candidate_forgot_password"))

    return render_template("auth/reset_password.html")


# ==================================================
# HR
# ==================================================

@app.route("/hr/register", methods=["GET", "POST"])
def hr_register():

    if request.method == "POST":

        company = request.form["company_name"]
        name = request.form["hr_name"]
        email = request.form["email"]
        mobile = request.form["mobile"]
        password = request.form["password"]
        confirm = request.form["confirm_password"]

        if password != confirm:
            flash("Passwords do not match.")
            return redirect(url_for("hr_register"))

        existing = User.query.filter_by(email=email).first()

        if existing:
            flash("Email already exists.")
            return redirect(url_for("hr_register"))

        hashed_password = generate_password_hash(password)

        hr = User(
            name=name,
            company=company,
            email=email,
            mobile=mobile,
            password=hashed_password,
            role="hr"
        )

        db.session.add(hr)
        db.session.commit()

        flash("HR Registered Successfully!")

        return redirect(url_for("hr_login"))

    return render_template("auth/hr_register.html")



@app.route("/hr/login", methods=["GET", "POST"])
def hr_login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        hr = User.query.filter_by(email=email, role="hr").first()

        if hr and check_password_hash(hr.password, password):

            print("HR Login Successful")
            print("HR ID:", hr.id)
            print("HR Name:", hr.name)

            session["hr_id"] = hr.id
            session["hr_name"] = hr.name

            return redirect(url_for("hr_dashboard"))

        else:
            print("Login Failed")
            flash("Invalid Email or Password")

    return render_template("auth/hr_login.html")


@app.route("/hr/forgot-password", methods=["GET", "POST"])
def hr_forgot_password():
    return render_template("auth/hr_forgot_password.html")




# ==========================
# HR Dashboard
# ==========================

@app.route("/hr/dashboard")
def hr_dashboard():

    print("===== HR Dashboard Called =====")
    print("Session:", session)

    if "hr_id" not in session:
        print("No hr_id found in session")
        return redirect(url_for("hr_login"))

    print("HR ID:", session["hr_id"])

    hr = User.query.get(session["hr_id"])
    print("HR Object:", hr)

    total_jobs = Job.query.filter_by(hr_id=hr.id).count()
    total_applicants = Application.query.count()

    print("Dashboard Loading...")

    return render_template(
        "hr/dashboard.html",
        hr=hr,
        total_jobs=total_jobs,
        total_applicants=total_applicants
    )

# ==========================
# Post Job
# ==========================

@app.route("/hr/post-job", methods=["GET", "POST"])
def post_job():

    # ==========================================
    # HR LOGIN CHECK
    # ==========================================

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    # ==========================================
    # CHECK ADMIN SETTING
    # ==========================================

    settings = AdminSettings.query.first()

    if settings and not settings.allow_job_posting:

        flash(
            "❌ Job posting is currently disabled by the administrator.",
            "warning"
        )

        return redirect(url_for("hr_dashboard"))

    # ==========================================
    # POST JOB
    # ==========================================

    if request.method == "POST":

        title = request.form.get("title", "").strip()
        company = request.form.get("company", "").strip()
        location = request.form.get("location", "").strip()
        salary = request.form.get("salary", "").strip()
        vacancies = request.form.get("vacancies")
        job_type = request.form.get("job_type", "").strip()
        experience = request.form.get("experience", "").strip()
        education = request.form.get("education", "").strip()
        skills = request.form.get("skills", "").strip()
        incentives = request.form.get("incentives", "").strip()
        deadline = request.form.get("deadline", "").strip()
        description = request.form.get("description", "").strip()
        working_days = request.form.get("working_days","").strip()
        timing = request.form.get("timing","").strip()
        user_type = request.form.get("user_type","All").strip()
        quick_apply = (request.form.get("quick_apply") == "on")

        # ==========================================
        # BASIC VALIDATION
        # ==========================================

        if not title or not company or not location:

            flash(
                "Please fill all required fields.",
                "warning"
            )

            return redirect(url_for("post_job"))

        # ==========================================
        # ML JOB FIELD PREDICTION
        # ==========================================

        predicted_field = predict_job_field(
            title,
            description,
            skills
        )

        print("================================")
        print("ML JOB FIELD PREDICTION")
        print("Job:", title)
        print("Predicted Field:", predicted_field)
        print("================================")

        # ==========================================
        # CHECK JOB APPROVAL SETTING
        # ==========================================

        settings = AdminSettings.query.first()

        if settings and settings.job_approval:

            job_status = "Pending"

        else:

            job_status = "Approved"

        # ==========================================
        # CREATE JOB
        # ==========================================

        job = Job(
    title=title,
    company=company,
    location=location,
    salary=salary,
    vacancies=int(vacancies),
    job_type=job_type,
    experience=experience,
    education=education,
    skills=skills,
    incentives=incentives,
    deadline=deadline,
    description=description,

    # ML predicted field
    field=predicted_field,

    # Additional filters
    working_days=working_days,
    timing=timing,
    user_type=user_type,
    quick_apply=quick_apply,

    # HR
    hr_id=session["hr_id"],

    # Approval
    status=job_status
)

        db.session.add(job)
        db.session.commit()

        # ==========================================
        # SUCCESS MESSAGE
        # ==========================================

        if job_status == "Pending":

            flash(
                f"✅ Job submitted successfully. "
                f"Predicted field: {predicted_field}. "
                f"Waiting for admin approval.",
                "success"
            )

        else:

            flash(
                f"✅ Job posted successfully under "
                f"{predicted_field}!",
                "success"
            )

        return redirect(url_for("manage_jobs"))

    # ==========================================
    # SHOW POST JOB PAGE
    # ==========================================

    return render_template("hr/post_job.html")






# ==========================
# Manage Jobs
# ==========================

@app.route("/hr/manage-jobs")
def manage_jobs():

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    search = request.args.get("search", "")

    jobs = Job.query.filter_by(hr_id=session["hr_id"])

    if search:
        jobs = jobs.filter(Job.title.ilike(f"%{search}%"))

    jobs = jobs.all()

    total_applications = (
        db.session.query(Application)
        .join(Job, Application.job_id == Job.id)
        .filter(Job.hr_id == session["hr_id"])
        .count()
    )

    return render_template(
        "hr/manage_jobs.html",
        jobs=jobs,
        total_applications=total_applications
    )


@app.route("/hr/delete_job/<int:job_id>")
def delete_job(job_id):

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    job = Job.query.get_or_404(job_id)

    if job.hr_id != session["hr_id"]:
        flash("Unauthorized")
        return redirect(url_for("manage_jobs"))

    db.session.delete(job)
    db.session.commit()

    flash("Job Deleted Successfully")

    return redirect(url_for("manage_jobs"))




@app.route("/hr/edit-job/<int:job_id>", methods=["GET", "POST"])
def edit_job(job_id):

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    job = Job.query.get_or_404(job_id)

    if request.method == "POST":

        job.title = request.form["title"]
        job.company = request.form["company"]
        job.location = request.form["location"]
        job.salary = request.form["salary"]
        job.vacancies = int(request.form["vacancies"])
        job.job_type = request.form["job_type"]
        job.experience = request.form["experience"]
        job.education = request.form["education"]
        job.skills = request.form["skills"]
        job.incentives = request.form["incentives"]
        job.deadline = request.form["deadline"]
        job.description = request.form["description"]

        db.session.commit()

        flash("Job updated successfully!")
        return redirect(url_for("manage_jobs"))

    return render_template("hr/edit_job.html", job=job)



# ==========================
# Applicants
# ==========================

@app.route("/hr/applicants")
def applicants():

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    search = request.args.get("search", "")

    applications = (
        db.session.query(Application, User, Job)
        .join(User, Application.candidate_id == User.id)
        .join(Job, Application.job_id == Job.id)
        .filter(Job.hr_id == session["hr_id"])
    )

    if search:
        applications = applications.filter(
            db.or_(
                User.name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%"),
                Job.title.ilike(f"%{search}%")
            )
        )

    applications = applications.all()

    return render_template(
        "hr/applicants.html",
        applications=applications
    )



@app.route("/hr/shortlist/<int:application_id>")
def shortlist_candidate(application_id):

    application = Application.query.get_or_404(application_id)

    application.status = "Shortlisted"

    db.session.commit()

    flash("Candidate Shortlisted Successfully!")

    return redirect(url_for("applicants"))


@app.route("/hr/shortlisted")
def shortlisted():

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    search = request.args.get("search", "")

    shortlisted = (
        db.session.query(Application, User, Job)
        .join(User, Application.candidate_id == User.id)
        .join(Job, Application.job_id == Job.id)
        .filter(
            Job.hr_id == session["hr_id"],
            Application.status == "Shortlisted"
        )
    )

    if search:
        shortlisted = shortlisted.filter(
            or_(
                User.name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%"),
                Job.title.ilike(f"%{search}%")
            )
        )

    shortlisted = shortlisted.all()

    return render_template(
        "hr/shortlisted.html",
        shortlisted=shortlisted
    )







@app.route("/hr/candidate/<int:candidate_id>")
def hr_view_candidate(candidate_id):

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    candidate = User.query.get_or_404(candidate_id)

    return render_template(
        "hr/candidate_profile.html",
        candidate=candidate
    )







@app.route("/hr/reject/<int:application_id>")
def reject_candidate(application_id):

    application = Application.query.get_or_404(application_id)

    application.status = "Rejected"

    db.session.commit()

    flash("Candidate Rejected!")

    return redirect(url_for("applicants"))







# ==================================================
# HR PROFILE
# ==================================================

@app.route("/hr/profile", methods=["GET", "POST"])
def hr_profile():

    # -----------------------------
    # CHECK HR LOGIN
    # -----------------------------

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    hr = User.query.get(session["hr_id"])

    if not hr:
        flash("HR account not found.", "danger")
        return redirect(url_for("hr_login"))

    # -----------------------------
    # UPDATE PROFILE
    # -----------------------------

    if request.method == "POST":

        hr.name = request.form.get(
            "name",
            ""
        ).strip()

        hr.mobile = request.form.get(
            "mobile",
            ""
        ).strip()

        hr.company = request.form.get(
            "company",
            ""
        ).strip()

        hr.location = request.form.get(
            "location",
            ""
        ).strip()

        hr.education = request.form.get(
            "education",
            ""
        ).strip()

        hr.skills = request.form.get(
            "skills",
            ""
        ).strip()

        hr.experience = request.form.get(
            "experience",
            ""
        ).strip()

        # -----------------------------
        # PROFILE PHOTO
        # -----------------------------

        profile = request.files.get("profile_pic")

        if profile and profile.filename != "":

            if allowed_image(profile.filename):

                filename = secure_filename(
                    profile.filename
                )

                profile_folder = os.path.join(
                    app.root_path,
                    "static",
                    "uploads",
                    "profile_pics"
                )

                os.makedirs(
                    profile_folder,
                    exist_ok=True
                )

                profile.save(
                    os.path.join(
                        profile_folder,
                        filename
                    )
                )

                hr.profile_pic = filename

            else:

                flash(
                    "Please upload a valid image file.",
                    "danger"
                )

                return redirect(
                    url_for("hr_profile")
                )

        # -----------------------------
        # SAVE DATABASE
        # -----------------------------

        db.session.commit()

        # Update HR name in session
        session["hr_name"] = hr.name

        flash(
            "Profile Updated Successfully!",
            "success"
        )

        return redirect(
            url_for("hr_profile")
        )

    # -----------------------------
    # PROFILE COMPLETION
    # -----------------------------

    total_fields = 8
    completed_fields = 0

    if hr.name:
        completed_fields += 1

    if hr.email:
        completed_fields += 1

    if hr.mobile:
        completed_fields += 1

    if hr.company:
        completed_fields += 1

    if hr.location:
        completed_fields += 1

    if hr.education:
        completed_fields += 1

    if hr.skills:
        completed_fields += 1

    if hr.experience:
        completed_fields += 1

    profile_completion = int(
        (completed_fields / total_fields) * 100
    )

    return render_template(
        "hr/profile.html",
        hr=hr,
        profile_completion=profile_completion
    )
















# ==========================
# Settings
# ==========================

# ==================================================
# HR SETTINGS
# ==================================================

@app.route("/hr/settings", methods=["GET", "POST"])
def hr_settings():

    # ------------------------------------------
    # HR LOGIN CHECK
    # ------------------------------------------

    if "hr_id" not in session:
        return redirect(url_for("hr_login"))

    hr = User.query.get(session["hr_id"])

    if not hr:
        flash("HR account not found.", "danger")
        return redirect(url_for("hr_login"))

    # ------------------------------------------
    # UPDATE SETTINGS
    # ------------------------------------------

    if request.method == "POST":

        # ======================================
        # ACCOUNT INFORMATION
        # ======================================

        name = request.form.get(
            "name", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip()

        mobile = request.form.get(
            "mobile", ""
        ).strip()

        company = request.form.get(
            "company", ""
        ).strip()

        location = request.form.get(
            "location", ""
        ).strip()

        if name:
            hr.name = name

        if email:
            hr.email = email

        hr.mobile = mobile
        hr.company = company
        hr.location = location

        # ======================================
        # CHANGE PASSWORD
        # ======================================

        current_password = request.form.get(
            "current_password", ""
        )

        new_password = request.form.get(
            "new_password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        # Only change password if user entered one
        if new_password:

            if not current_password:

                flash(
                    "Please enter your current password.",
                    "danger"
                )

                return redirect(
                    url_for("hr_settings")
                )

            # Check current password
            if not check_password_hash(
                hr.password,
                current_password
            ):

                flash(
                    "Current password is incorrect.",
                    "danger"
                )

                return redirect(
                    url_for("hr_settings")
                )

            # Check new passwords
            if new_password != confirm_password:

                flash(
                    "New passwords do not match.",
                    "danger"
                )

                return redirect(
                    url_for("hr_settings")
                )

            if len(new_password) < 6:

                flash(
                    "Password must contain at least 6 characters.",
                    "danger"
                )

                return redirect(
                    url_for("hr_settings")
                )

            # Hash new password
            hr.password = generate_password_hash(
                new_password
            )

        # ======================================
        # SAVE CHANGES
        # ======================================

        db.session.commit()

        # Update session name
        session["hr_name"] = hr.name

        flash(
            "HR settings updated successfully!",
            "success"
        )

        return redirect(
            url_for("hr_settings")
        )

    # ------------------------------------------
    # SHOW SETTINGS
    # ------------------------------------------

    return render_template(
        "hr/settings.html",
        hr=hr
    )


# ==========================
# Logout
# ==========================

@app.route("/logout")
def logout():

    session.clear()

    flash("Logged Out Successfully")

    return redirect(url_for("home"))


# ==================================================
# Admin
# ==================================================
def send_admin_otp_email(user_email, otp):

    configuration = Configuration()

    configuration.api_key["api-key"] = app.config["BREVO_API_KEY"]

    api_instance = TransactionalEmailsApi(
        ApiClient(configuration)
    )

    email = SendSmtpEmail(

        sender={
            "name": app.config["TalentBridge Recruitment"],
            "email": app.config["your-verified-sender@email.com"]
        },

        to=[
            {
                "email": user_email
            }
        ],

        subject="TalentBridge Admin Login OTP",

        html_content=f"""
        <html>
        <body>

            <h2>TalentBridge Admin Login</h2>

            <p>Your Admin Login OTP is:</p>

            <h1>{otp}</h1>

            <p>Please do not share this OTP with anyone.</p>

            <p>TalentBridge Team</p>

        </body>
        </html>
        """
    )

    response = api_instance.send_transac_email(email)

    print("Brevo Response:", response)






@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():

    # ==========================================
    # GET SETTINGS DIRECTLY FROM DATABASE
    # ==========================================

    settings = AdminSettings.query.first()

    if not settings:

        settings = AdminSettings(
            admin_name="Administrator",
            admin_email="admin@gmail.com",
            admin_password="admin123",
            two_factor=False,
            login_alerts=True,
            session_security=True
        )

        db.session.add(settings)
        db.session.commit()

    # ==========================================
    # LOGIN POST
    # ==========================================

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        # ======================================
        # CHECK LOGIN
        # ======================================

        if (
            email != settings.admin_email
            or password != settings.admin_password
        ):

            print("❌ ADMIN LOGIN FAILED")

            flash("Invalid Email or Password")

            return render_template(
                "admin/login.html"
            )

        print("\n========================================")
        print("       ADMIN LOGIN SUCCESS")
        print("========================================")

        print("ADMIN:", settings.admin_email)

        # ======================================
        # VERY IMPORTANT:
        # RELOAD SETTINGS FROM DATABASE
        # ======================================

        db.session.expire_all()

        settings = AdminSettings.query.first()

        print(
            "2FA DATABASE VALUE:",
            repr(settings.two_factor)
        )

        print(
            "2FA DATABASE TYPE:",
            type(settings.two_factor)
        )

        # ======================================
        # CLEAR OLD OTP
        # ======================================

        session.pop("admin_otp", None)
        session.pop("admin_otp_email", None)

        # ======================================
        # 2FA CHECK
        # ======================================

        if settings.two_factor is True:

            # ==================================
            # 2FA ENABLED
            # ==================================

            print("🔐 2FA = TRUE")
            print("🔐 GENERATING OTP")

            otp = random.randint(
                100000,
                999999
            )

            session["admin_otp"] = str(otp)

            session["admin_otp_email"] = email

            print(
                "🔐 ADMIN OTP:",
                otp
            )

            # ==================================
            # SEND EMAIL
            # ==================================

            try:

                send_admin_otp_email(
                    email,
                    otp
                )

                print(
                    "✅ OTP EMAIL SENT"
                )

                flash(
                    "OTP sent to your registered email."
                )

            except Exception as e:

                print(
                    "❌ OTP EMAIL ERROR:"
                )

                print(e)

                flash(
                    "OTP generated. Check terminal for OTP."
                )

            # ==================================
            # ONLY HERE DO WE OPEN OTP PAGE
            # ==================================

            return redirect(
                url_for("admin_verify_otp")
            )

        # ======================================
        # 2FA DISABLED
        # ======================================

        else:

            print("⚪ 2FA = FALSE")
            print("🚫 NO OTP")
            print("🚫 NO OTP EMAIL")
            print("🚫 NO OTP PAGE")
            print("✅ DIRECT LOGIN")

            # ==================================
            # CLEAR ANY OLD OTP
            # ==================================

            session.pop(
                "admin_otp",
                None
            )

            session.pop(
                "admin_otp_email",
                None
            )

            # ==================================
            # LOGIN ADMIN
            # ==================================

            session["admin"] = settings.admin_name

            # ==================================
            # SESSION SECURITY
            # ==================================

            if settings.session_security:

                session.permanent = True

                session[
                    "admin_last_activity"
                ] = time.time()

            # ==================================
            # LOGIN ALERT
            # ==================================

            if settings.login_alerts:

                print(
                    "🔔 ADMIN LOGIN ALERT"
                )

            flash("Welcome Admin!")

            # ==================================
            # DIRECT DASHBOARD
            # ==================================

            return redirect(
                url_for("admin_dashboard")
            )

    # ==========================================
    # LOGIN PAGE
    # ==========================================

    return render_template(
        "admin/login.html"
    )








@app.route("/admin/verify-otp", methods=["GET", "POST"])
def admin_verify_otp():

    # ==========================================
    # GET CURRENT SETTINGS FROM DATABASE
    # ==========================================

    settings = AdminSettings.query.first()

    # ==========================================
    # CRITICAL 2FA CHECK
    # ==========================================

    if not settings or settings.two_factor is not True:

        print("\n========================================")
        print("🚫 OTP PAGE BLOCKED")
        print("2FA IS DISABLED")
        print("========================================")

        # Remove any old OTP
        session.pop(
            "admin_otp",
            None
        )

        session.pop(
            "admin_otp_email",
            None
        )

        # Login admin directly
        if settings:

            session["admin"] = settings.admin_name

        flash(
            "2FA is disabled. Logged in directly."
        )

        return redirect(
            url_for("admin_dashboard")
        )

    # ==========================================
    # CHECK OTP EXISTS
    # ==========================================

    stored_otp = session.get(
        "admin_otp"
    )

    otp_email = session.get(
        "admin_otp_email"
    )

    if not stored_otp:

        flash(
            "No OTP verification is pending."
        )

        return redirect(
            url_for("admin_login")
        )

    # ==========================================
    # POST OTP
    # ==========================================

    if request.method == "POST":

        entered_otp = request.form.get(
            "otp",
            ""
        ).strip()

        print("\n========================================")
        print("       OTP VERIFICATION")
        print("========================================")

        print(
            "ENTERED OTP:",
            entered_otp
        )

        print(
            "STORED OTP:",
            stored_otp
        )

        # ======================================
        # VERIFY
        # ======================================

        if entered_otp == str(stored_otp):

            print("✅ OTP VERIFIED")

            # Remove OTP
            session.pop(
                "admin_otp",
                None
            )

            session.pop(
                "admin_otp_email",
                None
            )

            # ==================================
            # LOGIN ADMIN
            # ==================================

            session["admin"] = settings.admin_name



            if AdminSettings.query.first().session_security:
             session.permanent = True
             session["admin_last_activity"] = time.time()

            if settings.session_security:

                session.permanent = True

                session[
                    "admin_last_activity"
                ] = time.time()

            flash(
                "Welcome Admin!"
            )

            return redirect(
                url_for("admin_dashboard")
            )

        else:

            print("❌ INVALID OTP")

            flash(
                "Invalid OTP."
            )

    # ==========================================
    # OTP PAGE
    # ==========================================

    return render_template(
        "admin/verify_otp.html"
    )










@app.route("/admin/dashboard")
def admin_dashboard():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    total_candidates = User.query.filter_by(role="candidate").count()
    total_hr = User.query.filter_by(role="hr").count()
    total_jobs = Job.query.count()
    total_applications = Application.query.count()

    return render_template(
        "admin/dashboard.html",
        total_candidates=total_candidates,
        total_hr=total_hr,
        total_jobs=total_jobs,
        total_applications=total_applications
    )


@app.route("/admin/candidates")
def admin_candidates():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    candidates = User.query.filter_by(role="candidate").all()

    return render_template(
        "admin/candidates.html",
        candidates=candidates
    )








@app.route("/admin/delete-candidate/<int:user_id>", methods=["POST"])
def delete_candidate(user_id):

    candidate = User.query.get_or_404(user_id)

    # Delete applications of this candidate first
    Application.query.filter_by(
        candidate_id=candidate.id
    ).delete(
        synchronize_session=False
    )

    # Delete candidate
    db.session.delete(candidate)

    db.session.commit()

    return redirect(url_for("admin_candidates"))








@app.route("/admin/hr")
def admin_hr():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    hr_users = User.query.filter_by(role="hr").all()

    return render_template(
        "admin/hr.html",
        hr_users=hr_users
    )









@app.route("/admin/delete-hr/<int:user_id>", methods=["POST"])
def delete_hr(user_id):

    hr = User.query.get_or_404(user_id)

    # Make sure this is actually an HR account
    if hr.role != "hr":
        return "Invalid HR user", 400

    # Delete jobs posted by this HR first
    Job.query.filter_by(hr_id=hr.id).delete(
        synchronize_session=False
    )

    # Delete the HR user
    db.session.delete(hr)

    db.session.commit()

    return redirect(url_for("admin_hr"))









@app.route("/admin/jobs")
def admin_jobs():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    jobs = Job.query.all()

    return render_template(
        "admin/jobs.html",
        jobs=jobs
    )










@app.route("/admin/applications")
def admin_applications():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    applications = Application.query.all()

    return render_template(
        "admin/applications.html",
        applications=applications
    )














@app.route("/admin/reports")
def admin_reports():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    total_candidates = User.query.filter_by(role="candidate").count()

    total_hr = User.query.filter_by(role="hr").count()

    total_jobs = Job.query.count()

    total_applications = Application.query.count()

    return render_template(
        "admin/reports.html",
        total_candidates=total_candidates,
        total_hr=total_hr,
        total_jobs=total_jobs,
        total_applications=total_applications
    )





@app.route("/admin/settings", methods=["GET", "POST"])
def admin_settings():

    # ==========================================
    # ADMIN LOGIN CHECK
    # ==========================================

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    # ==========================================
    # GET SETTINGS
    # ==========================================

    settings = AdminSettings.query.first()

    # ==========================================
    # CREATE DEFAULT SETTINGS
    # ==========================================

    if not settings:

        settings = AdminSettings(
            admin_name="Administrator",
            admin_email="admin@gmail.com",
            admin_password="admin123",

            # IMPORTANT
            two_factor=False,

            login_alerts=True,
            session_security=True,

            candidate_notifications=True,
            hr_notifications=True,
            job_notifications=True,
            application_notifications=True,

            allow_applications=True,
            allow_job_posting=True,
            job_approval=False,
            hr_approval=False,

            website_name="TalentBridge",
            support_email="support@talentbridge.com"
        )

        db.session.add(settings)
        db.session.commit()

    # ==========================================
    # POST
    # ==========================================

    if request.method == "POST":

        # ======================================
        # ACCOUNT SETTINGS
        # ======================================

        admin_name = request.form.get(
            "admin_name", ""
        ).strip()

        admin_email = request.form.get(
            "admin_email", ""
        ).strip()

        if admin_name:
            settings.admin_name = admin_name

        if admin_email:
            settings.admin_email = admin_email

        new_password = request.form.get(
            "new_password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if new_password:

            if new_password != confirm_password:

                flash(
                    "New password and confirm password do not match."
                )

                return redirect(
                    url_for("admin_settings")
                )

            settings.admin_password = new_password

        # ======================================
        # SECURITY SETTINGS
        # ======================================

        # IMPORTANT:
        # Because HTML contains both hidden "off"
        # and checkbox "on", use getlist()

        two_factor_values = request.form.getlist(
            "two_factor"
        )

        print("\n========================================")
        print("       2FA SETTINGS DEBUG")
        print("========================================")

        print(
            "FORM VALUES:",
            repr(two_factor_values)
        )

        print(
            "BEFORE DB VALUE:",
            repr(settings.two_factor)
        )

        # ======================================
        # DETERMINE 2FA
        # ======================================

        if "on" in two_factor_values:

            settings.two_factor = True

        else:

            settings.two_factor = False

        # ======================================
        # OTHER SECURITY
        # ======================================

        settings.login_alerts = (
            request.form.get("login_alerts") == "on"
        )

        settings.session_security = (
            request.form.get("session_security") == "on"
        )

        print(
            "AFTER ASSIGNMENT:",
            repr(settings.two_factor)
        )

        # ======================================
        # NOTIFICATIONS
        # ======================================

        settings.candidate_notifications = (
            request.form.get("candidate_notifications")
            == "on"
        )

        settings.hr_notifications = (
            request.form.get("hr_notifications")
            == "on"
        )

        settings.job_notifications = (
            request.form.get("job_notifications")
            == "on"
        )

        settings.application_notifications = (
            request.form.get("application_notifications")
            == "on"
        )

        # ======================================
        # RECRUITMENT CONTROLS
        # ======================================

        settings.allow_applications = (
            request.form.get("allow_applications")
            == "on"
        )

        settings.allow_job_posting = (
            request.form.get("allow_job_posting")
            == "on"
        )

        settings.job_approval = (
            request.form.get("job_approval")
            == "on"
        )

        settings.hr_approval = (
            request.form.get("hr_approval")
            == "on"
        )

        # ======================================
        # PLATFORM SETTINGS
        # ======================================

        settings.website_name = request.form.get(
            "website_name",
            "TalentBridge"
        )

        settings.support_email = request.form.get(
            "support_email",
            "support@talentbridge.com"
        )

        settings.contact_number = request.form.get(
            "contact_number"
        )

        settings.location = request.form.get(
            "location"
        )

        settings.description = request.form.get(
            "description"
        )

        # ======================================
        # SAVE
        # ======================================

        db.session.commit()

        # ======================================
        # FORCE DATABASE RELOAD
        # ======================================

        db.session.expire_all()

        settings = AdminSettings.query.first()

        print("\n========================================")
        print("       DATABASE AFTER COMMIT")
        print("========================================")

        print(
            "2FA FROM DATABASE:",
            repr(settings.two_factor)
        )

        print(
            "TYPE:",
            type(settings.two_factor)
        )

        print("========================================\n")

        # ======================================
        # UPDATE ADMIN SESSION
        # ======================================

        session["admin"] = settings.admin_name

        flash(
            "Admin settings updated successfully!"
        )

        return redirect(
            url_for("admin_settings")
        )

    # ==========================================
    # SYSTEM COUNTS
    # ==========================================

    total_candidates = User.query.filter_by(
        role="candidate"
    ).count()

    total_hr = User.query.filter_by(
        role="hr"
    ).count()

    total_jobs = Job.query.count()

    total_applications = Application.query.count()

    # ==========================================
    # RENDER
    # ==========================================

    return render_template(
        "admin/settings.html",
        settings=settings,
        total_candidates=total_candidates,
        total_hr=total_hr,
        total_jobs=total_jobs,
        total_applications=total_applications
    )






@app.route("/admin/disable-2fa-now")
def disable_2fa_now():

    settings = AdminSettings.query.first()

    if not settings:
        return "ERROR: AdminSettings record does not exist."

    print("")
    print("==============================================")
    print("BEFORE DISABLE")
    print("ID:", settings.id)
    print("two_factor:", repr(settings.two_factor))
    print("TYPE:", type(settings.two_factor))
    print("==============================================")

    # FORCE FALSE
    settings.two_factor = False

    db.session.commit()

    # Reload from database
    db.session.expire_all()

    settings = AdminSettings.query.first()

    print("")
    print("==============================================")
    print("AFTER DISABLE")
    print("ID:", settings.id)
    print("two_factor:", repr(settings.two_factor))
    print("TYPE:", type(settings.two_factor))
    print("==============================================")

    return f"""
    <h2>2FA Database Result</h2>

    <p><b>Admin ID:</b> {settings.id}</p>

    <p>
        <b>two_factor:</b>
        {settings.two_factor}
    </p>

    <p>
        <b>Type:</b>
        {type(settings.two_factor)}
    </p>

    <hr>

    <h3>
        {'✅ 2FA IS OFF' if settings.two_factor is False else '❌ 2FA IS STILL ON'}
    </h3>
    """








@app.route("/admin/fix-job-status")
def fix_job_status():

    try:
        db.session.execute(
            db.text("""
                ALTER TABLE job
                ADD COLUMN status VARCHAR(30) DEFAULT 'Approved'
                NOT NULL
            """)
        )

        db.session.commit()

        return """
        <h2>✅ Job Status Column Added Successfully</h2>
        <p>The <b>status</b> column has been added to the job table.</p>
        <p>Default status: <b>Approved</b></p>
        """

    except Exception as e:

        db.session.rollback()

        return f"""
        <h2>❌ Error</h2>
        <pre>{e}</pre>
        """
    




# ==================================================
# Run Application
# ==================================================

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)