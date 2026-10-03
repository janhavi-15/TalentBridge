import joblib

# Load trained model
model = joblib.load("job_field_model.pkl")


test_jobs = [

    {
        "title": "Data Analyst",
        "description": "Analyze business data and create reports and dashboards.",
        "skills": "Python, SQL, Power BI, Excel"
    },

    {
        "title": "Financial Analyst",
        "description": "Analyze financial statements, prepare budgets and financial reports.",
        "skills": "Excel, Financial Modeling, Accounting, SQL"
    },

    {
        "title": "Financial Data Analyst",
        "description": "Analyze financial datasets and prepare financial performance reports.",
        "skills": "Python, SQL, Excel, Financial Modeling"
    },

    {
        "title": "Healthcare Data Analyst",
        "description": "Analyze patient data and healthcare performance metrics.",
        "skills": "SQL, Python, Power BI, Healthcare Analytics"
    },

    {
        "title": "Software Developer",
        "description": "Develop and maintain web applications and backend services.",
        "skills": "Java, Spring Boot, Python, REST API"
    },

    {
        "title": "HR Executive",
        "description": "Manage recruitment, employee relations and employee onboarding.",
        "skills": "Recruitment, HR, Communication, Payroll"
    },

    {
        "title": "Marketing Analyst",
        "description": "Analyze marketing campaigns, customer trends and campaign performance.",
        "skills": "Marketing Analytics, Excel, SQL, Power BI"
    },

    {
        "title": "Sales Operations Analyst",
        "description": "Analyze sales data, CRM information and revenue performance.",
        "skills": "Sales Analytics, Excel, SQL, CRM, Power BI"
    },

    {
        "title": "Maths Professor",
        "description": "Teach mathematics to students, prepare lesson plans, conduct assessments and explain mathematical concepts.",
        "skills": "Mathematics, Teaching, Classroom Management, Communication"
    },

    {
        "title": "Graphic Designer",
        "description": "Create visual designs for digital and print media, branding, advertisements and promotional materials.",
        "skills": "Adobe Photoshop, Illustrator, InDesign, Canva, Graphic Design"
    },

    {
        "title": "Staff Nurse",
        "description": "Provide nursing care, monitor patients and assist doctors with clinical procedures.",
        "skills": "Patient Care, Nursing, Clinical Documentation"
    },
    {
    "title": "Logistics Executive",
    "description": "Coordinate daily logistics and supply chain operations. Manage shipment scheduling, transportation coordination, inventory movement, warehouse activities and delivery tracking.",
    "skills": "Logistics, Supply Chain, Transportation, Inventory Management, Warehouse Management"
},
{
    "title": "Warehouse Operations Executive",
    "description": "Manage warehouse receiving, storage, stock movement, order picking, packing and dispatch operations.",
    "skills": "Warehouse Operations, Inventory Control, Stock Management, Dispatch, Logistics"
},
{
    "title": "Distribution Officer",
    "description": "Coordinate the movement of products from warehouses to stores, monitor deliveries and manage transportation schedules.",
    "skills": "Distribution, Transportation, Delivery Management, Inventory, Logistics"
},
{
    "title": "Supply Planning Executive",
    "description": "Monitor material requirements, inventory levels and supplier deliveries to ensure uninterrupted product availability.",
    "skills": "Supply Planning, Inventory Management, Procurement, Supply Chain, Excel"
},
{
    "title": "Warehouse Operations Supervisor",
    "description": "Supervise warehouse staff, monitor stock movement, coordinate receiving and dispatch and maintain inventory accuracy.",
    "skills": "Warehouse Management, Inventory, Dispatch, Stock Control, Logistics"
}
]


# Test model
for job in test_jobs:

    text = (
        "TITLE " + job["title"] + " " +
        "DESCRIPTION " + job["description"] + " " +
        "SKILLS " + job["skills"]
    )

    prediction = model.predict([text])[0]

    print("--------------------------------")
    print("Job:", job["title"])
    print("Predicted Field:", prediction)