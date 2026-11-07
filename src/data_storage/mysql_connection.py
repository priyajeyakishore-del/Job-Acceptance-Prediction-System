
from sqlalchemy import create_engine, URL

# MySQL database configuration
DB_USER = "root"
DB_PASSWORD = "Dinnu@world"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "job_acceptance_db"

# Safely construct MySQL connection URL
database_url = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

# Create SQLAlchemy engine
engine = create_engine(
    database_url,
    pool_pre_ping=True
)


def get_connection():
    return engine.connect()
