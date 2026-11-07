
from mysql_connection import engine

try:
    with engine.connect() as connection:
        result = connection.exec_driver_sql(
            "SELECT VERSION()"
        )

        print("MySQL connection successful!")
        print("MySQL Version:", result.fetchone()[0])

except Exception as error:
    print("MySQL connection failed!")
    print("Error:", error)
