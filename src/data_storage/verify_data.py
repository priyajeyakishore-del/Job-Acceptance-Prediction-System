import pandas as pd
from mysql_connection import engine

query = """
SELECT COUNT(*) AS total_records
FROM job_acceptance_data;
"""

result = pd.read_sql(query, engine)

print("Total records stored in MySQL:")
print(result)

# Display sample records
sample_query = """
SELECT *
FROM job_acceptance_data
LIMIT 5;
"""

sample_data = pd.read_sql(sample_query, engine)

print("\nSample records:")
print(sample_data)