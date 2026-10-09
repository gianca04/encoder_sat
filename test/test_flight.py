import flightsql
import os
from dotenv import load_dotenv

load_dotenv('.env', override=True)
token = os.getenv('INFLUXDB_TOKEN')
db = os.getenv('INFLUXDB_DATABASE')
host = os.getenv('INFLUXDB_HOST')

print(f"Token: {token}")

client = flightsql.FlightSQLClient(
    host=host,
    port=8182,
    insecure=True,
    metadata={'database': db, 'authorization': f'Bearer {token}'}
)
try:
    conn = flightsql.connect(client)
    print("Connected via FlightSQL!")
    cursor = conn.cursor()
    cursor.execute('SELECT time as timestamp, "PIT_001/Pressure" as PIT_001 FROM plc_siemens_lab LIMIT 1')
    print(cursor.fetchall())
except Exception as e:
    print(f"Error: {e}")
