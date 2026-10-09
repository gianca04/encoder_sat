from influxdb_client import InfluxDBClient

url = "http://192.168.10.145:8182"
token = "apiv3_75b75fda71d81cda9bd2b417aaa5ea678b31467fd940d093"
org = "sat_lab" # Sometimes org is same as db in InfluxDB 3 Edge

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
query = 'from(bucket: "sat_lab") |> range(start: -1m)'
try:
    result = query_api.query(org=org, query=query)
    print("Success. Records:", len(result))
except Exception as e:
    print(e)
