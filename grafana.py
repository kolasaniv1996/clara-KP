import time
import requests

# Configuration
APPNAME = 'test'
APPSECRET = 'Ns1EfpbW2gb9PxGHkd2FHKX86slEtH5P'
APPURL = 'https://vivek-inf.runailabs-cs.com'
REALM = 'runai'

def get_unix_timestamp(date_str, time_format="%Y-%m-%d %H:%M:%S"):
    """
    Convert date string to Unix timestamp.
    
    :param date_str: Date string in the format 'YYYY-MM-DD HH:MM:SS'
    :param time_format: The format of the date string.
    :return: Unix timestamp.
    """
    return int(time.mktime(time.strptime(date_str, time_format)))

def query_grafana(start_date, end_date, project_info, cluster_id, grafana_api_url, project):
    """
    Query Grafana API to get the result according to the given project.

    :param start_date: Start date string in the format 'YYYY-MM-DD HH:MM:SS'.
    :param end_date: End date string in the format 'YYYY-MM-DD HH:MM:SS'.
    :param project_info: The project info to be matched.
    :param cluster_id: Cluster ID.
    :param grafana_api_url: Grafana API URL.
    :param project: Run:AI project name.
    :return: JSON response of the Grafana query.
    """
    # Convert start and end dates to Unix timestamps
    start_timestamp = get_unix_timestamp(start_date)
    end_timestamp = get_unix_timestamp(end_date)

    # Form the payload
    payload = {
        "match[]": f"{project_info}{{clusterId=~\"{cluster_id}\"}}",
        "start": start_timestamp,
        "end": end_timestamp
    }

    # Query the Grafana API
    headers = {
        'APPNAME': APPNAME,
        'APPSECRET': APPSECRET,
        'REALM': REALM
    }

    try:
        response = requests.get(grafana_api_url, params=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error querying Grafana API: {str(e)}"

# Replace these variables with your desired values
grafana_api_url = "https://vivek-inf.runailabs-cs.com/grafana/api/datasources/uid/ab85bc66-3b02-4d59-a9d3-d6b3e12fa12d/resources/api/v1/series"
start_date = "2024-10-01 00:00:00"
end_date = "2024-10-10 23:59:59"
project_info = "runai_project_info"
cluster_id = "2da3cfee-2ea2-43f0-8e76-e6a5862a2cd2"
project = "test"

# Get results
results = query_grafana(start_date, end_date, project_info, cluster_id, grafana_api_url, project)
print(results)
