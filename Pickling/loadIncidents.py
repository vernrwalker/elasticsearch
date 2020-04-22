import os
from dotenv import load_dotenv
load_dotenv()

def servicenow_rest_call():
    # Need to install requests package for python
    # easy_install requests
    import requests
    import json
    # Set the request parameters
    url = os.getenv("SN_URL")
    # Eg. User name="admin", Password="admin" for this code sample.
    user = os.getenv("SN_USER")
    pwd = os.getenv("SN_PWD")
    # Set proper headers
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}
    # Do the HTTP request
    response = requests.get(url, auth=(user, pwd), headers=headers)
    # Check for HTTP codes other than 200
    if response.status_code != 200:
        print('Status:', response.status_code, 'Headers:',
              response.headers, 'Error Response:', response.json())
        exit()

    # Decode the JSON response into a dictionary and use the data
    #data = response.json()

    # Load incident data with manipulated severity levels including 2
    jsonf = open("incidentData.json")
    data = json.load(jsonf)

    #with open('incidentData.json', 'w') as f:
    #    json.dump(data, f)
    # print(data)
    return(data)


def Servicenow_pickler(incident_data):

    # Imports of import.
    import json
    import os
    import pandas as pd

    # Creating the dataframe into which all of the files will be stored:
    df = pd.DataFrame()
    data = incident_data

    # ...and creating new lists for the texts of the sentences...
    df_description = []
    df_severity = []

    # ...and adding the sentences to those new lists...
    for sent in data['result']:

        # ...creating the 'Sentences'...
        df_severity.append(sent['severity'])
        df_description.append(sent['description'])

    # ...and adding those to the previously instantiated dataframe:
    df['Sentences'] = df_description
    df['Severity'] = df_severity

    # Pickling the dataframe:
    df.to_pickle("incidents.pkl")

    # Now to pass the fact that this has completed as the return statement:
    pickled = 'Done'
    return pickled


incident_data = servicenow_rest_call()
Servicenow_pickler(incident_data)
