import pandas as pd
import requests
from pathlib import Path

PREDICT_URL = "http://localhost:8000/predict"  # adjust if needed
REQUEST_TIMEOUT = 10

root_path = Path(__file__).resolve().parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

try:
    df = pd.read_csv(data_path).dropna()
except Exception as e:
    print(f"Failed to read CSV at {data_path}: {e}")
    exit(1)

# sample row for testing the endpoint
sample_row = df.sample(1)
target_value = sample_row.iloc[:, -1].values.item().replace("(min) ", "")
print(f"The target value is {target_value}")

data = sample_row.drop(columns=[sample_row.columns[-1]]).squeeze().to_dict()
for key, value in data.items():
    if isinstance(value, str):
        data[key] = value.strip()

try:
    response = requests.post(url=PREDICT_URL, json=data, timeout=REQUEST_TIMEOUT)
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        resp_json = response.json()
        prediction = resp_json.get("predicted_delivery_time_minutes")

        if prediction is not None:
            print(f"Prediction from API: {prediction:.2f} min")
        else:
            print("Prediction key not found in response JSON.")
    else:
        print(f"API returned an error: {response.status_code} - {response.text}")

except requests.exceptions.Timeout:
    print(f"Request timed out after {REQUEST_TIMEOUT} seconds. API may be down or unreachable.")
except requests.exceptions.ConnectionError as ce:
    print(f"Connection failed: {ce}")
except Exception as e:
    print(f"Unexpected error occurred: {e}")


