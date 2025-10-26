

### Question 1
# running "uv --version" on bash outputs: uv 0.9.5 (d5f39331a 2025-10-21)

### Question 2
# sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e


### Question 3
import pickle

with open('pipeline_v1.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
    pipeline = pickle.load(f_in)
    
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X_val = dv.transform(record)

pred = model.predict_proba(X_val)[0, 1]

print(pred) # 0.5336072702798061


pred_2 = pipeline.predict_proba(record)[0, 1]
print(pred_2)

# Note to self - Need to run the above while on uv environment (i.e. after "uv init")


### Question 4
import requests

url = 'http://localhost:9696/predict'

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)
predictions = response.json()

print(predictions) # {'pred_probability': 0.5340417283801275, 'pred_decision': True}