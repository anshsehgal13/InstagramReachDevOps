from flask import Flask, request, jsonify
import os

# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNTER = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')

app = Flask(__name__)

# Try loading ML model if exists
MODEL = None
try:
    import pickle
    with open("model.pkl", "rb") as f:
        MODEL = pickle.load(f)
        print("Loaded model.pkl")
except Exception as e:
    print("No model found, using formula. (", e, ")")

@app.route('/')
def home():
    REQUEST_COUNTER.inc()
    return "Instagram Reach Predictor - running"

@app.route('/predict', methods=['GET','POST'])
def predict():
    REQUEST_COUNTER.inc()
    with REQUEST_LATENCY.time():
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args.to_dict()

        likes = float(data.get('likes', 0))
        comments = float(data.get('comments', 0))
        shares = float(data.get('shares', 0))

        if MODEL:
            pred = MODEL.predict([[likes, comments, shares]])[0]
        else:
            pred = 5*likes + 2*comments + 10*shares

        return jsonify({'prediction': float(pred)})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
