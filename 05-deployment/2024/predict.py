import pickle
from flask import Flask, request ,jsonify
import gunicorn

with open("dv.bin", "rb") as file:
    dv = pickle.load(file)

with open("model1.bin", "rb") as file:
    model = pickle.load(file)

def predict_subscription_prob(customer, dv, model):
    X_customer = dv.transform(customer)
    return model.predict_proba(X_customer)[:, 1]

app = Flask('predict')


@app.route('/predict', methods=['post'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    # model.predict_proba(X)
    y_pred = model.predict_proba(X)[:, 1]
    churn = y_pred >=0.5

    result ={
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)