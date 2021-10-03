from flask import Flask, render_template, request
import pandas as pd
import sklearn
import joblib

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


@app.route('/prediction')
def predict():
    return render_template('predict.html')


@app.route('/prediction1', methods=["POST", "GET"])
def pred():
    s = []
    if request.method == "POST":
        a = request.form['mr']
        b = request.form['mt']
        c = request.form['mp']
        d = request.form['ma']
        e = request.form['ms']
        f = request.form['mc']
        g = request.form['mk']
        h = request.form['mcc']
        i = request.form['mcp']

        s.extend([a, b, c, d, e, f, g, h, i])
        model = joblib.load('ITC.pkl')
        y_pred = model.predict([s])
        return render_template('predict.html', msg="success", op=y_pred)


if __name__ == '__main__':
    app.secret_key = "hai"
    app.run(debug=True)
