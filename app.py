from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('original.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs= float(request.form['fbs'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        cp = str(request.form['cp'])
        restecg = str(request.form['restecg'])
        slope = str(request.form['slope'])
        ca = str(request.form['ca'])
        thal = str(request.form['thal'])

        cp = cp.split()
        restecg = restecg.split()
        slope = slope.split()
        ca = ca.split()
        thal = thal.split()

        cp_0 = float(cp[0])
        cp_1 = float(cp[1])
        cp_2 = float(cp[1])
        cp_3 = float(cp[3])

        restecg_0 = float(restecg[0])
        restecg_1 = float(restecg[1])
        restecg_2 = float(restecg[2])

        slope_0 = int(slope[0])
        slope_1 = int(slope[1])
        slope_2 = int(slope[2])

        ca_0 = float(ca[0])
        ca_1 = float(ca[1])
        ca_2 = float(ca[2])
        ca_3 = float(ca[3])
        ca_4 = float(ca[4])

        thal_0 = float(thal[0])
        thal_1 = float(thal[1])
        thal_2 = float(thal[2])
        thal_3 = float(thal[3])

        pred_args = [age,sex,trestbps,chol,fbs,thalach,exang,oldpeak,cp_0,cp_1,cp_2,cp_3,restecg_0,restecg_1,restecg_2,slope_0,slope_1,slope_2,ca_0,ca_1,ca_2,ca_3,ca_4,thal_0,thal_1,thal_2,thal_3]

        mul_reg = open('heart_svm.pkl','rb')
        ml_model = joblib.load(mul_reg)
        model_predcition = ml_model.predict([pred_args])
        if model_predcition == 1:
            res = 'Affected'
        else:
            res = 'Not affected'
        #return res
    return render_template('predict.html', prediction = res)

if __name__ == '__main__':
    app.run()
