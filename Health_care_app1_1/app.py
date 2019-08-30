from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
    #Sending data to home page
    values = pd.read_excel('PLX-DAQ.xlsx', sheet_name='Simple Data')
    values = values.drop(['time', 'timer', 'date'], axis=1)
    m = np.array(values['MOISTURE'])
    moisture = m[0]
    t = np.array(values['TEMPERATURE'])
    temperature = t[0]
    p = np.array(values['PULSE'])
    pulse = p[0]

    dat = {"moisture": moisture, "temperature": temperature, "pulse": pulse}

    return render_template('home.html',task=dat)
    #return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    cv = CountVectorizer()
    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    tree_model = open('accuracy91_22.sav','rb')
    clf = joblib.load(tree_model)

    if request.method == 'POST':
        moisture = request.form['moisture']
        temperature=request.form['temperature']
        pulse = request.form['pulse']

        data = np.array([[moisture,temperature,pulse]])
        #vect = cv.transform(data).toarray()
        my_prediction = clf.predict(data)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
