import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app= Flask(__name__)
model= pickle.load(open('Loans_to_students1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Dep.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    int_f = [int(x) for x in request.form.values()]
    final_f = [np.array(int_f)]
    p = model.predict(final_f)
    return render_template('Dep.html', prediction_text='If 1 then yes for loan elgibility/else no Output= {}'.format(p))

@app.route('/prediction_api',methods=['POST'])
def prediction_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True,port=5002)



