from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
from train import sales_prediction


app = Flask(__name__)
sls=sales_prediction()

@cross_origin()
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def prediction():
    if request.method == 'POST':
        try:
            tv = float(request.form['TV'])
            radio = float(request.form['Radio'])
            newspaper = float(request.form['News paper'])
            l=[]
            l.append(tv)
            l.append(radio)
            l.append(newspaper)
            m=[]
            m.append(l)
            saved_model = pickle.load(open("ML2ndmodel", 'rb'))
            prediction = sls.get_prediction(data=m)
            prediction=prediction[0]
            print("prediction is ", prediction)
            return render_template("results.html", prediction=prediction)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()

