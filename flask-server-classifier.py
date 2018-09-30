# Load libraries
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from flask import Flask,request,jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	json_ = request.json
	print(json_)
	queryVector = [float(numeric_string) for numeric_string in json_["vector"].split(',')]
	print(queryVector)
	prediction = clf.predict([queryVector])
	predBool = True
	if (prediction[0]==0):
		predBool = False
	return jsonify({'prediction': predBool})

if __name__ == '__main__':
	nnlocation = sys.argv[1]
	clf = joblib.load(nnlocation)
	app.run(port=3333)
