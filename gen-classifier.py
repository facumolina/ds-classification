# Load libraries
import sys, ast
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand

from time import time
import mglearn

# Show confusion matrix
def showconfusionmatrix(cmatrix):
	totalpositives = cmatrix[1][0]+cmatrix[1][1]
	errorpositives = cmatrix[1][0]
	totalnegatives = cmatrix[0][0]+cmatrix[0][1]
	errornegatives = cmatrix[0][1]
	print('Total positives: %d' % (totalpositives))
	print('Errors predicting positives: %d ' % (errorpositives))
	print('')
	print('Total negatives: %d' % (totalnegatives))
	print('Errors predicting negatives: %d' % (errornegatives))

# Create features list
def createFeaturesList(total):
	featureList = []
	for i in range(0,total):
		featureList.append("feature"+str(i))
	return featureList

if __name__ == '__main__':

	# Get the data set location arugment
	dataSetLocation = sys.argv[1]
	testSetLocation = sys.argv[2]
	total_features = len(pandas.read_csv(str(dataSetLocation)).columns)-1
	feature_names = createFeaturesList(total_features)

	# Load dataset
	print('Init')
	print('----')
	print('Loading data from: '+str(dataSetLocation))
	print('Testing with file: '+str(testSetLocation))
	print('Total features: '+str(total_features))
	print('')

	train_dataset = pandas.read_csv(str(dataSetLocation), names=(feature_names+['class']))
	test_dataset = pandas.read_csv(str(testSetLocation), names=(feature_names+['class']))

	X_train = train_dataset.values[:,0:total_features]
	Y_train = train_dataset.values[:,total_features]
	XTest = test_dataset.values[:,0:total_features]
	YTest = test_dataset.values[:,total_features]
	
	# Report total instances
	print('Instances')
	print('---------')
	print('Training: '+str(X_train.shape[0]))
	print('Test: '+str(XTest.shape[0]))
	print('')

	# MultiLayer Perceptron classifier
	hiddenlayersize=total_features
	mlp = MLPClassifier(alpha=1.e-05,hidden_layer_sizes=[hiddenlayersize],activation='tanh',solver='lbfgs',max_iter=2000,random_state=2);

	# Report classifier
	print('Feed forward neural network')
	print('---------------------------')
	print('Input layer size: '+str(total_features))
	print('Hidden layers size: '+str(hiddenlayersize))
	print('Output layer size: 1')
  
  # Define the possible values of some parameters in order to perform a Random search
	maxhiddenlayer=100
	hiddenlayerunitsoptions = []
	for i in list(range(2,maxhiddenlayer+1)):
		hiddenlayerunitsoptions.append([i])

	alphas = np.logspace(-5, 3, 5)

 	# Training and hyperparameter optimization
	param_grid = {'alpha' : alphas,'hidden_layer_sizes':hiddenlayerunitsoptions}
	t0 = time()
	rsearch = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid,n_iter=10,scoring='roc_auc')
	rsearch.fit(X_train, Y_train)
	print(rsearch)
	# Summarize the results of the random parameter search
	print(rsearch.best_params_)
	print('Paramterr opt time: %0.3fs' % (time() - t0))
	print('Parameter optimization finished')
	print()

	# Determine the best model
	mlp = MLPClassifier(alpha=rsearch.best_estimator_.alpha,activation='tanh',solver='lbfgs',hidden_layer_sizes=rsearch.best_estimator_.hidden_layer_sizes,max_iter=2000,random_state=2);
	t0 = time()
	mlp.fit(X_train, Y_train)
	print('Training time: %0.3fs' % (time() - t0))
	prediction = mlp.predict(XTest)
	score = accuracy_score(YTest,prediction)
	print('')

	# Show the results on the validation set
	print('Testing')
	print('-------')
	print('Total accuracy: %f' % score)
	print('')
	cmatrix = confusion_matrix(YTest,prediction,labels=range(2))
	showconfusionmatrix(cmatrix)
	print()
	print('Classification report')
	print(classification_report(YTest,prediction))
	
	# Write the classifier to a file
	joblib.dump(mlp, 'mlpclassifier.joblib.pkl', compress=9)

