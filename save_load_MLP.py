from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Activation,LSTM,Reshape
from keras.regularizers import l2
from keras.initializers import RandomUniform, Ones
from keras.optimizers import SGD,adam, Adagrad, Adadelta
from sklearn.metrics import confusion_matrix
from os import path

import pandas as pd
import numpy as np



def Load_features_training_set():

	FeaturesTxt_path = '/home/christian/Desktop/Tesis/AnomalyDetectionCVPR2018-master/Data/Avg/'

	training_path = '/home/christian/Documents/repositories/Model/training_set.csv'

	training_set = pd.read_csv(training_path)

	#Enlistamos los nombres de los videos en el dataset de training
	videos_training = training_set['video']

	#Agregamos la extención de los features en txt 
	videos_training = videos_training + ".mp4_C.txt"

	videos_not_found = []

	#Inicializacion de la matriz
	features_x_train = np.empty((15167,4097))
	y_train = np.empty((15167,1),dtype=int)
	x = 0
	y = 0

	for video in videos_training:
		
		feature_txt = FeaturesTxt_path + video

		try:
			feature = pd.read_csv(feature_txt, sep = " ", header = None, nrows = 1)
			features_x_train[x] = feature
			y_train[x] = training_set.flag[y]
			print(x)
			#print(feature)
			x = x + 1 

		except:
			videos_not_found.append(video)
			print (video)



		y=y+1

	#Se elimina la ultima columna que contiene NaN
	features_x_train = np.delete(features_x_train,4096,1)

	#Se elimina los rows de exceso que estan llenos de 0 
	features_x_train = np.delete(features_x_train,np.s_[x:],axis=0)
	y_train = np.delete(y_train,np.s_[x:],axis=0)
	#print(y_train)
	
	np.savetxt("feature_training.txt", features_x_train, fmt = "%.6f",delimiter = ",", newline = "\n")
	np.savetxt("y_train.txt", y_train, fmt = "%d", delimiter = ",", newline = "\n")


	print("End!")

	return features_x_train, y_train


def Load_features_test_set():

	FeaturesTxt_path = '/home/christian/Desktop/Tesis/AnomalyDetectionCVPR2018-master/Data/Avg/'

	test_path = '/home/christian/Documents/repositories/Model/test_set.csv'

	test_set = pd.read_csv(test_path)

	#Enlistamos los nombres de los videos en el dataset de test
	videos_test = test_set['video']

	#Agregamos la extención de los features en txt 
	videos_test = videos_test + ".mp4_C.txt"

	videos_not_found = []

	#Inicializacion de la matriz
	features_x_test = np.empty((1684,4097))
	y_test = np.empty((1684,1),dtype=int)
	x = 0
	y = 0

	for video in videos_test:
		
		feature_txt = FeaturesTxt_path + video

		try:
			feature = pd.read_csv(feature_txt, sep = " ", header = None, nrows = 1)
			features_x_test[x] = feature
			y_test[x] = test_set.flag[y]
			x = x + 1 

		except:
			videos_not_found.append(video)
			print(video)

		y = y + 1


	#Se elimina la ultima columna que contiene NaN
	features_x_test = np.delete(features_x_test,4096,1)

	#Se elimina los rows de exceso que estan llenos de 0 
	features_x_test = np.delete(features_x_test,np.s_[x:],axis=0)
	y_test = np.delete(y_test,np.s_[x:],axis=0)
	#print(y_test)
	
	np.savetxt("feature_test.txt", features_x_test, fmt = "%.6f",delimiter = ",", newline = "\n")
	np.savetxt("y_test.txt", y_test, fmt = "%d", delimiter = ",", newline = "\n")


	return features_x_test, y_test



def save_multilayer_perceptron():

	features_x_train = np.loadtxt("feature_training.txt", delimiter = ",")
	features_x_test = np.loadtxt("feature_test.txt", delimiter = ",")

	y_train= np.loadtxt("y_train.txt", delimiter = ",")
	y_test = np.loadtxt("y_test.txt", delimiter = ",")


	print("Create Model..")

	model = Sequential()

	random_uniform = RandomUniform(minval=-0.05, maxval=0.05, seed=None)
	ones = Ones()

	model.add(Dense(512, input_dim=4096,kernel_initializer="glorot_normal",kernel_regularizer=l2(0.001),activation='relu'))
	model.add(Dropout(0.6))
	model.add(Dense(32,kernel_initializer="glorot_normal",kernel_regularizer=l2(0.001)))
	#model.add(Dropout(0.5))
	#model.add(Dense(16,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001)))
	model.add(Dropout(0.6))
	model.add(Dense(1,kernel_initializer="glorot_normal",kernel_regularizer=l2(0.001),activation='sigmoid'))

	adagrad=Adagrad(lr=0.01, epsilon=1e-08)

	model.compile(loss='mse', optimizer=adagrad,metrics=['accuracy'])

	print ("Training model...")
	model.fit(features_x_train, y_train ,epochs=200, batch_size=1024, verbose = 1)

	print("Testing model...")
	score = model.evaluate(features_x_test, y_test, batch_size=1024, verbose = 1)

	print(model.metrics_names)
	print(score)

	
	print("Saving model...")
	model_json = model.to_json()

	#model_path = '/home/christian/Documents/repositories/Model/Models/'
	#model_name = 'model_MLP'
	#model_config_path = path.relpath("Models/model_MLP.json")
	#model_weight_path = path.relpath("Models/model_MLP.h5")
	
	with open("model_MLP.json","w") as json_file:
		json_file.write(model_json)

	model.save_weights("model_MLP.h5")
	
	
	y_pred = model.predict(features_x_test, batch_size = 1024, verbose = 1)


	print (y_pred.shape)
	print (y_pred)
	
	np.savetxt("y_predict.txt", y_pred, fmt = "%.6f", delimiter = ",", newline = "\n")
	
	print (y_pred)
	
	ROC_AUC(y_test,y_pred)


def ROC_AUC(y_test,y_pred):

	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve
	from sklearn.metrics import auc

	#confusion = confusion_matrix(y_test, y_pred)
	#print (confusion)

	fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred, pos_label = 1)

	auc_keras= auc(fpr_keras, tpr_keras)

	#print ("AUC: " + auc_keras)

	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
	#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()



def load_multilayer_perceptron():

	print("Loading model ...")

	with open ("Models/model_num.json","r") as f:
		model = model_from_json(f.read())

	model.load_weights("Models/model_num.h5")


	print("Compiling model ...")

	adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

	model.compile(loss='mse', optimizer= adadelta ,metrics=['accuracy'])


	print("Testing model ...")

	features_x_test = np.loadtxt("feature_test.txt", delimiter = ",")
	y_test = np.loadtxt("y_test.txt", delimiter = ",")

	score = model.evaluate(features_x_test, y_test, batch_size=128)

	print(model.metrics_names)
	print(score)

	y_pred = model.predict(features_x_test, batch_size = 128, verbose = 1)

	ROC_AUC(y_test,y_pred)



def SVM():

	from sklearn import svm
	from joblib import dump, load

	features_x_train = np.loadtxt("feature_training.txt", delimiter = ",")
	features_x_test = np.loadtxt("feature_test.txt", delimiter = ",")

	y_train= np.loadtxt("y_train.txt", delimiter = ",")
	y_test = np.loadtxt("y_test.txt", delimiter = ",")

	clf = svm.SVC(gamma = 'scale', verbose = True)
	clf.fit(features_x_train,y_train)

	dump(clf, 'SVM_model.joblib')

	scor = clf.score(features_x_test,y_test)

	y_pred = clf.predict(features_x_test)

	print(scor)
	print(y_pred)

	np.savetxt("y_predict_SVM.txt", y_pred, fmt = "%.6f", delimiter = ",", newline = "\n")

	ROC_AUC(y_test,y_pred)


def Gaussian_Naive_Bayes():

	from sklearn.naive_bayes import GaussianNB
	from joblib import dump, load

	features_x_train = np.loadtxt("feature_training.txt", delimiter = ",")
	features_x_test = np.loadtxt("feature_test.txt", delimiter = ",")

	y_train= np.loadtxt("y_train.txt", delimiter = ",")
	y_test = np.loadtxt("y_test.txt", delimiter = ",")

	gnb = GaussianNB()

	y_pred =gnb.fit(features_x_train,y_train).predict_proba(features_x_test)

	dump(clf, 'Naive_Bayes_model.joblib')

	print(y_pred)

	#np.savetxt("y_predict_prob_GaussianNaiveBayes.txt", y_pred, fmt = "%.6f", delimiter = ",", newline = "\n")	
	
	y_pred = y_pred[:,1]
	print(y_pred)
	ROC_AUC(y_test,y_pred)


def K_NeighborsClassifier():

	from sklearn.neighbors import KNeighborsClassifier
	from joblib import dump, load

	features_x_train = np.loadtxt("feature_training.txt", delimiter = ",")
	features_x_test = np.loadtxt("feature_test.txt", delimiter = ",")

	y_train= np.loadtxt("y_train.txt", delimiter = ",")
	y_test = np.loadtxt("y_test.txt", delimiter = ",")

	from sklearn.neighbors import KNeighborsClassifier

	neigh = KNeighborsClassifier(n_neighbors = 32)
	neigh.fit(features_x_train,y_train)

	y_pred = neigh.predict_proba(features_x_test)
	y_pred = y_pred[:,1]
	print(y_pred)

	ROC_AUC(y_test,y_pred)

#main 



#Load_features_test_set()
#Load_features_training_set()
save_multilayer_perceptron()
