
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load #to_save_model
from sklearn import svm
import os 
from os import path
import pandas as pd
import numpy as np


def load_all_dataset():

	#Funcion para guardar todo el dataset en un solo txt para ahorrar tiempo en ejecucion. 

	labels = '/home/christian/Documents/repositories/Model/Labeling/All_labels.csv'
	FeaturesTxt_path = '/home/christian/Desktop/Tesis/AnomalyDetectionCVPR2018-master/Data/Avg/'	

	labels = pd.read_csv(labels,names=['video','category','flag'])

	all_videos = labels['video']

	all_videos = all_videos + ".mp4_C.txt"

	videos_not_found = []

	#Inicializacion de la matriz

	features_x = np.empty((16856,4097))
	#y_all = np.empty((16856,2),dtype=int)
	x = 0
	y = 0

	for video in all_videos:
		
		feature_txt = FeaturesTxt_path + video

		try:
			feature = pd.read_csv(feature_txt, sep = " ", header = None, nrows = 1)
			features_x[x] = feature
			features_x[x][4096] = labels.flag[y]

			print(x)
			x = x + 1 

		except:

			videos_not_found.append(video)
			print("Video no encontrado: " + video)

		y=y+1

	#Se elimina la ultima columna que contiene NaN
	#features_x = np.delete(features_x,4096,1)

	#Se elimina los rows de exceso que estan llenos de 0 
	features_x = np.delete(features_x,np.s_[x:],axis=0)
	#y_all = np.delete(y_all,np.s_[x:],axis=0)
	

	np.savetxt("all_features_y.txt", features_x, fmt = "%.6f",delimiter = ",", newline = "\n")
	#np.savetxt("flag_y_all.txt", y_all, fmt = "%d", delimiter = ",", newline = "\n")

	#np.savetxt("training_not_found.txt", videos_not_found, fmt = "%.6f", delimiter = ",", newline = "\n")

	print("Finished ! ")


def cross_validation_MLP():

	k_folds = 10

	features_x = np.loadtxt("all_features_y.txt", delimiter = ",")
	
	where_are_NaNs = np.isnan(features_x)
	features_x[where_are_NaNs] = 0

	for x in range (1,k_folds + 1): 

		print("FOLD NUMBER: " + str(x))

		second_size = 1681 / 16815

		print("Train and test split....")

		train, test = train_test_split(features_x, test_size=second_size)

		print(train.shape)
		print(test.shape)

		#Extraccion label 
		y_train = train[:,4096]
		y_test = test[:,4096]

		#ELiminacion label en x
		x_train = np.delete(train,4096,1)
		x_test = np.delete(test,4096,1)


		#Guardamos dataset_test para futuras pruebas, training test no es neceasario porque
		#se guardara el modelo entrenado 
		name_y_test = "y_test_k" + str(x) + ".txt"
		name_x_test = "x_test_k" + str(x) + ".txt"

		np.savetxt(name_y_test, y_test, fmt = "%d", delimiter = ",", newline = "\n")
		np.savetxt(name_x_test, x_test, fmt = "%.6f", delimiter = ",", newline = "\n")


		print("Creating Model..")

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
		model.fit(x_train, y_train ,epochs=200, batch_size=512, verbose = 1)

		print("Testing model...")
		score = model.evaluate(x_test, y_test, batch_size=1024, verbose = 1)

		print(model.metrics_names)
		print(score)

		
		print("Saving model...")
		model_json = model.to_json()

		model_name = "MLP_K" + str(x) + ".json"
		weights_name = "MLP_K" + str(x) + ".h5"

		with open(model_name,"w") as json_file:
			json_file.write(model_json)

		model.save_weights(weights_name)
	
		y_pred = model.predict(x_test, batch_size = 512, verbose = 1)


		print (y_pred.shape)
		print (y_pred)

		y_pred_name = "y_pred_k" + str(x) + ".txt"
	
		np.savetxt(y_pred_name, y_pred, fmt = "%.6f", delimiter = ",", newline = "\n")


		#Calculo AUC
		fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred, pos_label = 1)

		auc_keras= auc(fpr_keras, tpr_keras)

		print("AUC:")
		print(auc_keras)


		#Guardamos resultados en txt
		f = open("result_MLP.txt","w+")

		result = "MLP_K" + str(x) + "  AUC: " + str(auc_keras) + "\n"

		f.write(result)

		f.close()



def cross_validation_Bayes():


	k_folds = 10

	features_x = np.loadtxt("all_features_y.txt", delimiter = ",")

	#Se detecto que hay algunos valores del etiquetado vacios

	where_are_NaNs = np.isnan(features_x)
	features_x[where_are_NaNs] = 0

	for x in range (1,k_folds + 1): 

		print("FOLD NUMBER: " + str(x))

		second_size = 1681 / 16815

		print("Train and test split....")

		train, test = train_test_split(features_x, test_size=second_size)

		print(train.shape)
		print(test.shape)


		#Extraccion label 
		y_train = train[:,4096]
		y_test = test[:,4096]

		#ELiminacion label en x
		x_train = np.delete(train,4096,1)
		x_test = np.delete(test,4096,1)

		print("Creating Model..")
		print(x_train)
		print(y_train)


		gnb = GaussianNB()

		y_pred =gnb.fit(x_train,y_train).predict(x_test)

		#dump(gnb, 'Naive_Bayes_model.joblib')
		print("Testing")
		print(y_pred)

		#Calculo AUC
		fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred, pos_label = 1)

		auc_keras= auc(fpr_keras, tpr_keras)

		print("AUC:")
		print(auc_keras)


		#Guardamos resultados en txt
		f = open("result_Bayes.txt","a")

		result = "Bayes_K" + str(x) + "  AUC: " + str(auc_keras) + "\n"

		f.write(result)

		f.close()


def cross_validation_KN():

	k_folds = 10

	features_x = np.loadtxt("all_features_y.txt", delimiter = ",")

	#Se detecto que hay algunos valores del etiquetado vacios
	where_are_NaNs = np.isnan(features_x)
	features_x[where_are_NaNs] = 0

	for x in range (1,k_folds + 1): 

		print("FOLD NUMBER: " + str(x))

		second_size = 1681 / 16815

		print("Train and test split....")

		train, test = train_test_split(features_x, test_size=second_size)

		print(train.shape)
		print(test.shape)


		#Extraccion label 
		y_train = train[:,4096]
		y_test = test[:,4096]

		#ELiminacion label en x
		x_train = np.delete(train,4096,1)
		x_test = np.delete(test,4096,1)


		print("Creating Model..")
		print(x_train)
		print(y_train)


		neigh = KNeighborsClassifier(n_neighbors = 32)
		neigh.fit(x_train,y_train)

		print("Testing")
		y_pred = neigh.predict_proba(x_test)
		y_pred = y_pred[:,1]

		#dump(gnb, 'Naive_Bayes_model.joblib')

		print(y_pred)

		#Calculo AUC
		fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred, pos_label = 1)

		auc_keras= auc(fpr_keras, tpr_keras)

		print("AUC:")
		print(auc_keras)


		#Guardamos resultados en txt
		f = open("result_KN.txt","a")

		result = "KN_K" + str(x) + "  AUC: " + str(auc_keras) + "\n"

		f.write(result)

		f.close()



def cross_validation_SVM():

	k_folds = 10

	features_x = np.loadtxt("all_features_y.txt", delimiter = ",")

	#Se detecto que hay algunos valores del etiquetado vacios

	where_are_NaNs = np.isnan(features_x)
	features_x[where_are_NaNs] = 0

	for x in range (1,k_folds + 1): 

		print("FOLD NUMBER: " + str(x))

		second_size = 1681 / 16815

		print("Train and test split....")

		train, test = train_test_split(features_x, test_size=second_size)

		print(train.shape)
		print(test.shape)


		#Extraccion label 
		y_train = train[:,4096]
		y_test = test[:,4096]

		#ELiminacion label en x
		x_train = np.delete(train,4096,1)
		x_test = np.delete(test,4096,1)

		print("Creating Model..")
		print(x_train)
		print(y_train)


		clf = svm.SVC(gamma = 'scale', verbose = True)
		clf.fit(features_x_train,y_train)

		y_pred = clf.predict(features_x_test)

		print("Testing")
		print(y_pred)

		#Calculo AUC
		fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred, pos_label = 1)

		auc_keras= auc(fpr_keras, tpr_keras)

		print("AUC:")
		print(auc_keras)

		#Guardamos resultados en txt
		f = open("result_SVM.txt","a")

		result = "SVM_K" + str(x) + "  AUC: " + str(auc_keras) + "\n"

		f.write(result)

		f.close()


#Call def here


