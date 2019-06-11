import pandas as pd
import numpy as np
import os 
import random

labels = '/home/christian/Documents/repositories/Model/Labeling/All_labels.csv'

labels = pd.read_csv(labels,names=['video','category','flag'])

print("Creando datasets...")

#Dividir dataset obteniendo training y test de cada video 

labels_test = labels.sample(frac=0.1)


#Eliminar rows de test en el de training
labels_train = labels.merge(labels_test.drop_duplicates(), on=['video','category','flag'], 
                   how='left', indicator=True)

labels_train = labels_train[labels_train._merge == "left_only"]
del labels_train['_merge']


print(labels.shape)
print(labels_train.shape)
print(labels_test.shape)


print(labels_test.groupby(['flag']).size())

labels_train.to_csv('training_set.csv',sep=',', header=True, index=False)
labels_test.to_csv('test_set.csv', sep=',',header=True, index=False)



print("Datasets creados con exito!")
