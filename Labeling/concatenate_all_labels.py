import os 
import pandas as pd
import numpy as np
import glob

LabelsDir = '/home/christian/Documents/repositories/Model/Labeling/Labels'


outfile = LabelsDir + "All_labels.csv"

os.chdir(LabelsDir)
fileList=glob.glob("*.csv")
dfList=[]
for filename in fileList:
	print(filename)
	df = pd.read_csv(filename,header = None)
	dfList.append(df)

concatDf = pd.concat(dfList,axis=0)
concatDf.to_csv(outfile, index = None)

