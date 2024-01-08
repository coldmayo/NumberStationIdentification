from myCNN import *
from dataPreproc import *
import pandas as pd
import pickle
import os

df = pd.read_csv('data/valData.csv')
df = df.sample(frac = 1)
X_test = df.iloc[0:1000,1:401].to_numpy()
y_test = df.iloc[0:1000,0:1].to_numpy()

labels = {'e06_':0, 'e07_':1,'e11_':2,'e25_':3,'F01_':4,'F07_':5,'hm01':6,'m01_':7,'m12_':8,'p03_':9,'p07_':10,'s06_':11,'s11a':12,'v13_':13,'XPA2':14,'xpb_':15}

# retrieve layers from pickle file
layers = []
files = os.listdir(os.getcwd() + '/layers')
for i in files:
    if i[-3:-1] == 'kl':
        with open('layers/'+i, 'rb') as file2:
            layers.append(pickle.load(file2))

y_real = np.array([])
y_pred = np.array([])

for i, (image, label) in enumerate(zip(X_test, y_test)):
    y_real = np.append(y_real,int(label))
    y = CNN_testing(image.reshape(20,20), layers)
    y = y.tolist()   # unfortunately the index function does not work with numpy arrays (1984 + communism)
    y_pred = np.append(y_pred,y.index(max(y)))
print(confusionMatrix(y_pred,y_real,labels),'\nWhere:', labels)
print('Final model accuracy',get_accuracy(y_pred,y_real))