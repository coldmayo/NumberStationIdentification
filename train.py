from myCNN import *
from dataPreproc import *
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('data/imgData.csv')
df = df.sample(frac = 1)
X_train = df.iloc[0:1000,1:401].to_numpy()
y_train = df.iloc[0:1000,0:1].to_numpy()
epsilon = 1e-5

numClasses = 16
layers = [ConvolutionLayer(16,2),MaxPoolingLayer(2),SoftmaxLayer(1296, numClasses)]
acc = np.array([])
loss = np.array([])

# Train the CNN
for epoch in tqdm(range(25)):   # 60 epochs
    y_pred = np.array([])
    y_real = np.array([])
    instL = 0
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    for i, (image, label) in (enumerate(zip(X_train, y_train))):
        y_real = np.append(y_real,int(label))
        y,l = CNN_training(image.reshape(20,20),int(label), layers)
        y = y.tolist()   # unfortunately the index function does not work with numpy arrays (1984 + communism)
        y_pred = np.append(y_pred,y.index(max(y)))
        instL += l
    loss = np.append(loss,instL)
    acc = np.append(acc,get_accuracy(y_pred,y_real))
    

print('Final model accuracy',get_accuracy(y_pred,y_real))
print('Final model loss:',instL)
print('X-axis is for predictions and Y-axis is for real values\n',confusionMatrix(y_pred,y_real,labels))

fig, axs = plt.subplots(2)
axs[0].plot(acc)
axs[1].plot(loss)
plt.show()

# Save the layers into Pickle files for future use (validation/testing)

layerNames = ['ConvLay','MaxPooling','SoftLay']

for i in range(len(layers)):
    name = layerNames[i]
    with open('layers/{name}.pickle'.format(name=name), 'wb') as file:
        pickle.dump(layers[i], file)