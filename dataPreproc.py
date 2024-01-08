import matplotlib.pyplot as plt
from scipy.io import wavfile
from PIL import Image
from scipy.signal import spectrogram
from pylab import *
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

labels = {'e06_':0, 'e07_':1,'e11_':2,'e25_':3,'F01_':4,'F07_':5,'hm01':6,'m01_':7,'m12_':8,'p03_':9,'p07_':10,'s06_':11,'s11a':12,'v13_':13,'XPA2':14,'xpb_':15}

def value(d, val):
    keys = [k for k, v in d.items() if v == val]
    ans = 0
    if keys:
        ans = keys[0]
    else:
        ans = None
    return ans

def checkCSV(labels,df):
    label = df['y']
    labNames = list(labels.values())
    owned = []
    nums = np.zeros(len(labNames))
    notOwned = list(labels.keys())
    for i in label:
        for j in labNames:
            if int(i) == int(j):
                owned.append(str(value(labels,i)))
                nums[i] += 1
                notOwned = [k for k in notOwned if k!=str(value(labels,i))]
    owned = [*set(owned)]
    print('Number Station Data in CSV:',owned,nums,'\nNumber Station Data not in CSV',notOwned)


def makePics(wav,i):
    # x label: time (seconds), y label: frequency (Hz)
    Fs, samples = wavfile.read('soundFiles/{name}.wav'.format(name=wav))
    noise = np.random.normal(0,0.5,len(samples))    # add random noise
    if samples.shape != noise.shape:
        samples = samples[:,0]
    samples = samples+noise
    interval = int(Fs)
    overlap = int(Fs * 0.95)
    f, t, Sxx = spectrogram(samples,fs=Fs,nperseg=interval,noverlap=overlap)
    pcolormesh(t, f, 10 * log10(Sxx), cmap='jet')
    fmax = max(f)
    wav = wav+'_'+str(i)
    plt.savefig('images/{name}.png'.format(name=wav))
    img = Image.open('images/{name}.png'.format(name=wav)).convert('L')
    img.save('images/{name}.png'.format(name=wav))
    return fmax

def makeData(img,i):
    img = img+'_'+str(i)
    image_test = np.array(Image.open('images/{name}.png'.format(name=img)).resize((20,20)))
    img_data=np.array(image_test)
    img_data=np.concatenate(img_data)
    return img_data

def makeTrainData(file,labels,df):
    counts = 0
    for i in tqdm(range(len(file))):
        for j in tqdm(range(3)):
            name = file[i][:len(file[i])-4]
            label = name[:4]
            makePics(name,j)
            data = makeData(name,j)
            df2 = pd.DataFrame({'y':[labels[label]]})
            for k in range(len(data)):
                feat = 'x'+str(k)
                df2['{name}'.format(name=feat)] = data[k]
            df = pd.concat([df,df2])
        counts+=1
    df.to_csv('data/imgData.csv',index=False)
    checkCSV(labels,df)
    return df

def makeTestData(file,labels,df,nums=1):
    counts = 0
    for i in tqdm(range(len(file))):
        for j in tqdm(range(nums)):
            name = file[i][:len(file[i])-4]
            label = name[:4]
            makePics(name,j)
            data = makeData(name,j)
            df2 = pd.DataFrame({'y':[labels[label]]})
            for k in range(len(data)):
                feat = 'x'+str(k)
                df2['{name}'.format(name=feat)] = data[k]
            df = pd.concat([df,df2])
        counts+=1
    df.to_csv('data/valData.csv',index=False)
    checkCSV(labels,df)
    return df

if __name__ == '__main__':
    files = os.listdir(os.getcwd()+'\soundFiles')
    print(len(files[:12]),len(files[12:24]),len(files[24:36]),len(files[36:48]))
    df = pd.read_csv('data/imgData.csv')
    df_blank = pd.DataFrame()
    #print(files)
    #checkCSV(labels,df)
    makeTrainData(['F01_part1.wav', 'F01_part2.wav', 'F01_part3.wav'],labels,df)
    #makeTestData(files,labels,df_blank,1)