from dataPreproc import *
from myCNN import *
import pickle
import statistics as stat
from pydub import AudioSegment
from pydub.utils import make_chunks
import random


def wavChunks(filePath):
    myaudio = AudioSegment.from_file(filePath , "wav") 
    chunk_length_ms = 5000 # millisecs
    chunks = make_chunks(myaudio, chunk_length_ms)

    for i, chunk in (enumerate(chunks)):
        chunk_name = "chunk{n}.wav".format(n=i)
        chunk.export('testChunk/'+chunk_name, format="wav")

def wavToData(filePath):
    Fs, samples = wavfile.read(filePath)
    noise = np.random.normal(0,1,len(samples))
    if samples.shape != noise.shape:
        samples = samples[:,0]
    if len(samples) <= Fs:
        Fs = int(len(samples)*0.95)
    interval = int(Fs)
    overlap = int(Fs * 0.95)
    f, t, Sxx = spectrogram(samples,fs=Fs,nperseg=interval,noverlap=overlap)
    pcolormesh(t, f, 10 * log10(Sxx),cmap='jet')

    plt.savefig('images/image.png')
    img = Image.open('images/image.png').convert('L')
    img.save('images/image.png')
    image_test = np.array(Image.open('images/image.png').resize((20,20)))
    img_data=np.array(image_test)
    img_data=np.concatenate(img_data)
    return img_data

def getPrediction(filePath):

    layers = []
    files = os.listdir(os.getcwd() + '/layers')
    for i in files:
        if i[-3:-1] == 'kl':
            with open('layers/'+i, 'rb') as file2:
                layers.append(pickle.load(file2))

    Fs, samples = wavfile.read(filePath)
    if len(samples)/Fs > 10:
        preds = []
        wavChunks(filePath)
        files = []
        num = int(len(os.listdir(os.getcwd()+'/testChunk'))/2)
        for i in range(num):
            n = random.randint(0, num)
            files.append(os.listdir(os.getcwd()+'/testChunk')[n])
        for i in files:
            chunkPath = 'testChunk/'+i
            img = wavToData(chunkPath)
            img = img.reshape(20,20)
            y_pred= CNN_testing(img, layers).tolist()
            label = y_pred.index(max(y_pred))
            preds.append(label)
        finalLabel = stat.mode(preds)
        print(preds)
    else:
        img = wavToData(filePath)
        img = img.reshape(20,20)
        y_pred= CNN_testing(img, layers).tolist()
        finalLabel = y_pred.index(max(y_pred))
    return value(labels,finalLabel)

def getURL(l):
    link = {'e':'https://priyom.org/number-stations/english/',
            's':'https://priyom.org/number-stations/slavic/',
            'f':'https://priyom.org/number-stations/digital/',
            'm':'https://priyom.org/number-stations/morse/',
            'v':'https://priyom.org/number-stations/other/',
            'h':'https://priyom.org/number-stations/digital/',
            'x':'https://priyom.org/number-stations/digital/',
            'p':'https://priyom.org/number-stations/digital/'}
    
    if l[3] == '_':
        l = l[0:3]

    url = "<a href=\"{fullLink}\">Link</a>".format(fullLink=link[l[0]]+l)

    if l == 'p03' or l == 'f03':
        url = "<a href=\"{fullLink}\">Link</a>".format(fullLink='https://priyom.org/number-stations/operators/polish-11/digital-modes')
    return url

if __name__ == '__main__':
    print(getPrediction('soundFiles/e07_part3.wav'))