<h1 align="center">
  <br>
  <br>
    Number Station Identification
  <br>
</h1>

<h4 align="center">A program that predicts what number station you are listening to</h4>

<p align="center">
    <a href="#about">About</a> •
    <a href="#how-to-use">How to Use</a> •
    <a href="#contact">Contact</a> •
    <a href="#license">License</a>
</p>

## About
This project is used to identify number stations where the user inputs an audio clip (in the form of a .wav file). This is done using a CNN (Convolutional Neural Network). CNN's are made for image classification, so the audio file was converted into a spectrogram, which is best described as a spectrum of frequencies of a signal as a function of time. In an effort to make use of the program as easy as possible, an interactive GUI was created using PyQt5. Within this interface, users can effortlessly drag and drop audio files into the first tab, observe predicted number station outputs and playback options in the second, and access further insights about the identified station from <a href="https://priyom.org">priyom.org</a> via the third tab. 

## How To Use

### Run the GUI

```bash
# clone this repo
$ git clone https://github.com/coldmayo/NumberStationIdentification.git

# cd into repo
$ cd NumberStationIdentification

# install required packages
$ pip install -r requirements.txt

# change run.sh permissions
$ chmod +x run.sh

# run executable
$ ./run.sh
```

## Contact

If you have any suggestions or found a bug you could email me at <coldmayo@proton.me>. You can find my other projects <a href="https://coldmayo.github.io/templates/projects.html">here</a>.

## License

MIT