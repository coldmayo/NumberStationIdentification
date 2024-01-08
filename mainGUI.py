import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTabWidget, QHBoxLayout
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from musicPlayerWidget import MusicPlayerWidget
from pyqtgraph import PlotWidget
from backendGUI import *

class error(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('icons/logo.png'))
        self.resize(300, 100)
        self.layout = QVBoxLayout()
        self.label = QLabel("Error: Needs to be in wav file format")

        self.layout.addWidget(self.label)
        self.setWindowTitle("Error")
        self.setLayout(self.layout)

class dragDrop(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop File Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa;
                font-size: 16pt;
                font-family: Arial
            }
        ''')

class mainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('icons/logo.png'))
        self.layout = QVBoxLayout(self)
        self.resize(400, 500)
        self.setWindowTitle('Number Station Predictions')
        self.setAcceptDrops(True)

        # set up tabs
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab1,"Drop Files")
        self.tabs.addTab(self.tab2,"Predictions")
        self.tabs.addTab(self.tab3,"More Info")

        # Create first tab
        self.tab1.layout = QVBoxLayout(self)

        self.photoViewer = dragDrop()
        self.tab1.layout.addWidget(self.photoViewer)
        self.tab1.setLayout(self.tab1.layout)

        # Create second tab

        self.tab2.layout = QVBoxLayout(self)

        self.musicPlayerWidget = MusicPlayerWidget()
        path = os.path.abspath("playedSounds/noFileSelected.wav")
        self.musicPlayerWidget.setMedia(path,'wav')
        self.tab2.layout.addWidget(self.musicPlayerWidget)

        self.classPred = QLabel('Predicted Label: None')
        self.classPred.setFont(QFont('Arial',16))
        self.tab2.layout.addWidget(self.classPred)

        self.filePlot = PlotWidget()
        self.sample = np.array([0])
        self.t = np.array([0])
        self.filePlot.plot(self.t, self.sample)
        self.tab2.layout.addWidget(self.filePlot)

        self.tab2.setLayout(self.tab2.layout)

        #Create third tab

        self.tab3.layoutTop = QHBoxLayout(self)
        self.tab3.layoutBot = QVBoxLayout(self)
        
        self.info = QLabel('More Info:')
        url = "<a href=\"{fullLink}\">Link</a>".format(fullLink='https://priyom.org/number-stations')
        self.link = QLabel(url)
        self.link.setOpenExternalLinks(True)
        self.info.setFont(QFont('Arial',16))
        self.link.setFont(QFont('Arial',16))
        self.tab3.layoutTop.addWidget(self.info)
        self.tab3.layoutTop.addWidget(self.link)
        twitUrl = "<a href=\"{fullLink}\">Twitter</a>".format(fullLink = 'https://twitter.com/priyom_org')
        discUrl = "<a href=\"{fullLink}\">Discord</a>".format(fullLink = 'https://discord.com/invite/788JPdSgsd')
        self.info = QLabel(
            'The link above sends you to website hosted by Priyom. They are a fantastic organization that provides tons of information on number stations. Make sure to support them by following them on {twit} or joining their {disc}.'.format(twit=twitUrl, disc=discUrl)
        )
        self.tab3.layoutBot.addLayout(self.tab3.layoutTop)
        self.tab3.layoutBot.addWidget(self.info)
        self.info.setWordWrap(True)
        self.tab3.setLayout(self.tab3.layoutBot)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for i in files:
            if i[-3:] != 'wav':
                self.disp = error()
                self.disp.show()

            else:
                self.classPred.setText('Predicted Label:' + getPrediction(i))
                self.classPred.setFont(QFont('Arial',16))
                
                self.layout.removeWidget(self.filePlot)
                self.filePlot = PlotWidget()
                Fs, samples = wavfile.read(i)
                noise = np.random.normal(0,0.5,len(samples))
                if samples.shape != noise.shape:
                    samples = samples[:,0]
                self.sample = samples
                fileLen = len(samples)/Fs
                self.t = np.linspace(0, fileLen, len(samples))
                self.filePlot.setXRange(0, fileLen, padding=0)
                self.filePlot.setYRange(-max(samples)+20, max(samples)+20, padding=0)
                self.filePlot.plot(self.t, self.sample)
                self.tab2.layout.addWidget(self.filePlot)
                
                self.musicPlayerWidget.setMedia(i,'wav')

                self.link.setText(getURL(getPrediction(i).lower()))
                self.link.setFont(QFont('Arial',16))
                self.link.setOpenExternalLinks(True)

app = QApplication(sys.argv)
display = mainApp()
display.show()
app.exec()