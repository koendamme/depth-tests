#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 2022-04-25

from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import time
import threading
#import multiprocessing, time, signal
from numpy import savetxt
from datetime import datetime
from matplotlib import pyplot
import datetime
import pickle
import numpy as np
import os
import gui_secondary as gui_secondary
import tqdm
import copy


class Gui_opcard(QtWidgets.QDialog):
    def __init__(self, parent=None, deviceNr='USB0::0x0547::0x1003::SN_18.196::RAW', record_time=15):
    #def __init__(self, parent=None, deviceNr='USB0::0x0547::0x1003::SN_18.196::RAW'):
        QtWidgets.QDialog.__init__(self, parent)


        #------------------------------------------------------------------------------

        self.on_off_d = False
        self.fname = "."
        self.record_time = record_time
        
        #------------------------------------------------------------------------------ color
        self.background_color = '#336799'
        text_color='color: white'
        
        pal=QtGui.QPalette()
        role = QtGui.QPalette.Background
        pal.setColor(role, QtGui.QColor(self.background_color))
        self.setPalette(pal)
        
        #------------------------------------------------------------------------------ plot

        self.sc = gui_secondary.Plot(background_color=self.background_color, text_color='color: white')
        
        #------------------------------------------------------------------------------ settings

        self.settings = gui_secondary.Settings(text_color='color: white', deviceNr=deviceNr)

        #------------------------------------------------------------------------------ buttons

        self.buttons = gui_secondary.Buttons()
        self.buttons.quit.clicked.connect(self.close)
        self.buttons.save_scan_A.clicked.connect(self.save_scan_A)
        
        #------------------------------------------------------------------------------ connect

        #QtCore.QObject.connect(self.buttons.on_off, QtCore.SIGNAL("clicked()"), self.on_off) # PyQt4
        self.buttons.on_off.clicked.connect(self.on_off)

        #------------------------------------------------------------------------------ box and setLayout

        self.plotbox = QtWidgets.QVBoxLayout()
        self.plotbox.addWidget(self.sc)
        
        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addLayout(self.plotbox)
        self.hbox.addWidget(self.settings)

        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.buttons)
                
        self.setLayout(self.vbox)

        #--------------------------------------------------
        self.recording = []
        self.times = []
        self.prev = time.perf_counter()

    def __del__(self):
        self.timer.stop()

    def on_off(self):
        if self.on_off_d:
            self.on_off_d = False
            self.buttons.color_on_off(self.on_off_d)

            # fname = datetime.datetime.now().strftime("%Y-%m-%d %H,%M,%S").replace('.', ',')
            # os.path.join("C:", os.sep, "dev", "depth-testing", "techmed_test_2703", "Test1")
            # with open(os.path.join("data", f"{fname}.pickle"), 'wb') as f:
            #     pickle.dump([self.recording, self.times], f)
            #     print(f"saved to file with {len(self.recording)} recordings")
            self.timer.stop()

        else:
            self.on_off_d = True
            self.buttons.color_on_off(self.on_off_d)
            self.recording = []
            self.timer = QtCore.QTimer()
            # QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"), self.on_off_r) # old pyqt4
            self.timer.timeout.connect(self.on_off_r)
            self.timer.start(50)



    def on_off_r(self):
        #read one frame from opbox
        try:
            self.settings.opcard.trigger_and_one_read__offset(self.settings.offset_value, self.sc.type_measurement.currentText())
        except:
            print( "err opcard" )
        #set plot time scale values    
        try:
            self.settings.set_time()
        except:
            print( "err calc time" )
        #set cursors
        try:
            self.settings.cursors(mouse=self.sc.mouse, mouse_click=self.sc.mouse_click, marker_range=self.sc.sc.marker_range)
                
        except:
            print( "err cursors" )
            
        try:
            self.sc.block_measurement = True
            data = self.settings.opcard.data
            # self.recording.append(data)
            # self.times.append(time.time())
            self.sc.on_off(time=self.settings.time,data=data, mode_pe_tt=self.settings.mode_pe_tt.selection.currentText(), point_to_show=self.settings.point_to_show, marker = self.settings.marker)
            # t = time.perf_counter()
            # print(t - self.prev)
            # self.prev = copy.deepcopy(t)
        except:
            print( "err plot", sys.exc_info()[0] )
            
    def closeEvent(self, event):
        quit_msg = "Do You realy want to close the program?"
        reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        
        if reply == QtWidgets.QMessageBox.Yes:
            icon.on_off_d = False
            event.accept()
        else:
            event.ignore()
    
    def save_scan_A(self):
        current_datetime = datetime.datetime.now()
        self.fname = current_datetime.strftime("%Y-%m-%d %H,%M,%S")
        # self.fname = current_datetime.strftime("test2")
        # self.fname = time.time()
        # self.fname = self.fname.replace('.',',')
        # print('Recording File Name:', self.fname)
        print('Recording Frequency:', self.settings.prf*1000)

        self.reclength = self.record_time #in seconds
        self.framecount = int(self.reclength*1000*self.settings.prf)
        self.datalib = [[[], []] for _ in range(self.framecount)]
        self.SecPerPulse = 1/(self.settings.prf * 1000)
        print(self.SecPerPulse)
        self.times = [0] * self.framecount

        prev_time = time.time()
        for i in tqdm.tqdm(range(self.framecount)):
            self.times[i] = time.perf_counter()
            self.settings.opcard.trigger_and_one_read__offset__timestamp()

            self.datalib[i][0] = self.settings.opcard.data #pulse selection > time or data > value
            self.datalib[i][1] = self.sc.time
            print(self.datalib[i][1])

            target_time = self.times[0] + self.SecPerPulse * (i+1)
            if target_time > time.perf_counter():
                #target_time = self.times[i] + self.secPerPulse * (i+1)

                time.sleep(target_time - time.perf_counter())
                #time.sleep(self.SecPerPulse - (time.perf_counter()-self.times[i]))
            else:
                print('Epic yikers, looks like we have have a certified uwu lagg moment')
                break

            curr = time.time()
            print(curr-prev_time)
            prev_time = copy.deepcopy(curr)
            
        print('I am SO done with this...')
        print('time of measurement is {} sec'.format(self.times[-1] - self.times[0] + self.SecPerPulse))

        # path = os.path.join("D:", os.sep, "techmed_synchronisatie_1-5", "Test1", "US")
        # path = os.path.join("C:", os.sep, "dev", "depth-testing", "1-5", "Test3", "US")
        # path = os.path.join("C:", os.sep, "dev", "depth-testing", "experiment-13-5", "Test2", "us")
        path = os.path.join("mri_experiment", "test1")
        # path = os.path.join("data")
        # self.fname = "pretty_long_cable"
        with open(os.path.join(path, self.fname + '.pickle'), 'wb') as f:
            pickle.dump(self.datalib, f)
            print('saved file to ' + path)

        with open(os.path.join(path, self.fname + '_times.pickle'), 'wb') as f:
            pickle.dump(self.times, f)


# main
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    icon = Gui_opcard(record_time=420)
    icon.setWindowTitle("Opbox")
    #icon.setFixedSize(1024,763)
    icon.show()
    #icon.showMaximized()
    #icon.showFullScreen()
    app.exec_()
    ####################################################################################################################################################################################################################################