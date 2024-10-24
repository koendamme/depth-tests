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

import gui_secondary 



class Gui_opcard(QtWidgets.QDialog):
       
    def __init__(self, parent=None, deviceNr='USB0::0x0547::0x1003::SN_18.196::RAW'):
    #def __init__(self, parent=None, deviceNr='USB0::0x0547::0x1003::SN_18.196::RAW'):
        QtWidgets.QDialog.__init__(self, parent)


        #------------------------------------------------------------------------------

        self.on_off_d = False
        self.fname = "."

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


    def __del__(self):
        self.timer.stop()
        

    def on_off(self):
        
        if self.on_off_d:
            self.on_off_d = False
            self.buttons.color_on_off(self.on_off_d)
            self.timer.stop()
            
                                    
        else:
            self.on_off_d = True
            self.buttons.color_on_off(self.on_off_d)

            self.timer = QtCore.QTimer()
            #QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"), self.on_off_r) # old pyqt4
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
            self.sc.on_off(time=self.settings.time,data=self.settings.opcard.data, mode_pe_tt=self.settings.mode_pe_tt.selection.currentText(), point_to_show=self.settings.point_to_show, marker = self.settings.marker)
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
        header = "OPBOX 2.1\n"
        header += "http:\\www.optel.pl\n\n"
        
        header += "HEADER = 24 lines\n"
        header += "Number of points = %s\n" % len(self.sc.time)
        header += "date = %s\n\n" % datetime.now()

        header += "Plot information:\n"
        header += "y step = 0.781250 %\n"
        header += "x step = %s us\n\n" % (self.sc.time[1]-self.sc.time[0])

        header += "Setting parameters:\n"
        """header += "Depth = %s us\n" % self.settings.window.data.value()
        header += "Delay = %s us\n" % self.settings.delay.data.value()
        header += "Source = %s\n" % self.settings.mode_pe_tt.selection.currentText()
        header += "Sampling Frequency = %s\n" % self.settings.sampling_frequency.selection.currentText()
        header += "Analog Filters = %s\n" % self.settings.analog_filters.selection.currentText()
        header += "Pre-amplifier = %s\n" % self.settings.pre_amplifier.selection.currentText()
        header += "Gain Mode = Constant\n" % self.settings.gain_mode.selection.currentText()
        header += "Gain Value = %s dB\n" % self.settings.gain.data.value()
        header += "Avarage = %s\n" % self.settings.avarage.selection.currentText()
        header += "Measure mode = %s\n\n" % self.sc.type_measurement.currentText()"""

        self.fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file')[0]
        
        if self.fname != "":

            pyplot.clf()
            pyplot.plot(self.sc.time, self.sc.data)
            pyplot.grid(True)

            if self.sc.velocity_list.selection.currentText() == "mm":
                pyplot.xlabel("S, mm")
                header += "S[mm], U[V] - %s m/s" % self.sc.velocity.data.value()
            elif self.sc.velocity_list.selection.currentText() == "in":
                pyplot.xlabel("S, in")
                header += "S[in], U[V] - %s m/s" % self.sc.velocity.data.value()
            else:
                pyplot.xlabel("t, $\mu$s")
                header += "t[us], U[V]"
                
            pyplot.ylabel("U, V")
            pyplot.savefig(r"%s.png" % self.fname)

            savetxt(self.fname, list(zip(self.sc.time, self.sc.data)), header=header )

            

     
        



# main
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    icon = Gui_opcard()
    #icon.setWindowTitle("Opbox")
    #icon.setFixedSize(1024,763)
    icon.show()
    icon.showMaximized()
    #icon.showFullScreen()
    app.exec_()

    
    
