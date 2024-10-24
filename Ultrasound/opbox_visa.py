#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 2022-07-28

import pyvisa
#import usb
from numpy import array, absolute, linspace
from opbox_visa_class import *
from scipy import signal
import time
import ctypes
import binascii

class Opbox_v21():
    """
        OPBOX ver 2.1

        u'www.OPTEL.pl'
        u'OPBOX-2.1 Ultrasonic Box'
        u'SN_13.34'

        use help() function, example:
        opcard = v22()
        help(opcard)
        help(opcard.SetTimePulse)
    """

    trigger = Trigger()
    measure = Measure()
    analogCtrl = AnalogCtrl()
    PulserTime = PulserTime()

    
    
    def __init__(self, deviceNr='USB0::0x0547::0x1003::SN_18.196::RAW'):

        self.header_size = 52
        self.depth = 1000
        self.data = []
        self.data_average = []
        self.average = 1
        
        #self.__rm = pyvisa.highlevel.ResourceManager() # old
        self.__rm = pyvisa.ResourceManager()

        try:
            #self.__rm.list_resources('?*').index('USB0::0x0547::0x1003::SN_13.34::RAW') # old
            #self.__rm.list_resources('?*').index('USB0::0x0547::0x1003::SN_12.25::RAW') # old
            #self.__rm.list_resources('?*').index('USB0::0x0547::0x1003::SN_12.25::RAW')
            #self.__rm.list_resources('?*').index('USB0::0x0547::0x1003::SN_19.203::RAW')
            self.__rm.list_resources('?*').index(deviceNr)
            
        except:
            raise ValueError('Device not found')
            return 1

        #self.__opbox = self.__rm.open_resource('USB0::0x0547::0x1003::SN_13.34::RAW') # old
        #self.__opbox = self.__rm.open_resource('USB0::0x0547::0x1003::SN_12.25::RAW') # old
        #self.__opbox =  self.__rm.open_resource('USB0::0x0547::0x1003::SN_12.25::RAW')
        #self.__opbox =  self.__rm.open_resource('USB0::0x0547::0x1003::SN_19.203::RAW')
        self.__opbox =  self.__rm.open_resource(deviceNr)

        self.session = self.__opbox.session        

    def __del__(self):
        print( "PowerOff" )
        self.PowerOnOff(0)
                

    def default_configuration(self):

        
        self.GetInfo()
        self.Instr_Restet()
        self.PowerOnOff(1)
        self.Instr_RestetFIFO()
        self.Get_Power_Info()
        
        self.SetPulseVoltage(50)
            
        self.SetPulseVoltage_level_stop_run(15)

        self.SetAnalogFilters(1)
        self.SetBufferDepth()
        self.SetAttenHilo(0)
        self.SetConstGain(20)
        self.SetDelay(0)
        #print( self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xE1, 0, 0x10, 2) )
        self.TrigEnable(1)
        #print( self.setMeasure() )
        #print( self.setAnalogCtrl() )

        #for i in range(80):
            #self.StartSW()
            #print( self.Check_Frame_Ready() )

        #self.ReadData()


    def ConfigInternalTimer(self, time_period=50000):
        """
        time_period - timer period (1/pulse repetition frequency (prf)), 100-65535 [us], 0x16 [15:0]
        """

        self.trigger.TriggerSource = 0b011 # set trigger from internal Timer, 0x10 [3:0]
        self.trigger.TriggerEnable = 0b1   # set trigger enable, 0x10 [4]
        self.trigger.TimeEnable = 0b1      # timer enable, 0x10 [10]

        self.SetTimerPeriod(period=time_period) # timer period (1/pulse repetition frequency (prf)), 100-65535 [us], 0x16 [15:0]

        self.trigger.setValue()
        self.setTrigger()


    def GetInfo(self):
        """
            Info1 - SN_YEAR
            Info2 - SN_NO
        """
        data = self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xD0, 0, 0, 2)
        return ord(chr(data[0][1])), ord(chr(data[0][0]))


    def Instr_Restet(self):
        return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xD1, 0, 0, chr(1))

    
    def PowerOnOff(self, on_off):
        
        if on_off:
            return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x02, chr(1))
        else:
            return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x02, chr(0))

    def Get_Power_Info(self):
        #data = ord( self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xE1, 4, 0x02, 1)[0])
        data = int(str(bin(self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xE1, 4, 0x02, 1)[0][0]))[-1])
        #data = self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xE1, 4, 0x02, 1)
                        
        if data > 0:
            return 1
        else:
            return data
        
    def SetConstGain(self, gain):
        """
            gain - gain constant -31...+65 dB
        """

        gain = int(2*(gain+35))
                
        if gain>200:
            gain = 200

        if gain<8:
            gain = 8

        gain = (gain).to_bytes(1, byteorder='big')

        return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x28, gain)

    def SetTGCGain(self, data):
        #print data
        tout = 54 * [data[0]] + list(data)
        tout = array(262088/len(tout) * tout , dtype=ctypes.c_char_p)

        #s = ''
        #for i in tout:
        #    s += i
            
        #self.__opbox.write(s)
        self.__opbox.write_ascii_values('', tout)

        return 0
        

    def Instr_RestetFIFO(self):
        return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xD2, 0, 0, chr(1))

    def StartSW(self):
        """Trigger from aplication"""
        return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xD3, 0, 0)

    def Check_Frame_Ready(self):
        """Check that packet is ready to read"""
        return self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xD5, 0, 0, 1)[0][0]

    def Check_Vreg_Status(self):
        """Vreg Status that packet is ok"""
        d = self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xE1, 0, 0x02, 1)
        return str(bin(d[0][0]))[2]

    def ReadsetTrigger(self):
        """Reads the trigger setting """
        d = self.__opbox.visalib.usb_control_in(self.session, 0xC0, 0xE1, 0, 0x10, 1)
        return bin(d[0][0])[-3:]


    def setTrigger(self):
        return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x10, self.trigger.value)

    def setMeasure(self):
        return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x20, self.measure.value)

    def setAnalogCtrl(self):
        return self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x1A, self.analogCtrl.value)

    def TrigEnable(self, value=1):
        self.trigger.TriggerEnable = value
        self.trigger.setValue()
                
        return self.setTrigger()

    
    def ReadData_offset(self, offset=0, type_measurement="RF"):
        data_org = self.__opbox.read_raw(size=self.depth)

        self.header = data_org[:54]
        #self.data = 0.5*(array(map(ord, data_org[54:]), dtype=float) - 128)/128 # python2
        self.data = 0.5*(array( [ord(chr(i)) for i in data_org[54:]] , dtype=int) - 128)/128

        
        try:
            if type_measurement == "ABS":
                self.data = absolute(self.data)
            elif type_measurement == "POSITIVE":
                self.data = array([i if i>0 else 0 for i in self.data])
            elif type_measurement == "NEGATIVE":
                self.data = absolute([i if i<0 else 0 for i in self.data])
            elif type_measurement == "ENVELOPE":
                self.data = absolute(signal.hilbert(self.data))
                  
            self.data_argmax = self.data.argmax()
            self.data_max = self.data[self.data_argmax]
        except:
            print( "err data array" )

        
        if offset == 0:
            pass
            #self.data_argmax1 = self.data_argmax + self.depth/3 + self.data[self.data_argmax + self.depth/3:].argmax() # python2
            #self.data_argmax1 = self.data_argmax + self.depth/3 + self.data[int(self.data_argmax + self.depth):].argmax() # python3
        else:
            offset = int(offset * self.freq)
            self.data_argmax1 = self.data_argmax + offset + self.data[self.data_argmax + offset:].argmax()

        """if self.data_argmax1 < self.depth:
            self.data_max1 = self.data[self.data_argmax1]
        else:
            self.data_max1 = self.data[self.depth]"""
           

        return 0

    def ReadData_offset_timestamp(self, offset=0, type_measurement="RF"):
        data_org = self.__opbox.read_raw(size=self.depth)

        self.header = data_org[:54]
        #print(bin(self.header[3]), bin(self.header[4]))
        #print(int(self.header[3]), int(self.header[4]))
        self.frameIdx = (int(self.header[2])*256+int(self.header[1]))
        self.timeStamp = (int(self.header[4])*256+int(self.header[3]))
        self.overrun = (int(self.header[6])*256+int(self.header[5]))
        #self.data = 0.5*(array(map(ord, data_org[54:]), dtype=float) - 128)/128 # python2
        self.data = 0.5*(array( [ord(chr(i)) for i in data_org[54:]] , dtype=int) - 128)/128

        
        try:
            if type_measurement == "ABS":
                self.data = absolute(self.data)
            elif type_measurement == "POSITIVE":
                self.data = array([i if i>0 else 0 for i in self.data])
            elif type_measurement == "NEGATIVE":
                self.data = absolute([i if i<0 else 0 for i in self.data])
            elif type_measurement == "ENVELOPE":
                self.data = absolute(signal.hilbert(self.data))
                
            self.data_argmax = self.data.argmax()
            self.data_max = self.data[self.data_argmax]
        except:
            print( "err data array" )

        
        if offset == 0:
            pass
            #self.data_argmax1 = self.data_argmax + self.depth/3 + self.data[self.data_argmax + self.depth/3:].argmax() # python2
            #self.data_argmax1 = self.data_argmax + self.depth/3 + self.data[int(self.data_argmax + self.depth):].argmax() # python3
        else:
            s = int(offset * self.freq)
            self.data_argmax1 = self.data_argmax + offset + self.data[self.data_argmax + offset:].argmax()

        """if self.data_argmax1 < self.depth:
            self.data_max1 = self.data[self.data_argmax1]
        else:
            self.data_max1 = self.data[self.depth]"""
           

        return 0

    def ReadData(self):
        data_org = self.__opbox.read_raw(size=self.depth) #[i for i in self.__opbox.read_raw()]

        self.header = data_org[:54]
        #self.data = array(map(ord, data_org[54:]), dtype=int) - 128
        self.data = array( [ord(chr(i)) for i in data_org[54:]] , dtype=int) - 128

    def SetDelay(self, delayVal=0):
        """
        delayVal - umber of sampling time periods; t=1./self.sampling_frequency[MHz]; delayVal=int(delay[us]/t)
        """
        #print(delayVal)

        d = binascii.unhexlify( (delayVal).to_bytes(2, byteorder='little').hex() )
        #print(d)

        self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x22, d )
                
        return 0

    def SetBufferDepth(self, bufferDepth=1000):
        
        self.depth = bufferDepth
                
        d = binascii.unhexlify( (bufferDepth).to_bytes(4, byteorder='little').hex() )

        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x24, d):
            return 1
        return 0


    def SetAttenHilo(self, attenHilo=0):
        """
            This function set the input Attenuator (-20dB) or PostAmplifier (+24dB)
            attenHilo = 0 – 0dB – Attenuator and PostAmplifier are turned off
            attenHilo = 1 – +24dB – PostAmplifier turned on
            attenHilo = 2 – - 20dB – Attenuator turned on
        """

        if attenHilo < 0: attenHilo = 0
        if attenHilo > 2:  attenHilo = 1

        if attenHilo == 0:
            self.analogCtrl.InputAttenuator = 0b0
            self.analogCtrl.PostAmplifier = 0b0
            self.analogCtrl.setValue()
        elif attenHilo == 1:
            self.analogCtrl.InputAttenuator = 0b0
            self.analogCtrl.PostAmplifier = 0b1
            self.analogCtrl.setValue()
        elif attenHilo == 2:
            self.analogCtrl.InputAttenuator = 0b1
            self.analogCtrl.PostAmplifier = 0b0
            self.analogCtrl.setValue()
                
        return self.setAnalogCtrl()


    def SetAnalogFilters(self, analogFilters=1):
        """
            This function sets the analog filters like specified below:
            analogFilters 	Settings
            0 	0.5 – 6 MHz,
            1 	1 – 6 MHz,
            2 	2 – 6 MHz,
            3 	4 – 6 MHz,
            4 	0.5 – 10 MHz,
            5 	1 – 10 MHz,
            6 	2 – 10 MHz,
            7 	4 – 10 MHz,
            8 	0.5 – 15 MHz,
            9 	1 – 15 MHz,
            10 	2 – 15 MHz,
            11 	4 – 15 MHz,
            12 	0.5 – 25 MHz,
            13 	1 – 25 MHz,
            14 	2 – 25 MHz,
            15 	4 – 25 MHz
        """

        self.analogCtrl.AnalogFilter = analogFilters
        self.analogCtrl.setValue()
                
        return self.setAnalogCtrl()

    def trigger_and_one_read(self):

        status = self.StartSW()
        if status:
            return 1
        
        if self.Check_Frame_Ready():
                self.ReadData()
        else:
            print("New data not ready")

    def trigger_and_one_read__offset(self, offset=0, type_measurement="RF"):
        """Trigger the data aquisition, read the data average times and set
        data to averaged value."""
        for j in range(self.average):
            status = self.StartSW()
            if status:
                return 1
        
            for i in range(10):
                if self.Check_Frame_Ready():
                        self.ReadData_offset(offset, type_measurement)
                        break
                else:
                    if i>1:
                        print("New data not ready, %s attempt" % (i+1))
                        if i == 9:
                            return 1
                time.sleep(0.01)
            if j == 0:
                self.data_average = self.data/self.average
            else:
                self.data_average = self.data_average + self.data/self.average

        self.data = self.data_average 
        self.data_average = []
        return 0

    def trigger_and_one_read__offset__timestamp(self, offset=0, type_measurement="RF"):

        status = self.StartSW()
        if status:
            return 1

        for i in range(50):
            if self.Check_Frame_Ready():
                self.ReadData_offset_timestamp(offset, type_measurement)
                return 0

            time.sleep(0.01)

        print("New data not ready, %s attempt" % (i+1))

        return 1

    def ack_trigger_and_one_read(self):
        return self.trigger_and_one_read()
        

    def ack_trigger_and_one_read__offset(self, offset=0, type_measurement="RF"):
        return self.trigger_and_one_read__offset(offset,type_measurement)
        


    def SetAnalogInput_TT(self):
        self.analogCtrl.AnalogInput = 0b1
        self.analogCtrl.setValue()
                
        self.setAnalogCtrl()
        
        return 0

    def SetAnalogInput_PE(self):
        self.analogCtrl.AnalogInput = 0b0
        self.analogCtrl.setValue()
                
        self.setAnalogCtrl()
    
        return 0

    def SetPulseVoltage_level_stop_run(self, pulseVoltage_level=7):
        if self.SetPulseVoltage(4*pulseVoltage_level) != 0: return 1
        return 0

    def SetPulseVoltage(self, pulseVoltage=60):

        """
            pulseVoltage - Pulse Amplitude value 1...63
            (0...360V)
        """

        if pulseVoltage < 0: pulseVoltage = 0
        if pulseVoltage > 63:  pulseVoltage = 63

        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xD6, int(pulseVoltage), 0):
            return 1
            
        time.sleep(0.2)
                
        return 0

    def SetTimePulse_stop_run(self, timeP=3.1):
        self.PulserTime.PulseTime = int(timeP*10)
        self.PulserTime.setValue()
        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x1C, self.PulserTime.value):
            return 1
        return 0
    
    def SetPulserSelect_PE1(self):
        self.PulserTime.PulserSelect = 0b0
        self.PulserTime.setValue()
        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x1C, self.PulserTime.value):
            return 1
        return 0
    def SetPulserSelect_PE2(self):
        self.PulserTime.PulserSelect = 0b1
        self.PulserTime.setValue()
        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x1C, self.PulserTime.value):
            return 1
        return 0
    def SetPulserDriver_Enable(self):
        self.PulserTime.DriverEnable = 0b0
        self.PulserTime.setValue()
        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x1C, self.PulserTime.value):
            return 1
        return 0
    def SetPulserDriver_Disable(self):
        self.PulserTime.DriverEnable = 0b1
        self.PulserTime.setValue()
        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x1C, self.PulserTime.value):
            return 1
        return 0

    def SetAnalogFilters(self, analogFilters=1):
        """
            This function sets the analog filters like specified below:
            analogFilters 	Settings
            0 	0.5 – 6 MHz,
            1 	1 – 6 MHz,
            2 	2 – 6 MHz,
            3 	4 – 6 MHz,
            4 	0.5 – 10 MHz,
            5 	1 – 10 MHz,
            6 	2 – 10 MHz,
            7 	4 – 10 MHz,
            8 	0.5 – 15 MHz,
            9 	1 – 15 MHz,
            10 	2 – 15 MHz,
            11 	4 – 15 MHz,
            12 	0.5 – 25 MHz,
            13 	1 – 25 MHz,
            14 	2 – 25 MHz,
            15 	4 – 25 MHz
        """

        self.analogCtrl.AnalogFilter = analogFilters
        self.analogCtrl.setValue()
                
        self.setAnalogCtrl()

        return 0

    def SetGainMode(self, mode):
        
        self.measure.GainMode = mode
        self.measure.setValue()
                
        self.setMeasure()

        return 0

    def SetSamplingFreq(self, samplingFreq=0):
        """
            Function sets the sampling frequency. 
            SamplingFreq	Frequency
            0	100 MHz
            1	100 MHz
            2	50 MHz
            3	33.3 MHz
            4	25 MHz
            5	20 MHz
            6	16.7 MHz
            7	14.3 MHz
            8	12.5 MHz
            9	11.1 MHz
            10	10 MHz
            11	9.1 MHz
            12	8.3 MHz
            13	7.7 MHz
            14	7.1 MHz
            15	6.7 MHz
        """

        if samplingFreq < 0: samplingFreq = 0
        if samplingFreq > 15:  samplingFreq = 15

        self.measure.SamplingFreq = samplingFreq
        self.measure.setValue()
                
        self.setMeasure()

        return 0

    def SetTimerPeriod(self, period=10000):

        """
        PRF (pulse repetition frequency) setting
        100..65535 [us] – period setting for internal Timer
        (max 10kHz)
        """

        if period < 100:
            period = 100  # 10000Hz
        elif period > 65535:
            period = 65535  # 15Hz

        period = binascii.unhexlify( (period).to_bytes(2, byteorder='little').hex() )

        if self.__opbox.visalib.usb_control_out(self.session, 0x40, 0xE0, 0, 0x16, period):
            return 1

    
    def plot(self, delay=0, window=100):

        from matplotlib import pyplot

        x = delay+linspace(0,window, len(self.data))
        
        pyplot.clf()
        pyplot.plot(x, self.data)
        pyplot.show()

    def plot_save(self, file_name,delay=0, window=100):

        from matplotlib import pyplot

        x = delay+linspace(0,window, len(self.data))
        
        pyplot.clf()
        pyplot.plot(x, self.data)
        pyplot.savefig(file_name)
        
        



if __name__ == "__main__":

    # conect and config
    opbox = Opbox_v21(deviceNr='USB0::0x0547::0x1003::SN_18.196::RAW')
    #opbox = Opbox_v21(deviceNr='USB0::0x0547::0x1003::SN_21.06::RAW')
    
    
    
    opbox.default_configuration()
    opbox.TrigEnable(1)

    #--------------------------------------- Sampling Freq
    # set Sampling Freq
    """
            Function sets the sampling frequency. 
            SamplingFreq	Frequency
            0	100 MHz
            1	100 MHz
            2	50 MHz
            3	33.3 MHz
            4	25 MHz
            5	20 MHz
            6	16.7 MHz
            7	14.3 MHz
            8	12.5 MHz
            9	11.1 MHz
            10	10 MHz
            11	9.1 MHz
            12	8.3 MHz
            13	7.7 MHz
            14	7.1 MHz
            15	6.7 MHz
        """
    if opbox.SetSamplingFreq(1) != 0:
        print("err SamplingFreq")
    #---------------------------------------

    #--------------------------------------- time pulse
    opbox.SetTimePulse_stop_run(3.1)
    #---------------------------------------

    #--------------------------------------- window
    # set window, freq = 100MHz
    t = 1./(100)
    window = 100 # us
    n = int(window/t)
    opbox.SetBufferDepth( n )
    #---------------------------------------

    #--------------------------------------- delay
    freq = 100 #MHz
    delay = 60
    t = 1./(freq)
    delay_Ts = int(delay/t)

    if opbox.SetDelay( delay_Ts ) != 0:
            print("err change delay")

    #---------------------------------------
    
    #--------------------------------------- plot
    opbox.ack_trigger_and_one_read__offset()
    opbox.plot(delay=delay, window=window)
    #del(opbox)
