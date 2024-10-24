#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 2022-08-03

class Trigger():
    """
    Address 0x10

    value0, value1 - auxiliary variable, calc in setValue()

    TriggerSource:
        b000 (0) trigger from software (via bit [6] - “Trigger Sw”)
        b001 (1) trigger from external X pin (DB15 pin 11)
        b010 (2) trigger from external Y pin (DB15 pin 4)
        b011 (3) trigger from internal TIMER
        b100 (4) trigger from Encoder 1 (ENC1)
        b101 (5) trigger from Encoder 2 (ENC2)


    TriggerEnable:
        0 – trigger disabled (from any source)
        1 – trigger enabled

    TriggerReset:
        0 – writing '0' has no effect
        1 – reset current acquisition and/or discard stored data


    TriggerSw:
        0 – writing '0' has no effect
        1 – start measurement (in “Trigger Sw” source mode)

    XY_Divider_Enable:
        0 – divider disabled
        1 – divider enabled (ready to counting)


    XY_Divider_Reset:
        0 – divider reset and disabled
        1 – divider enabled

    TimeEnable:
        0 – disabled
        1 – enabled (source „b011” TIMER)
    
    Trigger_Status:
        0 – waiting for trigger (idle)
        1 – triggered (measurement in progress)

        
    Trigger_Overrun_Status:
        0 – no lost trigger from last Acquisition
        1 – lost trigger(s) from last Acquisition(number of missing trigger in TRG_OVERRUN register)
    """
    value0 = 0b00000000
    value1 = 0b00000111
    value = '%s%s' % (chr(value0),chr(value1))
    
    TriggerSource = 0b0000
    TriggerEnable = 0b0
    TriggerReset = 0b0
    TriggerSw = 0b0
    
    XY_Divider_Enable = 0b1
    XY_Divider_Reset = 0b1
    TimeEnable = 0b1
    Trigger_Status = 0b0
    Trigger_Overrun_Status = 0b0

    def setValue(self):
        self.value0 = (self.TriggerSw<<6) + (self.TriggerReset<<5) + (self.TriggerEnable<<4) + self.TriggerSource
        self.value1 = (self.Trigger_Overrun_Status<<6) + (0<<5) + (self.Trigger_Status<<4) + (0<<3) + (self.TimeEnable<<2) + (self.XY_Divider_Reset<<1) + self.XY_Divider_Enable
        self.value = '%s%s' % (chr(self.value0),chr(self.value1))



class Measure():
    """
    Address 0x22

    value0, value1 - auxiliary variable, calc in setValue()

    SamplingFreq:
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

    GainMode:
            0b00 – constant gain (register CONST_GAIN)
            0b01 – TGC curve (gain from array / memory)

    DataProcessingMode:
            0 – Raw data
            1 – Absolute data

    StoreDisable:
            0 – normal mode (Header and sample data stored in memory)
            1 - store disabled (sample data are not stored in memory, only Header)
    """
    
    value0 = 0b00000000
    value1 = 0b00000000
    value = '%s%s' % (chr(value0),chr(value1))
    
    SamplingFreq = 0b0000
    GainMode = 0b00
    DataProcessingMode = 0b0
    StoreDisable = 0b0

    def setValue(self):
        self.value0 = (self.DataProcessingMode<<7) + (0<<6) + (self.GainMode<<4) + self.SamplingFreq
        self.value1 = self.StoreDisable<<1 + (0<<1)
        self.value = '%s%s' % (chr(self.value0),chr(self.value1))


class AnalogCtrl():
    """
    Address 0x1A

    value0, value1 - auxiliary variable, calc in setValue()

    AnalogFilter:
        b0000 (0) 0.5 - 6 MHz
        b0001 (1) 1 - 6 MHz
        b0010 (2) 2 - 6 MHz
        b0011 (3) 4 - 6 MHz
        b0100 (4) 0.5 - 10 MHz
        b0101 (5) 1 - 10 MHz
        b0110 (6) 2 - 10 MHz
        b0111 (7) 4 - 10 MHz
        b1000 (8) 0.5 - 15 MHz
        b1001 (9) 1 - 15 MHz
        b1010 (10) 2 - 15 MHz
        b1011 (11) 4 - 15 MHz
        b1100 (12) 0.5 - 25 MHz
        b1101 (13) 1 - 25 MHz
        b1110 (14) 2 - 25 MHz
        b1111 (15) 4 - 25 MHz

    InputAttenuator:
        0 – off (0dB)
        1 – on (-20dB)

    
    PostAmplifier:
        0 – off (0dB)
        1 – on (+24dB)

    
    AnalogInput:
        0 – PE1 (Pulser/Receiver channel 1 – white BNC input)
        1 – PE2 (Pulser/Receiver channel 2 – black BNC input)

    """
    value0 = 0b00000000
    value1 = 0b00000000
    value = '%s%s' % (chr(value0),chr(value1))
    
    AnalogFilter = 0b0000
    InputAttenuator = 0b0
    PostAmplifier = 0b0
    AnalogInput = 0b0

    def setValue(self):
        self.value0 = (0<<7) + (self.AnalogInput<<6) + (self.PostAmplifier<<5) + (self.InputAttenuator<<4) + self.AnalogFilter
        self.value1 = 0b00000000
        self.value = '%s%s' % (chr(self.value0),chr(self.value1))
        
class PulserTime():
    """
    Adress 0x1C

    value0, value1 - auxiliary variable, calc in setValue()
    
    PulseTime:
        0b000000 - 0.0 us
        0b000001 - 0.1 us
        ...
        0b111110 - 6.2 us
        0b111111 - 6.3 us

    PulserSelect:
        0 - PE1 (Pulser/Receiver channel 1 – white BNC input)
        1 – PE2 (Pulser/Receiver channel 2 – black BNC input)

    DriverEnable:
        0 - driver enabled
        1 - driver disabled  
    """
    value0 = 0b00000000
    value1 = 0b00000000
    value = '%s%s' % (chr(value0),chr(value1))
    
    PulseTime = 0b000000
    PulserSelect = 0b0
    DriverEnable = 0b0

    def setValue(self):
        self.value0 = (self.DriverEnable<<7) + (self.PulserSelect<<6) + self.PulseTime
        self.value1 = 0b00000000
        self.value = '%s%s' % (chr(self.value0),chr(self.value1))
    

if __name__ == "__main__":

    #------------------------------------------
    # assert: Trigger
    t = Trigger()
    t.setValue()
    assert t.value0 == 0b00000000
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.setValue()
    assert t.value0 == 0b00000011
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.setValue()
    assert t.value0 == 0b00010011
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.setValue()
    assert t.value0 == 0b00110011
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 0
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00000110

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 0
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00000101

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 1
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 1
    t.TimeEnable = 0
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00000011

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 1
    t.TimeEnable = 1
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 1
    t.TimeEnable = 1
    t.Trigger_Status = 1
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00010111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 1
    t.TimeEnable = 1
    t.Trigger_Status = 0
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00000111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 1
    t.TimeEnable = 1
    t.Trigger_Status = 1
    t.Trigger_Overrun_Status = 1
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b01010111

    t.TriggerSource = 3 # trigger from internal TIMER
    t.TriggerEnable = 1
    t.TriggerReset = 1
    t.TriggerSw = 1
    t.XY_Divider_Enable = 1
    t.XY_Divider_Reset = 1
    t.TimeEnable = 1
    t.Trigger_Status = 1
    t.Trigger_Overrun_Status = 0
    t.setValue()
    assert t.value0 == 0b01110011
    assert t.value1 == 0b00010111

    #------------------------------------------

    #------------------------------------------
    # assert: Measure
    m = Measure()
    m.setValue()
    assert m.value0 == 0b00000000
    assert m.value1 == 0b00000000

    m.SamplingFreq = 1
    m.setValue()
    assert m.value0 == 0b00000001
    assert m.value1 == 0b00000000

    m.SamplingFreq = 1
    m.GainMode = 1
    m.setValue()
    assert m.value0 == 0b00010001
    assert m.value1 == 0b00000000

    m.SamplingFreq = 1
    m.GainMode = 1
    m.StoreDisable = 1
    m.setValue()
    assert m.value0 == 0b00010001
    assert m.value1 == 0b00000010

    m.SamplingFreq = 2
    m.GainMode = 1
    m.StoreDisable = 1
    m.setValue()
    assert m.value0 == 0b00010010
    assert m.value1 == 0b00000010

    m.SamplingFreq = 2
    m.GainMode = 1
    m.StoreDisable = 1
    m.DataProcessingMode = 1
    m.setValue()
    assert m.value0 == 0b10010010
    assert m.value1 == 0b00000010

    m.SamplingFreq = 10
    m.GainMode = 1
    m.StoreDisable = 1
    m.DataProcessingMode = 1
    m.setValue()
    assert m.value0 == 0b10011010
    assert m.value1 == 0b00000010
    #------------------------------------------

    #------------------------------------------
    # assert: Trigger
    a = AnalogCtrl()
    a.setValue()
    assert a.value0 == 0b00000000
    assert a.value1 == 0b00000000

    a = AnalogCtrl()
    a.AnalogFilter = 9
    a.setValue()
    assert a.value0 == 0b00001001
    assert a.value1 == 0b00000000

    a = AnalogCtrl()
    a.AnalogFilter = 9
    a.InputAttenuator = 1
    a.setValue()
    assert a.value0 == 0b00011001
    assert a.value1 == 0b00000000

    a = AnalogCtrl()
    a.AnalogFilter = 9
    a.InputAttenuator = 1
    a.PostAmplifier = 1
    a.setValue()
    assert a.value0 == 0b00111001
    assert a.value1 == 0b00000000

    a = AnalogCtrl()
    a.AnalogFilter = 9
    a.InputAttenuator = 1
    a.PostAmplifier = 1
    a.AnalogInput = 1
    a.setValue()
    assert a.value0 == 0b01111001
