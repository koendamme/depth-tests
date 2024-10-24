from opbox_visa import Opbox_v21
import os
import pickle
import time
import copy


def main():
    # conect and config
    opbox = Opbox_v21(deviceNr='USB0::0x0547::0x1003::SN_18.196::RAW')
    # opbox = Opbox_v21(deviceNr='USB0::0x0547::0x1003::SN_21.06::RAW')

    opbox.GetInfo()
    opbox.Instr_Restet()
    opbox.PowerOnOff(1)
    opbox.Instr_RestetFIFO()
    opbox.Get_Power_Info()

    opbox.SetPulseVoltage_level_stop_run(10)

    opbox.SetAnalogFilters(2)
    opbox.SetConstGain(15)
    opbox.SetTimePulse_stop_run(2.8)

    prf = 50 #Hz
    opbox.ConfigInternalTimer(1000000//prf)
    # Index 3 is 33.3 sampling freq
    opbox.SetSamplingFreq(3)

    opbox.SetAttenHilo(1)

    t = 1. / float(33.3)
    window = 80
    opbox.SetBufferDepth(int(window / t))

    delay = 10
    t = 1. / (float(33.3))
    delay_Ts = int(delay / t)
    opbox.SetDelay(delay_Ts)

    fname = "session2.pickle"

    # i = 0
    # start_time = time.perf_counter()
    with open(os.path.join("C:", os.sep, "data", "MRI-28-5", fname), 'wb') as f:
        while True:
            opbox.trigger_and_one_read__offset__timestamp()
            data = opbox.data
            pickle.dump({"data": data, "ts": time.time()}, f)

            # diff = time.perf_counter() - start_time
            # freq = i/diff
            # print(freq)

            # i+=1


if __name__ == '__main__':
    main()