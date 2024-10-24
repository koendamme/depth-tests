import numpy as np
import matplotlib.pyplot as plt
import os

class Analysis:
    def __init__(self):
        self.fig, (self.ax1) = plt.subplots(1, 1)
        self.figure_number = plt.gcf().number
        self.fig.show()
        plt.ion()
        self.lower_bound = 0
        self.upper_bound = -1
        self.deltaT = 0.150
        self.lowpass_cFreq = 0.0005
        self.peak_distance_threshold = 100
        self.gain_proportion = 0.0001
        self.deltaX = None
        self.line1, = self.ax1.plot([], [], 'r-')
        #self.line2, = self.ax2.plot([], [], 'r-')
        #self.line3, = self.ax3.plot([], [], 'r-')
        #self.line4, = self.ax4.plot([], [], 'r-')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_xlabel('Distance (cm)')
        
        
    def initialize(self, x):
        #self.line1.set_data(self.x[self.lower_bound:self.upper_bound], np.zeros(len(x[self.lower_bound:self.upper_bound])))
        #self.line1.set_data(np.linspace(0, len(x[0][self.lower_bound:self.upper_bound]), len(x[0][self.lower_bound:self.upper_bound])), np.zeros(len(x[0][self.lower_bound:self.upper_bound])))
        self.line1.set_data(x[self.lower_bound:self.upper_bound], np.zeros(len(x[self.lower_bound:self.upper_bound])))

        #self.line2.set_data(np.linspace(0, 100, 100) * self.deltaT, [0] * 100)
        #self.line3.set_data(np.linspace(0, 100, 100) * self.deltaT, [0] * 100)
        #self.line4.set_data(np.linspace(0, 100, 100) * self.deltaT, [0] * 100)
    

    
    
    def update_plot(self, graph1, graph2, graph3, graph4):
        
        self.line1.set_ydata(graph1)
        self.ax1.relim()
        self.ax1.autoscale_view()
                          
        #self.line2.set_ydata(graph2)
        #self.ax2.relim()
        #self.ax2.autoscale_view()

        #self.line3.set_ydata(graph3)
        #self.ax3.relim()
        #self.ax3.autoscale_view()

        #self.line4.set_ydata(graph4)
        #self.ax4.relim()
        #self.ax4.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

   

if __name__ == '__main__':
    import pickle
    
    ana = Analysis()
    with open("./result/2024-03-04 15,35,52.pickle", 'rb') as f:
    # with open("../../collect_data/us.pickle", 'rb') as f:
        rawdata = pickle.load(f)
    axis = rawdata[0][1]
    rawdata = [pulse[0] for pulse in rawdata]
     
    #with open(r'C:\Users\banaa\Nextcloud\Documents\Bcs\win_Python36\Subject 6\Collective normalization\normalized_data_new_axis', 'rb') as f:
    #    axis = pickle.load(f)   
    #axis = [x for x,_ in enumerate(rawdata[0])]
    

    ana.initialize(axis)
    print(len(rawdata))
    i = 0
    data2 = [3] * 100
    data3 = [1] * 100
    data4 = [1] * 100
    
    while plt.fignum_exists(ana.figure_number):
        i = (i + 1) % len(rawdata[:])
        data1 = np.abs(rawdata[i][ana.lower_bound:ana.upper_bound])
        print(data1[0], data1[1])
        # data2 = np.roll(data2, -1)
        # data2[-1] = ana.datapoints_to_cm(ana.find_peak(data1))
        # data3 = np.gradient(data2, ana.deltaT)
        # data4 = np.gradient(data3, ana.deltaT)
        ana.update_plot(data1, data2, data3, data4)
        plt.pause(0.04)  
        if not plt.fignum_exists(ana.figure_number):
            break



        ##########################