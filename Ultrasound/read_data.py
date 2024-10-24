import pickle
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def main():
    # with open("data/test2.pickle", "rb") as openfile:
    #     data = pickle.load(openfile)
    #
    # print()
    #
    data = []
    ts = []
    with (open("data/test.pickle", "rb")) as openfile:
        while True:
            try:
                row = pickle.load(openfile)
                data.append(row["data"])
                ts.append(row["ts"])
            except EOFError:
                print("Hi")
                break

    # start, finish = ts[0], ts[-1]
    # total_time = datetime.fromtimestamp(finish) - datetime.fromtimestamp(start)
    # n_frames = len(ts)
    # fps = n_frames/total_time.seconds

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    def animate(i):
        i = i % len(data)
        # d = data[i][0]
        d = data[i]

        ax1.clear()
        ax1.plot(d)
        ax1.set_ylim([-0.15, 0.15])
        return ax1

    ani = animation.FuncAnimation(fig, animate, interval=0.04)
    plt.show()

    print()



if __name__ == '__main__':
    main()