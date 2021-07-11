# Display the time for error-checking purposes
import time
# Used for visualisation
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import matplotlib.animation as animation
# Used to `tail -F` the log files
import subprocess
import select

NUM_SENSORS = 4
NUM_DATAPOINTS = 100
INCLUDED_SENSORS = ["1", "2", "4", "5", "7", "8", "10", "11"]
KEYS_FILE = 'keys.log'
GLOVE_FILE = 'glove.log'

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

f = subprocess.Popen(
	['tail','-F', GLOVE_FILE], 
	stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
pipe = select.poll()
pipe.register(f.stdout)

def animate(i):
    """
    Code based off that found from PythonProgramming.net
    https://pythonprogramming.net/python-matplotlib-live-updating-graphs/
    1625086538155741	1	178
    1625086538155741	2	 151
    1625086538155741	3	 259
    1625086538155741	4	 226
    1625086538167592	1	178
    1625086538167592	2	 151
    1625086538167592	3	 259
    1625086538167592	4	 226
    """
    #if pipe.poll(1):
    #    pullData = f.stdout.readline()
    #else:
    f = subprocess.Popen(
        ['tail', '-n', str(NUM_SENSORS * NUM_DATAPOINTS), GLOVE_FILE], 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    lines = [line.decode('utf-8').strip() for line in f.stdout.readlines()]
    series = {}
    for line in lines:
        items = line.split('\t')
        if len(items) == 3:
            x, k, y = items
            if k in INCLUDED_SENSORS:
                if k in series.keys():
                    series[k]["xs"].append(x)
                    series[k]["ys"].append(y)
                else:
                    series[k] = {
                        "xs": [x],
                        "ys": [y]
                    }
    ax1.clear()
    print(f"Plotting {len(series.keys())} keys each with {[len(series[key]['xs']) for key in series.keys()]} datapoints")
    for key in series.keys():
        ax1.plot(
            series[key]["xs"],
            series[key]["ys"],
            linewidth=1
        )
    f.kill()
ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()



