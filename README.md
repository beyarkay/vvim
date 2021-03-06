# Vvim - Keyboardless Vim interactions

This is done via a hardware glove that the user wears. The glove detects the
finger's positions and translates them into key presses. It's currently a work
in progress.

### The fully 3D printed glove
The glove is custom designed and completely 3D printed.
![](images/glove_V1.jpg)

And an image from the side, showing how the flex sensors feed through each of
the finger brackets.
![](images/glove_V1_side.jpg)

### Graphs of the sensor readings
As the user's hand moves, the 8 sensors(positioned over the knuckles of the
four fingers) pick up this movement and feed it back to the computer. These
graphs show those sensor data, with the sensor closest to the wrist's labelled
`right-<FINGER_NAME>-1` and the sensors closest to the fingertip labelled
`right-<FINGER_NAME>-2`. For example, `right-pinky-2` or `right-middle-1`.


#### All sensor readings over a 1 minute long arbitrarily chosen interval
![](images/1min_all_sensors.png)

#### All sensor readings over a 5 second long arbitrarily chosen interval
![](images/5sec_all_sensors.png)

#### All sensors 500ms before and after pressing the `RETURN` key
![](images/return_500ms.png)

#### All sensors 500ms before and after pressing the `DEL` key
![](images/del_500ms.png)

#### All sensors 500ms before and after pressing the `u` key
![](images/u_500ms.png)

#### All sensors 500ms before and after pressing the `i` key
![](images/i_500ms.png)

#### A small-multiples plot of average sensor data for common keys
This set of graphs might need some explaining, but is very informative.  There
are 8 graphs in total, one for each finger. The x-axis on each of those graphs
depicts the sensor readings (in arbitrary units) and each of those graphs has
10 box-and-whisker-plots oriented horizontally, corresponding to the 10 most
frequently pressed keys in the dataset (`[return]`, `l`, `i`, `m`, `n`,
`[del]`, `u`, `h`, `o`, `k`).
![](images/boxenplot.png)
From this you can see that each key has a unique set of average sensor readings
at the moment it is pressed.


## Current Features
- A fully 3D printed glove containing 8 sensors (2 per finger)

- Glove can detect finger movements of the right fore finger and right middle
  finger (With space to expand to more fingers if these first two actually
  work)
    - This corresponds to the following keys, shown with how often those keys
      show up in the current dataset: `h`: 628, `u`: 291, `y`: 171, `m`: 171,
      `b`: 155, `k`: 120, `j`:  21, 

- Glove records finger movements via an Arduino script `vvim.ino` on an Uno,
  and sends them to serial output.
- Serial output is read by the python script `glove_logger.py` and saved to the
  file `glove.log` along with the Unix milliseconds since epoch.
- A keylogger is installed on the developer's machine, and logs key presses to
  the file `keys.log` along with Unix milliseconds since epoch.
- Running `cleanup.sh` cleans up the data from the keylogger and the serial
  output into one file named `sorted.log`.
- A Gradient Boosted tree has been trained and saved to `model.pkl`. Currently
  it has a Training Accuracy of 0.986 and a Test Accuracy of 0.813. A LSTM will
  likely perform better.
- Each finger has 2 sensors, with space to add an additional sensor per finger
- The file `eda.py` saves plots to `plots/` such as:

## Graphs

Each colour is a differently positioned sensor. Each line is one stream of data
recorded by a sensor. The streams have each been zeroed so that every instance
of pressing a certain key is centred.
![](plots/u_500ms.png)
#### Keys on the home row
Some keys are easier to spot, and others less so as my fingers move a lot when
pressing a `y` compared to a `k` just because of where the keys are positioned
on the keyboard.
![](plots/k_500ms.png)

#### More or less data
The data has not been normalised, so there's far more data for when common keys
like `h` are pressed compared to when a `j` is pressed
![](plots/j_500ms.png)
![](plots/m_500ms.png)
![](plots/h_500ms.png)


## In Progress
- Add a category to allow the glove to predict that no key is being pressed.
- (Re)train the ML model

## To Do
- Figure out some way of doing a residual analysis on any model so you can see
  where it's going wrong and what feature engineering you need to do.
- plot the predictions as a polar graph so that you can better distinguish the
  shape of each key. So the different sensors are mapped to theta, the flex is
  mapped to r and maybe time is mapped to a segment within each sensor's little
  slice of theta? That or just keep theta the same over time and change the
  color of the point being plotted.
- replace the costly flex sensors with an in-house version.
    - Maybe have wires connected from base to fingertip, and we measure how
      much that wire is paid out as the finger flexs?
    - Maybe use IMU sensors instead?
- Need a visualiser to see exactly what is happening with every sensor around
  certain time points
- Write some sort of visualiser to live track sensor data, actual key presses,
  and predicted key presses. Visualiser should:
    - `tail` the `keys.log` keys.log file and the `glove_measurements.log`
      file, so that the serial USB communication isn't blocked
    - Draw graphs of the sensor data live, displaying the past `n` seconds of
      historical data
    - Use the keylogging data to annotate when the actual key presses are.
    - Some sort of visualisation of what the model (as saved under `model.pkl`)
      is predicting for the current sensor values.
- If flex sensors aren't enough to predict exactly when a key is pressed, add
  force sensors to the fingertips.
- Experiment to see if you _really_ need two sensors per finger, or if you can
  get away with just 1 for some fingers
- Use an Arduino Nano 3.3v BLE because:
    - Small enough to have one on each hand
    - Can connect via BlueTooth instead of via wires
    - They also contain an IMU, so hand acceleration can be measured which will
      improve accuracy for keys further from the home row.
- Current models don't have the option of categorizing an sequence of sensor
  readings as not pressing any key at all. This should be fixed so the model
  isn't constantly assuming at least one key is being pressed
    - This could be done easily with pressure sensors

## Keys and which finger tends to press them
Note that this list is likely very specific to the author, as different people
will type differently. I think I probably use my right ring finger much more
than I really should. Also I type a `y` with my index finger for words like
`type` or `you` (where I subsequently have to type another letter with me right
hand), but I type it with my middle finger for words like `yes`, `yank`, or
`keyboard`.

- Right Hand
    - Thumb: `space`
    - Index: `j`, `m`, `n`, `b`, `h`, `y`
    - Middle: `k`, `y`, `u`, `i`, `<`, `(`, `[` 
    - Ring: `l`, `:`, `BACKSPACE`, `o`, `p`, `>`, `)`, `]`, `0`, `_`, `-`, `+`, `=`, `,`, `.`
    - Pinky: `;`, `ENTER`, `/`, `?`
- Left Hand (Incomplete as I've not yet built a glove for the left hand)
    - Pinky: 
    - Ring:
    - Middle: 
    - Index: 
    - Thumb: 

Here's a picture of my keyboard for reference:
![](images/keyboard.jpg)

## How to Start Recording Data
Probably best to do this all in `tmux` since handling multiple terminal windows
is a pain otherwise. A keylogger (I use [Casey Scarborough's
keylogger](https://github.com/caseyscarborough/keylogger)) is also required.

0. Install requirements
``` 
pip3 install -r requirements.txt
```

1. Run the command to clear the logfile:
``` 
sudo keylogger clear
```

2. Start the keylogger:
``` 
sudo keylogger ./keys.log
```

3. Start recording glove movements:
``` 
python3 glove_logger.py
```

4. Put the glove on, and start typing things out. I usually do this by opening
   a text file (like Alice in Wonderland available on Gutenberg) in vim (`vim
   alice.txt`), and then splitting the window vertically (`:vsp`), and then
   opening a temporary file in which to type in (`:e tmp`). Finally, type
   (`:set cursorbind`) into both frames so that the source text scrolls as you
   type it.  They keystrokes and finger movements will be recorded separately

5. Remove the glove

6. Stop the keylogger with `CTRL-C`

7. Stop recording the finger movements with `CTRL-C`

8. Now the data is recorded, clean it up:
```
./cleanup.sh
```

9. And analyse the data with `eda.py`
```
python3 eda.py
```
The images will be stored to `plots/` for your viewing pleasure

## License
This work is licensed under GNU GPLv3. See the attached LICENSE. See
https://choosealicense.com/licenses/gpl-3.0/# for a non-legalese explanation of
the license.

