# VVIM - Keyboardless Vim interactions

This is done via a hardware glove that the user wears. The glove detects the
finger's positions and translates them into key presses. It's currently a work
in progress.

## Current Features
- Glove has been constructed and 
- Glove can detect finger movements of the right fore finger and right middle
  finger (With space to expand to more fingers if these first two actually
  work)
- Glove records finger movements via an Arduino Uno and sends them to serial
  output.
- Keylogger is installed on the developer's machine, and logs key presses to a
  file along with Unix milliseconds since epoch

## In Progress
- Capture the glove output from serial and save it to file (along with the
  timestamp)

## To Do
- Use an Arduino Nano instead of an Uno, and host the entire thing on the
  user's hand
- Connect the glove to the computer via Bluetooth, instead of a wired
  connection
- Collect lots of key presses and finger movements
    - Use the collected data to train a prediction model


## How to Start Recording Data
1. Run the command to clear the logfile:
``` 
sudo keylogger clear
```

2. Start the keylogger:
``` 
sudo keylogger
```


3. Start recording glove movements:
``` 
# This hasn't been figured out yet, but the follwing
#  command will use `screen` to view the serial output:
screen /dev/cu.usbmodem141101 9600
``

4. Put the glove on, and start typing things out. They keystrokes and finger
   movements will be recorded separately

5. Remove the glove, 

6. Stop recording the finger movements

7. Stop the keylogger with `CTRL-C`:

8. And you're done. The data will be stored until you start recording again.


# License
This work is licensed under GNU GPLv3. See the attached LICENSE. See
https://choosealicense.com/licenses/gpl-3.0/# for a non-legalese explanation of
the license.

