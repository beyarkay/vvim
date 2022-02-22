from serial import Serial
import time

# Set the options for the serial port
port = "/dev/cu.usbmodem141101"
baudrate = 57600 

# Clear the movement log file
with open(f"glove.log", "w") as mvmtfile:
    mvmtfile.write("")

# Open the serial port for reading
print(f"Reading from serial port with port={port} and baud={baudrate}")
with Serial(port=port, baudrate=baudrate, timeout=1) as ino_serial:
    last_write = time.time();
    while True:
        if ino_serial.isOpen():
            now = str(time.time_ns() // 1_000)
            values = ino_serial.readline().decode('utf-8').strip()
            vals = [f"{i}\t{v}" for i,v in enumerate(values.split(','), 1)]
            with open(f"glove.log", "a") as mvmtfile:
                for val in vals:
                    mvmtfile.write(f"{now}\t{val}\n")
            ino_serial.flush()
            if time.time() - last_write > 5:
                last_write = time.time()
                print(".", flush=True, end="")
print("Finished")
