from serial import Serial
import time

port = "/dev/cu.usbmodem141101"
baudrate = 9600

with Serial(port=port, baudrate=baudrate, timeout=1) as ino_serial:
    while True:
        if ino_serial.isOpen():
            now = str(time.time_ns() // 1_000)
            values = ino_serial.readline().decode('utf-8').strip()
            vals = [f"{i}\t{v}" for i,v in enumerate(values.split(','))]
            with open("mvmt.log", "a") as mvmtfile:
                for val in vals:
                    mvmtfile.write(f"{now}\t{val}\n")
            ino_serial.flush()
