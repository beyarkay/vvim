/*
   Documentation for Mux.h: https://github.com/stechio/arduino-ad-mux-lib
 */

#include <Mux.h>
using namespace admux;

int numSensors = 13;
int data[13];
// 16-channel Mux declared with analog input signal on pin A0 and channel
// control on digital pins 8, 9, 10 and 11.
Mux mux(Pin(A0, INPUT, PinType::Analog), Pinset(8, 9, 10, 11));


void setup() {
    // Use a high baud rate so that the sensor readings are more accurate.
    Serial.begin(57600);
}

void loop() {
    // Read each sensor value from the multiplexor.
    for (int i = 0; i < numSensors; i++) {
        if (i % 3 == 0) {
            // All of 0,3,6,9,12 are used for force sensors, which aren't
            // currently implemented. Just print 0 to keep the indexes correct.
            Serial.print("0, ");
            continue;
        }
        mux.channel(i);
        data[i] = mux.read();
        Serial.print(data[i]);
        Serial.print(", ");
        delay(1);
    }
    Serial.println("");
}
