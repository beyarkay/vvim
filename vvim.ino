/*
TODO:
- [ ] Get finger data logging working properly
    - [x] Make the glove easier to wear
    - [x] Make the glove easier to setup
    - [ ] Get the glove logging data to file.
- [ ] Get keylogging data working properly
- [ ] Collect data fro keylogging and typing
- [ ] Train a model on the data
*/
int flexPins[] = {A0, A1, A2, A3, A4};
int flexValues[] = {0, 0, 0, 0};
int offsets[] =    {0, 0, 0, 0};
int numFlexSensors = 4;
//int flexPins[] = {A0};
//int flexValues[] = {0};
//int offsets[] =    {0};
//int numFlexSensors = 1;

void setup() {
    // put your setup code here, to run once:
    Serial.begin(9600);
    for (int i = 0; i < numFlexSensors; i++) {
        pinMode(flexPins[i], INPUT);
    }
}

void loop() {
    // put your main code here, to run repeatedly:
    for (int i = 0; i < numFlexSensors - 1; i++) {
        flexValues[i] = analogRead(flexPins[i]);
        Serial.print(flexValues[i]);
        Serial.print(", ");
    }
    flexValues[numFlexSensors - 1] = analogRead(flexPins[numFlexSensors - 1]);
    Serial.println(flexValues[numFlexSensors - 1]);
    delay(10);
}
