#include <Arduino.h>

float f(float x) {
    return 41.635 * x - 311;
}

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(9600);
}

void loop() {
    float result = f(100);
    Serial.println(result);
    delay(1000);
}