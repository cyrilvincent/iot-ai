#include <Arduino.h>
#include "heart_rf_model.h"   // généré par emlearn

// Scaler RobustScaler : x_scaled = (x - center) / scale
// Valeurs issues du print Python, à remplacer par les vraies
const float CENTER[30] PROGMEM = {
  47.7500, 0.7115, 2.9519, 132.0481, 249.6010, 0.0673, 0.2548, 140.3510, 0.2885, 0.5736
};

const float SCALE[30] PROGMEM = {
  7.8382, 0.4530, 0.9843, 18.0637, 63.9118, 0.2506, 0.4976, 23.4856, 0.4530, 0.9455
};

// Exemple : un sample de test
const float SAMPLE_RAW[10] = {
  28,1,2,130,132,0,2,185,0,0
};

void setup() {
  Serial.begin(9600);
}

void loop() {
    // Scaling
  int16_t features[10];
  for (int i = 0; i < 10; i++) {
    float center = pgm_read_float(&CENTER[i]);
    float scale  = pgm_read_float(&SCALE[i]);
    float scaled = (SAMPLE_RAW[i] - center) / scale;
    // Clamp vers int16 [-32768, 32767]
    scaled = constrain(scaled * 10000.0f, -32768.0f, 32767.0f);
    features[i] = (int16_t)scaled;
  }

  // Inférence
  int result = rf_predict(features, 10);

  Serial.print("Prediction: ");
  Serial.println(result);
  delay(2000);
}