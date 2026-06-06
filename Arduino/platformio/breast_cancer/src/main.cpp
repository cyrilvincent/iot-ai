#include <Arduino.h>
#include "cancer_rf_model.h"   // généré par emlearn

// Scaler RobustScaler : x_scaled = (x - center) / scale
// Valeurs issues du print Python, à remplacer par les vraies
const float CENTER[30] PROGMEM = {
  13.37, 18.84, 86.26, 551.1, 0.09575, 0.08722, 0.06074, 0.03336,
  0.1780, 0.06254, 0.2635, 1.025, 1.769, 20.86, 0.005220, 0.01757,
  0.02066, 0.008012, 0.01520, 0.002857, 15.16, 25.91, 97.89, 686.5,
  0.1313, 0.2119, 0.2267, 0.09752, 0.2839, 0.08294
};

const float SCALE[30] PROGMEM = {
  3.560, 4.385, 24.19, 262.6, 0.01721, 0.03449, 0.05308, 0.02032,
  0.02768, 0.008668, 0.2170, 0.5600, 1.665, 28.98, 0.002364, 0.008452,
  0.01530, 0.003578, 0.006274, 0.001309, 4.300, 6.290, 29.72, 452.2,
  0.02240, 0.09946, 0.1566, 0.05676, 0.06444, 0.01800
};

// Exemple : un sample de test (à remplacer par tes vraies features)
const float SAMPLE_RAW[30] = {
  17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
  0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
  0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
  0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
};
// Label réel : 0 (malignant)

void setup() {
  Serial.begin(9600);
}

void loop() {
    // Scaling
  int16_t features[30];
  for (int i = 0; i < 30; i++) {
    float center = pgm_read_float(&CENTER[i]);
    float scale  = pgm_read_float(&SCALE[i]);
    float scaled = (SAMPLE_RAW[i] - center) / scale;
    // Clamp vers int16 [-32768, 32767]
    scaled = constrain(scaled * 1000.0f, -32768.0f, 32767.0f);
    features[i] = (int16_t)scaled;
  }

  // Inférence
  int result = breast_cancer_rf_predict(features, 30);

  Serial.print("Prediction: ");
  Serial.print(result == 0 ? "Malignant" : "Benign");
  Serial.print(" (class ");
  Serial.print(result);
  Serial.println(")");
  delay(2000);
}