#include <avr/pgmspace.h>
#include "cancer_rf_model.h"

const float CENTER[30] PROGMEM = {14.1176, 19.1850, 91.8822, 654.3776, 0.0957, 0.1036, 0.0889, 0.0483, 0.1811, 0.0628, 0.4020, 1.2027, 2.8583, 40.0713, 0.0070, 0.0256, 0.0328, 0.0119, 0.0206, 0.0038, 16.2351, 25.5357, 107.1031, 876.9870, 0.1315, 0.2527, 0.2746, 0.1142, 0.2905, 0.0839 };
const float SCALE[30] PROGMEM = { 3.5319, 4.2613, 24.2953, 354.5529, 0.0139, 0.0524, 0.0794, 0.0380, 0.0275, 0.0072, 0.2828, 0.5412, 2.0689, 47.1844, 0.0031, 0.0186, 0.0321, 0.0063, 0.0082, 0.0028, 4.8060, 6.0584, 33.3380, 567.0487, 0.0231, 0.1548, 0.2092, 0.0653, 0.0631, 0.0178 };
const float SAMPLE_RAW[30] = { 3.5319, 4.2613, 24.2953, 354.5529, 0.0139, 0.0524, 0.0794, 0.0380, 0.0275, 0.0072, 0.2828, 0.5412, 2.0689, 47.1844, 0.0031, 0.0186, 0.0321, 0.0063, 0.0082, 0.0028, 4.8060, 6.0584, 33.3380, 567.0487, 0.0231, 0.1548, 0.2092, 0.0653, 0.0631, 0.0178 };

void setup() {
  Serial.begin(9600);
  while (!Serial);

  int16_t features[30];
  for (int i = 0; i < 30; i++) {
    float center = pgm_read_float(&CENTER[i]);
    float scale  = pgm_read_float(&SCALE[i]);
    float scaled = (SAMPLE_RAW[i] - center) / scale;
    scaled = constrain(scaled * 1000.0f, -32768.0f, 32767.0f);
    features[i] = (int16_t)scaled;
  }

  int result = breast_cancer_rf_predict(features, 30);
  Serial.println(result == 0 ? "Malignant" : "Benign");
}

void loop() {}