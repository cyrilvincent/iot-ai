#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "cancer_mlp.h"
#include "scaler_cancer.h"

constexpr size_t kTensorArenaSize = 10 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Erreur : version schema incompatible !");
    while (1);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter   // requis sur cette version
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Erreur : AllocateTensors() failed !");
    while (1);
  }

  Serial.println("Modele charge !");
}

void predict(float* raw_features, int n_features) {
  TfLiteTensor* input = interpreter->input(0);

  // Normalisation MinMaxScaler : x_scaled = (x - min) / (max - min)
  for (int i = 0; i < n_features; i++) {
    input->data.f[i] = (raw_features[i] - SCALER_MIN[i]) / (SCALER_MAX[i] - SCALER_MIN[i]);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Erreur : Invoke() failed !");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  float prob_benigne = output->data.f[0];
  float prob_maligne = output->data.f[1];
  int predicted_class = prob_maligne >= prob_benigne ? 1 : 0;

  Serial.print("P(Benigne) : "); Serial.println(prob_benigne, 4);
  Serial.print("P(Maligne) : "); Serial.println(prob_maligne, 4);
  Serial.print("Classe     : ");
  Serial.println(predicted_class == 1 ? "Maligne (1)" : "Benigne (0)");
}

void loop() {
  float sample[30] = {
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
    0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  };

  predict(sample, 30);
  delay(5000);
}