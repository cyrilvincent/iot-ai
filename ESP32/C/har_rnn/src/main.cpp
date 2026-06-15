#include <Arduino.h>
#include <LittleFS.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "har_conv1d_int8.h"

#define WINDOW_SIZE   100
#define N_FEATURES    3
#define N_CLASSES     6
#define TENSOR_ARENA  (70 * 1024)

const char* CLASS_NAMES[N_CLASSES] = {
  "bike", "sit", "stairsdown", "stairsup", "stand", "walk"
};

const float SCALER_MEAN[N_FEATURES] = {-1.80785739f, 0.05559214f,  9.09959427f};
const float SCALER_STD[N_FEATURES]  = { 3.88284957f, 1.5370627f,   2.335477f};

namespace {
  const tflite::Model*      model       = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor*             input       = nullptr;
  TfLiteTensor*             output      = nullptr;
  uint8_t tensor_arena[TENSOR_ARENA];
}

bool load_csv_window(float data[WINDOW_SIZE][N_FEATURES]) {
  File f = LittleFS.open("/accelerometer_rnn.csv", "r");
  if (!f) {
    Serial.println("ERREUR : impossible d'ouvrir le CSV");
    return false;
  }

  // Skip header si présent
  String first = f.readStringUntil('\n');
  if (isAlpha(first[0])) {
    Serial.println("Header ignoré : " + first);
  } else {
    f.seek(0);
  }

  int row = 0;
  while (f.available() && row < WINDOW_SIZE) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;

    int idx = 0, start = 0;
    for (int i = 0; i <= (int)line.length() && idx < N_FEATURES; i++) {
      if (i == (int)line.length() || line[i] == ',') {
        float raw = line.substring(start, i).toFloat();
        data[row][idx] = (raw - SCALER_MEAN[idx]) / SCALER_STD[idx];
        idx++;
        start = i + 1;
      }
    }
    if (idx == N_FEATURES) row++;
  }
  f.close();

  if (row < WINDOW_SIZE) {
    Serial.printf("ERREUR : %d lignes lues (besoin %d)\n", row, WINDOW_SIZE);
    return false;
  }
  Serial.printf("CSV OK : %d lignes\n", row);
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=== HAR Conv1D Inference ===");

  if (!LittleFS.begin()) {
    Serial.println("ERREUR : LittleFS mount failed !");
    while (1);
  }
  Serial.println("LittleFS OK");

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERREUR : schema incompatible !");
    while (1);
  }

  static tflite::AllOpsResolver      resolver;
  static tflite::MicroErrorReporter  error_reporter;
  static tflite::MicroInterpreter    static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA, &error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERREUR : AllocateTensors() failed !");
    while (1);
  }

  // ✅ Assigner AVANT d'utiliser
  input  = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("ERREUR : tenseurs null !");
    while (1);
  }

  Serial.printf("Input  type  : %d (1=f32, 9=int8)\n", input->type);
  Serial.printf("Output type  : %d (1=f32, 9=int8)\n", output->type);
  Serial.printf("Input  shape : [%d, %d, %d]\n",
    input->dims->data[0], input->dims->data[1], input->dims->data[2]);
  Serial.printf("Output shape : [%d, %d]\n",
    output->dims->data[0], output->dims->data[1]);

  Serial.printf("Input  scale=%.6f  zero_point=%d\n",
    input->params.scale, input->params.zero_point);
  Serial.printf("Output scale=%.6f  zero_point=%d\n",
    output->params.scale, output->params.zero_point);
}

void run_inference() {
  static float window[WINDOW_SIZE][N_FEATURES];
  if (!load_csv_window(window)) return;

  // Paramètres de quantisation input
  float in_scale      = input->params.scale;
  int   in_zero_point = input->params.zero_point;

  // Rempli tenseur input INT8
  for (int t = 0; t < WINDOW_SIZE; t++)
    for (int f = 0; f < N_FEATURES; f++) {
      int val = (int)(window[t][f] / in_scale) + in_zero_point;
      val = val < -128 ? -128 : (val > 127 ? 127 : val);  // clamp
      input->data.int8[t * N_FEATURES + f] = (int8_t)val;
    }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERREUR : Invoke() failed !");
    return;
  }

  // Paramètres de quantisation output
  float out_scale      = output->params.scale;
  int   out_zero_point = output->params.zero_point;

  int   best_class = 0;
  float best_score = -1.0f;

  Serial.println("\n── Résultats ──────────────────");
  for (int i = 0; i < N_CLASSES; i++) {
    float score = (output->data.int8[i] - out_zero_point) * out_scale;
    Serial.printf("  %-12s : %.1f%%\n", CLASS_NAMES[i], score * 100.0f);
    if (score > best_score) { best_score = score; best_class = i; }
  }
  Serial.printf("→ Classe prédite : %s (%.1f%%)\n",
    CLASS_NAMES[best_class], best_score * 100.0f);
  Serial.println("────────────────────────────────");
}

void loop() {
  run_inference();
  delay(5000);
}