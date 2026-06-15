#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mnist_mlp_int8.h"

constexpr size_t kTensorArenaSize = 20 * 1024;
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
    error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Erreur : AllocateTensors() failed !");
    while (1);
  }

  Serial.println("Modele MNIST charge !");
}

void predict(uint8_t* image_pixels) {
  TfLiteTensor* input = interpreter->input(0);

  // Récupère les paramètres de quantisation du tenseur d'entrée
  float scale      = input->params.scale;
  int   zero_point = input->params.zero_point;

  // Pixel uint8 [0-255] → float [0,1] → int8 quantisé
  for (int i = 0; i < 784; i++) {
    float normalized = image_pixels[i] / 255.0f;
    input->data.int8[i] = (int8_t)(normalized / scale + zero_point);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Erreur : Invoke() failed !");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  float out_scale = output->params.scale;
  int   out_zp    = output->params.zero_point;

  // Dequantisation + argmax sur 10 classes
  int   predicted = 0;
  float max_prob  = -999.0f;
  Serial.println("Scores :");
  for (int i = 0; i < 10; i++) {
    float prob = (output->data.int8[i] - out_zp) * out_scale;
    Serial.print("  ["); Serial.print(i); Serial.print("] ");
    Serial.println(prob, 4);
    if (prob > max_prob) {
      max_prob  = prob;
      predicted = i;
    }
  }

  Serial.print("Chiffre predit : "); Serial.println(predicted);
  Serial.print("Confiance      : "); Serial.println(max_prob, 4);
}

void loop() {
  // Exemple : image factice 28x28 = 784 pixels (remplace par ta vraie image)
  uint8_t sample_image[784] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0, 60,120,180,220,240,240,220,180,120, 60,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,100,200,240,255,255,255,255,255,255,240,200,100,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,180,240,255,255,255,200,180,180,200,255,255,240,180,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0, 80,180,240,255,200, 80, 40, 40, 80,200,255,240,180, 80,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,160,240,255,200, 80,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,180,255,255,220,100,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 80,180,240,255,255,180, 60,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 60,160,240,255,255,220,140, 40,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 40,140,240,255,255,255,200,100, 20,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,100,220,255,255,240,180,255,255,200,100, 20,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 80,200,255,255,180, 40,180,255,255,220,120,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,160,255,255,200, 80,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,100,240,255,220,100,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 80,220,255,240,140,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 40,180,255,255,180,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0, 60,160,220,255,240,180, 80, 20,  0,  0,  0, 60,220,255,255,180,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,100,220,255,255,255,255,240,200,140, 80, 80,160,240,255,255,140,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0, 80,180,240,255,255,255,255,255,255,240,255,255,255,220,140, 40,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0, 60,140,200,240,255,255,255,255,255,255,240,180, 80,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 40, 80,120,160,200,200,160,120, 80, 40,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

  predict(sample_image);
  delay(5000);
}
