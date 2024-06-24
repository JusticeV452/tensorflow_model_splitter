#include "nnom.h"
#include "weights.h"
// #include <SoftwareSerial.h>
#include <HardwareSerial.h>

#define MY_STM32C0116_DK
int DEVICE_ID = 0;

int ROOT_ID = 0;
int MAX_WRITE_AVAILABLE = 63;

int BUTTON_PIN = PA8;
#ifdef MY_STM32C0116_DK
  int COMM_RX_PIN = PA3;
  int COMM_TX_PIN = PA2;
  HardwareSerial COMM(COMM_RX_PIN, COMM_TX_PIN);
#else
  int COMM_RX_PIN = 46;
  int COMM_TX_PIN = 45;
  SoftwareSerial COMM(COMM_RX_PIN, COMM_TX_PIN);
#endif

nnom_model_t *model;
bool inputReady;
bool outputReady;

// DEBUG
int LOOP_PERIOD_MS = 500;
int loops = 0;
long loopTimer;


void randomizeInput(int min, int max) {
  for (int i = 0; i < INPUT_LENGTH; i++) {
    nnom_input_data[i] = random(min, max);
  }
}

void sendData(int8_t* data, int data_length) {
  for (int i = 0; i < data_length; i++) {
    COMM.write(data[i]);
  }
}

void readData(int8_t* data, int data_length) {
  for (int i = 0; i < data_length; i++) {
    data[i] = (int8_t) COMM.read();
  }
}

void printData(int8_t* data, int data_length) {
  Serial.print("{");
  for (int i = 0; i < data_length; i++) {
    Serial.print(data[i]);
    if (i < data_length - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("}");
}

void setup() {
  #ifdef MY_STM32C0116_DK
    const int _RX = PA10_R;
    const int _TX = PA9_R;
    Serial.setRx(_RX);
    Serial.setTx(_TX);
  #else
    pinMode(COMM_RX_PIN, INPUT);
    pinMode(COMM_TX_PIN, OUTPUT);
  #endif
  pinMode(BUTTON_PIN, INPUT);
  Serial.begin(115200);
  COMM.begin(115200);

	model = nnom_model_create();
  randomizeInput(0, 100);
  inputReady = DEVICE_ID == ROOT_ID;
  outputReady = true;
  loopTimer = millis();
}

void loop() {
  if (millis() - loopTimer >= LOOP_PERIOD_MS) {
    if (!digitalRead(BUTTON_PIN)) {
      randomizeInput(0, 100);
      inputReady = true;
    }
    
    // Calculate next output and send
    if (inputReady && outputReady) {
      Serial.printf("[DEVICE: %d, LOOP: %d] Input: ", DEVICE_ID, loops);
      printData(nnom_input_data, INPUT_LENGTH);
      model_run(model);
      Serial.printf("[DEVICE: %d, LOOP: %d] Output: ", DEVICE_ID, loops);
      printData(nnom_output_data, OUTPUT_LENGTH);
      sendData(nnom_output_data, OUTPUT_LENGTH);
      inputReady = false;
      outputReady = false;
    }

    inputReady = COMM.available() == INPUT_LENGTH;
    // TODO: allow filling buffer with more than 1 outputs
    outputReady = COMM.availableForWrite() >= MAX_WRITE_AVAILABLE;
    if (inputReady) {
      Serial.printf("[DEVICE: %d, LOOP: %d]  Input ready.\n", DEVICE_ID, loops);
      readData(nnom_input_data, INPUT_LENGTH);
    }

    loopTimer = millis();
    loops++;
  }

  if (DEVICE_ID == ROOT_ID && Serial.available() > 0) {
    Serial.println(Serial.readString());
  }
}
