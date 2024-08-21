#include "nnom.h"
#include "weights.h"
// #include <SoftwareSerial.h>
#include <HardwareSerial.h>

#define STM32C0116_DK
#if defined(MULT_REDUCE)
  const int8_t INPUT_FILL_VALUE = 1;
  int8_t reduce(int8_t v1, int8_t v2) { return v1 * v2; }
#else
  const int8_t INPUT_FILL_VALUE = 0;
  int8_t reduce(int8_t v1, int8_t v2) { return v1 + v2; }
#endif
// Specific device config
const int PRIMARY_ID = {{PRIMARY_ID}};
const int SECONDARY_ID = {{SECONDARY_ID}};
const bool IS_CONCAT = {{IS_CONCAT}};
const int NUM_THRESHOLDS = {{NUM_THRESHOLDS}};
const int INPUT_THRESHOLDS[] = {{INPUT_THRESHOLDS}};

// Shared device config
const int ROOT_ID = {{ROOT_ID}};
const int TAIL_ID = {{TAIL_ID}};
const int ROW_SIZE = {{ROW_SIZE}};

const int MAX_WRITE_AVAILABLE = 63;

const int BUTTON_PIN = PA8;
#ifdef STM32C0116_DK
  const int COMM_RX_PIN = PA3;
  const int COMM_TX_PIN = PA2;
  HardwareSerial COMM(COMM_RX_PIN, COMM_TX_PIN);
#else
  const int COMM_RX_PIN = 10;
  const int COMM_TX_PIN = 11;
  SoftwareSerial COMM(COMM_RX_PIN, COMM_TX_PIN);
#endif

nnom_model_t *model;
bool inputReady;
bool outputReady;

int dataBeforeWrite = 0;
int thresholdIndex = 0;
int inputLength = 0;

// DEBUG
const int LOOP_PERIOD_MS = 500;
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

void resetInput() {
  inputReady = false;
  inputLength = 0;
  thresholdIndex = 0;
  for (int8_t i = 0; i < INPUT_LENGTH; i++) {
    nnom_input_data[i] = INPUT_FILL_VALUE;
  }
}

void setup() {
  #ifdef STM32C0116_DK
    const int _RX = PA10_R;
    const int _TX = PA9_R;
    Serial.setRx(_RX);
    Serial.setTx(_TX);
  #endif
  pinMode(BUTTON_PIN, INPUT);
  Serial.begin(115200);
  COMM.begin(115200);

	model = nnom_model_create();
  if (PRIMARY_ID != ROOT_ID) {
    resetInput();
  } else {
    randomizeInput(0, 128);
    inputReady = true;
  }
  outputReady = PRIMARY_ID == ROOT_ID && SECONDARY_ID == 0;
  loopTimer = millis();
}

// Concatenate: INPUT_LENGTH = INPUT_THRESHSOLDS[-1], reduce = add
// Add: INPUT_LENGTH = INPUT_THRESHOLDS[n], reduce = add
// Mult: INPUT_LENGTH = INPUT_THRESHOLDS[n], reduce = mult
int updateInput() {
  int currentDataAvailable = COMM.available();
  if (!currentDataAvailable) { return currentDataAvailable; }
  int data = COMM.read();
  do {
    nnom_input_data[inputLength % INPUT_LENGTH] = reduce(nnom_input_data[inputLength % INPUT_LENGTH], (int8_t) data);
    data = COMM.read();
    inputLength++;
  } while (data != -1);

  // Signal next device
  if (IS_CONCAT && thresholdIndex < NUM_THRESHOLDS - 1 && inputLength == INPUT_THRESHOLDS[thresholdIndex]) {
    thresholdIndex++;
    Serial.write(1);
  }
  return currentDataAvailable;
}

void loop() {
  if (millis() - loopTimer >= LOOP_PERIOD_MS) {
    if (PRIMARY_ID == ROOT_ID && !digitalRead(BUTTON_PIN)) {
      randomizeInput(0, 128);
      inputReady = true;
    }
    
    // Calculate next output and send
    if (inputReady && outputReady) {
      Serial.printf("[ROW: %d, IDX: %d, LOOP: %d] Input: ", PRIMARY_ID, SECONDARY_ID, loops);
      printData(nnom_input_data, INPUT_LENGTH);
      model_run(model);
      Serial.printf(
        "[ROW: %d, IDX: %d, LOOP: %d] %sOutput: ",
        PRIMARY_ID, SECONDARY_ID, loops, PRIMARY_ID == TAIL_ID ? "Final " : ""
      );
      printData(nnom_output_data, OUTPUT_LENGTH);
      sendData(nnom_output_data, OUTPUT_LENGTH);
      resetInput();
      outputReady = false;
    }

    int currentDataAvailable = updateInput();    
    if (PRIMARY_ID == ROOT_ID && currentDataAvailable) {
      Serial.println("ROOT got:");
      printData(nnom_input_data, INPUT_LENGTH);
      Serial.println("Generating next input...");
      randomizeInput(0, 128);
      inputReady = true;
    } else if (PRIMARY_ID != ROOT_ID && thresholdIndex == NUM_THRESHOLDS - 1 && inputLength == INPUT_THRESHOLDS[thresholdIndex]) {
      inputReady = true;
      Serial.printf("[ROW: %d, IDX: %d, LOOP: %d] Input ready.\n", PRIMARY_ID, SECONDARY_ID, loops);
    }

    // May function incorrectly if Serial port on CONCAT device is used for printing
    int serialAvailable = Serial.available();
    if (!IS_CONCAT && serialAvailable > 0) {
      while (Serial.read() != -1) {}; // Empty Serial of all data
      // Try to distinguish between irrelevant prints and signals to send data
      if (serialAvailable == 1) { dataBeforeWrite += 1; }
    }
    
    bool outputWasReady = outputReady;
    // TODO: allow filling buffer with more than 1 outputs
    // Should only update outputReady if it was not ready before? (once made available, stays available until next send)
    outputReady = COMM.availableForWrite() >= MAX_WRITE_AVAILABLE && dataBeforeWrite == SECONDARY_ID;
    if (dataBeforeWrite == ROW_SIZE - 1) { dataBeforeWrite = 0; }
    if (!outputWasReady && outputReady && PRIMARY_ID != TAIL_ID) {
      Serial.printf("[ROW: %d, IDX: %d, LOOP: %d] Output ready.\n", PRIMARY_ID, SECONDARY_ID, loops);
    }

    loopTimer = millis();
    loops++;
  }
}

