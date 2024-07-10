#include "nnom.h"
#include "weights.h"
#include <SoftwareSerial.h>
#include <HardwareSerial.h>

{{DEVICE_NAME_DEF}}
int PRIMARY_ID = {{PRIMARY_ID}};
int SECONDARY_ID = {{SECONDARY_ID}};
bool IS_CONCAT = {{IS_CONCAT}};

int ROOT_ID = {{ROOT_ID}};
int TAIL_ID = {{TAIL_ID}};
int ROW_SIZE = {{ROW_SIZE}};

int MAX_WRITE_AVAILABLE = 63;

int BUTTON_PIN = PA8;
#ifdef STM32C0116_DK
  int COMM_RX_PIN = PA3;
  int COMM_TX_PIN = PA2;
  HardwareSerial COMM(COMM_RX_PIN, COMM_TX_PIN);
#else
  int COMM_RX_PIN = 10;
  int COMM_TX_PIN = 11;
  SoftwareSerial COMM(COMM_RX_PIN, COMM_TX_PIN);
#endif

nnom_model_t *model;
bool inputReady;
bool outputReady;

int lastDataAvailable;
int dataBeforeWrite = 0;

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
  randomizeInput(0, 128);
  inputReady = PRIMARY_ID == ROOT_ID;
  outputReady = PRIMARY_ID == ROOT_ID && SECONDARY_ID == 0;
  loopTimer = millis();
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
      inputReady = false;
      outputReady = false;
    }
    int currentDataAvailable = COMM.available();
    bool newDataAvailable = currentDataAvailable > lastDataAvailable;
    lastDataAvailable = currentDataAvailable;

    if (PRIMARY_ID == ROOT_ID && currentDataAvailable) {
      Serial.println("ROOT got:");
      readData(nnom_input_data, INPUT_LENGTH);
      printData(nnom_input_data, INPUT_LENGTH);
      Serial.println("Generating next input...");
      randomizeInput(0, 128);
      inputReady = true;
    } else if (currentDataAvailable == INPUT_LENGTH) {
      inputReady = true;
    } else if (IS_CONCAT && newDataAvailable) {
      // Ping other input nodes to send data
      Serial.write(1);
    }

    if (inputReady && PRIMARY_ID != ROOT_ID) {
      Serial.printf("[ROW: %d, IDX: %d, LOOP: %d] Input ready.\n", PRIMARY_ID, SECONDARY_ID, loops);
      readData(nnom_input_data, INPUT_LENGTH);
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
