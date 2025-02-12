#include "macros.h"
#include "firmware.h"
#include "comm.h"
#include "model.h"

#ifdef MULT_REDUCE
  constexpr int8_t FILL_VALUE = 1;
  inline int8_t reduce(int8_t v1, int8_t v2) { return v1 * v2; }
#else
  constexpr int8_t FILL_VALUE = 0;
  inline int8_t reduce(int8_t v1, int8_t v2) { return v1 + v2; }
#endif

#ifdef SLOT_MALLOC
  Communicator COMM(DEVICE_ID, NUM_DEVICES, SEND_BUFFER_SIZE, RECEIVE_BUFFER_SIZE, COMMStream);
#else
  Communicator COMM(DEVICE_ID, NUM_DEVICES, COMMStream);
#endif

bool inputReady;
#ifdef DEBUG_PRINT
  bool outputWasReady;
#endif

uint8_t thresholdIndex = 0;
uint8_t inputLength = 0;

// DEBUG
constexpr int LOOP_PERIOD_MS = 0;
unsigned long loops = 0;
unsigned long loopTimer;
#ifdef TRACK_STATS
  int messageIdTrackers[] = {NULL_ID, NULL_ID};
  unsigned long inputIds[] = {0, 0};
  constexpr int REPORT_PERIOD_MS = 60000;
  unsigned long successfullSends = 0;
  unsigned long startTime;
  unsigned long lastSendTime;
  float avgLatency = 0;
  unsigned long maxLatency = 0;
  unsigned long reportTimer = 0;
  #ifndef STOP_ON_FAIL
  unsigned long fails = 0;
  float MTT = 0; // Mean time to failure
  unsigned long lastFailTime;
  #endif
  #ifdef EXTERNAL_REPORTING
  template <typename T>
  void simpleSend(int idx, T data, bool isFloat=false) {
    Serial.write(idx);
    Serial.write(isFloat);
    Serial.write((byte*)&data, 4);
  }
  #else
    unsigned long lastSuccessfullSends = 0;
    unsigned long lastDuration = 0;
  #endif
#endif

int8_t* getInput(int8_t idx) {
  int8_t* ret = nnom_input_data;
  for (size_t i = 0; i < idx; i++) {
    ret += INPUT_LENGTHS[i];
  }
  return ret;
}

int8_t* getOutput(int8_t idx) {
  int8_t* ret = nnom_output_data;
  for (size_t i = 0; i < idx; i++) {
    ret += OUTPUT_LENGTHS[i];
  }
  return ret;
}

void randomizeArr(int8_t* arr, int arr_len, int min, int max) {
  for (int i = 0; i < arr_len; i++) {
    arr[i] = random(min, max);
  }
}

void randomizeInput(int min, int max) {
  randomizeArr(nnom_input_data, ALL_INPUTS_SIZE, min, max);
}

#ifdef DEBUG_PRINT
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
#endif

void resetInput(int8_t idx=0) {
  inputReady = IS_INPUT_NODE;
  inputLength = 0;
  thresholdIndex = 0;
  memset(nnom_input_data, FILL_VALUE, ALL_INPUTS_SIZE * sizeof(int8_t));
}

void setup() {
  deviceSetup();

  #if (defined(DEBUG_PRINT) || defined(TRACK_STATS))
    Serial.begin(115200);
    while (!Serial);
  #endif

  #ifdef STATIC_SLOT_ASSIGN
    for (int d = 0; d < COMM.numDevices; d++) {
      COMM.slots[d].setId(CONNECT[d]);
    }
  #endif

  initializeModel();
  resetInput();
  if (IS_INPUT_NODE) {
    #ifndef USE_ACC
      randomizeInput(0, 128);
    #endif
  }
  inputReady = IS_INPUT_NODE;
  loopTimer = millis();
  #ifdef TRACK_STATS
    startTime = loopTimer;
    lastSendTime = startTime;
    #ifndef STOP_ON_FAIL
    lastFailTime = startTime;
    #endif
  #endif
}

// Concatenate: INPUT_LENGTH = INPUT_THRESHSOLDS[-1], reduce = add
// Add: INPUT_LENGTH = INPUT_THRESHOLDS[n], reduce = add
// Mult: INPUT_LENGTH = INPUT_THRESHOLDS[n], reduce = mult
void updateInput(int8_t* dataArr, uint8_t dataLen, int8_t idx=0) {
  for (int i = 0; i < dataLen; i++) {
    nnom_input_data[inputLength % INPUT_LENGTHS[idx]] = reduce(nnom_input_data[inputLength % INPUT_LENGTHS[idx]], dataArr[i]);
    inputLength++;
  }

  // Signal next device
  if (IS_MULTI_INPUT && thresholdIndex < NUM_THRESHOLDS - 1 && inputLength == INPUT_THRESHOLDS[thresholdIndex]) {
    thresholdIndex++;
  }
}

void loop() {
  #ifdef PER_LOOP_INPUT_UPDATE
    perLoopInputUpdate();
  #endif

  #if defined(UPDATE_ON_END) && !defined(INPUT_NODE)
  if (!COMM.update()) {
    loops++;
    return;
  }
  #else
  COMM.update();
  #endif
  if (millis() - loopTimer < LOOP_PERIOD_MS) {
    return;
  }

  // Check messaging buffers
  #ifndef INPUT_NODE
  for (uint8_t d = 0; d < COMM.numDevices; d++) {
    MessageSlot* slot = COMM.slots + d;
    uint8_t expectedData;
    if (IS_MULTI_INPUT) expectedData = INPUT_THRESHOLDS[thresholdIndex];
    if (thresholdIndex > 0) expectedData -= INPUT_THRESHOLDS[thresholdIndex - 1];
    #ifdef ENABLE_HOLD
    if (inputReady || slot->deviceId == NULL_ID || !slot->dataAvailable) continue;
    #else
    if (inputReady || slot->deviceId == NULL_ID || !slot->dataAvailable || (!IS_INPUT_NODE && slot->deviceId != RECEIVE_ORDER[thresholdIndex])) continue;
    #endif
    #ifdef DEBUG_PRINT
      #ifdef ENABLE_HOLD
      if (!slot->receiveHold) {
        Serial.printf("data with id (%u) frm %d: ", slot->lastReceivedId, slot->deviceId);
        printData(slot->receiveBuffer, slot->dataAvailable);
      }
      #else
      Serial.printf("data with id (%u) frm %d: ", slot->lastReceivedId, slot->deviceId);
	    printData(slot->receiveBuffer, slot->dataAvailable);
      #endif
    #endif
    #ifdef TRACK_STATS
      #ifdef ENABLE_HOLD
      if (slot->dataAvailable == expectedData && !slot->receiveHold) {
      #else
      if (slot->dataAvailable == expectedData) {
      #endif
        uint8_t newId = messageIdTrackers[d] >= 2 ? 0 : messageIdTrackers[d] + 1;
        if (messageIdTrackers[d] != NULL_ID && slot->lastReceivedId != newId) {
          #ifdef DEBUG_PRINT
          Serial.printf("CRITICAL FAILURE: received unexpected message id from device %u for input %d - expected (%u) but got (%u)\r\n", slot->deviceId, inputIds[d], newId, slot->lastReceivedId);
          #endif
          #ifdef STOP_ON_FAIL
            while (true);
          #else
            fails++;
            unsigned long currTime = millis();
            MTT += ((float)(currTime - lastFailTime) - MTT) / fails;
            lastFailTime = currTime;
          #endif
        }
        messageIdTrackers[d] = slot->lastReceivedId;
      }
    #endif
    if (slot->receiveType == SEND && IS_INPUT_NODE) {
      if (!IS_RECURRENT) {
        #ifndef USE_ACC
          #ifdef DEBUG_PRINT
            Serial.println("Generating next input...");
          #endif
          resetInput();
          randomizeInput(0, 128);
        #endif
      }
      COMM.ack(slot->deviceId);
    } else if (slot->dataAvailable == expectedData && slot->receiveType == SEND) {
      #ifdef ENABLE_HOLD
      if (slot->deviceId == RECEIVE_ORDER[thresholdIndex]) {
        updateInput(slot->receiveBuffer, slot->dataAvailable);
        if (!slot->receiveHold) {
          COMM.ack(slot->deviceId);
        } else {
          slot->dataAvailable = 0;
          COMM.hold(slot->deviceId, false);
        }
		    #ifdef TRACK_STATS
        inputIds[d]++;
		    #endif
      } else if (!slot->receiveHold) {
        uint8_t origDataAvailable = slot->dataAvailable;
        COMM.ack(slot->deviceId, SENDANDHOLD_ACK);
        slot->dataAvailable = origDataAvailable; // Return to slot later
      }
      #else
      updateInput(slot->receiveBuffer, slot->dataAvailable);
      COMM.ack(slot->deviceId);
      #endif
    } else {
      #ifdef DEBUG_PRINT
        Serial.printf("No ack sent for msg (%d)\n", slot->lastReceivedId);
      #endif
      slot->discardMessage();
    }
  }
  #endif

  if ((!IS_INPUT_NODE || IS_RECURRENT) && thresholdIndex == NUM_THRESHOLDS - 1 && inputLength == INPUT_THRESHOLDS[thresholdIndex]) {
    inputReady = true;
    #ifdef DEBUG_PRINT
      Serial.printf("[ID: %d, LOOP: %lu] Input ready.\n", DEVICE_ID, loops);
    #endif
  }

  bool outputReady = IS_OUTPUT_NODE || COMM.ready(OUT_DEVICE);
  // Calculate next output and send
  if (inputReady && outputReady) {
    #ifdef DEBUG_PRINT
      Serial.printf("[ID: %d, LOOP: %lu] Input: ", DEVICE_ID, loops);
      printData(nnom_input_data, INPUT_LENGTHS[0]);
    #endif
    forward(model);
    #ifdef DEBUG_PRINT
      Serial.printf(
        "[ID: %d, LOOP: %lu] %sOutput: ",
        DEVICE_ID, loops, IS_OUTPUT_NODE ? "Final " : ""
      );
      printData(send_data, SEND_LENGTH);
    #endif
    #ifdef DISPLAY_OUTPUT
      displayOutput(send_data);
    #endif
    #ifdef DEBUG_PRINT
      if (!IS_OUTPUT_NODE) Serial.printf("sent data: %d\n", COMM.send(OUT_DEVICE, SEND, send_data, SEND_LENGTH, true));
    #else
      if (!IS_OUTPUT_NODE) COMM.send(OUT_DEVICE, SEND, send_data, SEND_LENGTH);
    #endif
    #ifdef TRACK_STATS
      successfullSends++;
      unsigned long currTime = millis();
      unsigned long latency = currTime - lastSendTime;
      unsigned long duration = currTime - startTime;
      long sec = duration / 1000;
      long min = sec / 60;
      long hr = min / 60;
      avgLatency += ((float)latency - avgLatency) / successfullSends;
      maxLatency = max(maxLatency, latency);
      float combAPR = 0;
      unsigned long maxReceiveAttempts = 0;
      for (int i = 0; i < COMM.numDevices; i++) {
        combAPR += COMM.slots[i].avgAttemptsPerReceive;
        maxReceiveAttempts = max(maxReceiveAttempts, COMM.slots[i].maxReceiveAttempts);        
      }
      if (reportTimer == 0 || millis() - reportTimer >= REPORT_PERIOD_MS) {
        #ifdef EXTERNAL_REPORTING
          simpleSend(1, successfullSends);
          simpleSend(2, duration);
          simpleSend(3, loops);
          simpleSend(4, avgLatency, true);
          simpleSend(5, maxLatency);
          simpleSend(6, combAPR / COMM.numDevices, true);
          simpleSend(7, maxReceiveAttempts);
          #ifndef STOP_ON_FAIL
          simpleSend(8, fails);
          simpleSend(9, MTT, true);
          #endif
        #else
          Serial.printf("Successful sends: %lu\r\n", successfullSends);
          Serial.printf("Run duration: %02d:%02d:%02d.%03d\r\n", hr, min % 60, sec % 60, duration % 1000);
          Serial.printf("Past %lu ms average mps: ", duration - lastDuration);
          Serial.println(1000.0 * (successfullSends - lastSuccessfullSends) / (duration - lastDuration));
          Serial.print("Overall average mps: "); Serial.println(1000.0 * successfullSends / duration);
          Serial.printf("Average cpm: %lu\r\n", loops / successfullSends);
          Serial.print("Average latency: "); Serial.print(avgLatency); Serial.println(" ms");
          Serial.printf("Max latency: %lu ms\r\n", maxLatency);
          Serial.print("Average attempts/(device*receive): "); Serial.println(combAPR / COMM.numDevices);
          Serial.printf("Max receive attempts: %lu\r\n", maxReceiveAttempts);
          lastDuration = duration;
          lastSuccessfullSends = successfullSends;
          #ifndef STOP_ON_FAIL
          Serial.printf("Failures: %lu\r\n", fails);
          Serial.print("MTT: "); Serial.print(MTT); Serial.println(" ms");
          #endif
          #ifdef TRACK_INTERLEAVING
          Serial.print("Approx intervleave %: "); Serial.print(100.0 * COMM.interleaveApprox / COMM.processedSlots); Serial.println("%");
          Serial.print("Valid slot %: "); Serial.print(100.0 * COMM.processedSlots / COMM.allSlots); Serial.println("%");
          #endif
        #endif
        reportTimer = millis();
      }
      lastSendTime = currTime;
    #endif
    #if defined(INPUT_NODE) && defined(POST_SEND_INPUT_UPDATE)
      postSendInputUpdate();
    #else
      resetInput();
    #endif
  }
  #ifdef DEBUG_PRINT
    if (!outputWasReady && outputReady && !IS_OUTPUT_NODE) {
      Serial.printf("[ID: %d, LOOP: %lu] Output ready.\n", DEVICE_ID, loops);
    }
    outputWasReady = outputReady;
  #endif
  loopTimer = millis();
  loops++;
}
