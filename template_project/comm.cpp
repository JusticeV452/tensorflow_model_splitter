#include "comm.h"

#ifndef HARDWARE_CRC
uint8_t crc8(const uint8_t* data, size_t length) {
  uint8_t crc = 0;

  for (size_t i = 0; i < length; i++) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; j++) {
      // Check the MSB of the 8-bit CRC and XOR if set
      if (crc & 0b10000000) {
        crc = (crc << 1) ^ CRC_POLY;
      } else {
        crc <<= 1;
      }
    }
  }

  return crc;
}
#endif

#ifdef SLOT_MALLOC
  MessageSlot::MessageSlot(uint8_t sendBufferSize, uint8_t receiveBufferSize) {
    deviceId = NULL_ID;
    receiveSize = receiveBufferSize;

    sendBuffer = (int8_t*)malloc(sendBufferSize * sizeof(int8_t));
    if (!sendBuffer) {
      Serial.print(FAILED_BUFFER_ALLOC);
      while (true);
    }

    receiveBuffer = (int8_t*)malloc(receiveBufferSize * sizeof(int8_t));
    if (!receiveBuffer) {
      Serial.print(FAILED_BUFFER_ALLOC);
      while (true);
    }
  }
  
  MessageSlot::~MessageSlot() {
    if (sendBuffer) free(sendBuffer);
    if (receiveBuffer) free(receiveBuffer);
  }
#endif

void MessageSlot::setId(uint8_t id) {
  deviceId = id;
  dataAvailable = 0;
  lastReceivedId = NULL_ID;
  waitingForAck = -1;
  messagePos = 0;
  tempIdx = 0;
}

void MessageSlot::updateCounter() {
  sendCounter = (sendCounter + 1) % 3;
}

void MessageSlot::discardMessage() {
  dataAvailable = 0;
}

#ifdef SLOT_MALLOC
Communicator::Communicator(uint8_t id, uint8_t nDevices, uint8_t sendBufferSize, uint8_t receiveBufferSize, Stream& stream) : strm(stream) {
#else
Communicator::Communicator(uint8_t id, uint8_t nDevices, Stream& stream) : strm(stream) {
#endif
  numDevices = nDevices;
  #ifndef STATIC_SLOT_ASSIGN
    nextMessageSlot = 0;
  #endif

  slots = (MessageSlot*)malloc(numDevices * sizeof(MessageSlot));
  if (!slots) {
    Serial.print(FAILED_GENERAL_MEM_ALLOC);
    while (true);
  }
  
  for (uint8_t i = 0; i < numDevices; i++) {
    #ifdef SLOT_MALLOC
      new (&slots[i]) MessageSlot(sendBufferSize, receiveBufferSize);
    #else
      new (&slots[i]) MessageSlot();
    #endif
    // Could be later adjusted to support sending alias (different id per slot)
    (slots + i)->selfId = id;
  }
}

Communicator::~Communicator() {
  if (!slots) return;
  #ifdef SLOT_MALLOC
    for (uint8_t i = 0; i < numDevices; i++) {
      slots[i].~MessageSlot(); // Explicitly call the destructor for each slot
    }
  #endif
  free(slots);
}

bool MessageSlot::ready() {
  #ifdef ENABLE_HOLD
    return waitingForAck == -1 && !sendHold;
  #else 
    return waitingForAck == -1;
  #endif
}

bool Communicator::ready(uint8_t deviceId) {
  #ifdef TEMP_MEM_SLOT
  slot = getSlot(deviceId);
  #else
  MessageSlot* slot = getSlot(deviceId);
  #endif
  #ifdef STATIC_SLOT_ASSIGN
    return slot && slot->ready();
  #else
    return slot->ready();
  #endif
}

void Communicator::write(uint8_t byte) {
  #if defined(TEMP_MEM_SLOT) && defined(DEBUG_PRINT) && defined(TRACK_SEND_STATS)
    if (slot && slot->sendAttempts >= 10) {
      Serial.printf("sending packet: %d\n", byte);
    }
  #endif
  #ifdef BUFFERED_WRITE
    message[messageIdx] = byte;
    messageIdx++;
  #else
    strm.write(byte);
  #endif
  #if defined(DEBUG_PRINT) && defined(PRINT_SEND_BYTES)
    Serial.printf("%u,", byte);
  #endif
  #ifdef RANDOM_WRITE_DELAY
    delay(random(WR_DELAY_MIN, WR_DELAY_MAX));
  #endif
}

void Communicator::nibbleSplitWrite(uint8_t orByte, uint8_t sendByte) {
  write(orByte | ((sendByte >> 4) + DATA_VALUE_OFFSET));
  write(orByte | ((sendByte & 0x0F) + DATA_VALUE_OFFSET));
}

void Communicator::crc8Write(uint8_t orByte, uint8_t sendByte) {
  write(orByte | sendByte);
  nibbleSplitWrite(orByte, crc8(&sendByte, 1));
}

#ifdef DEBUG_PRINT
MessageSlot* Communicator::getSlot(uint8_t deviceId, bool receiving) {
#else
MessageSlot* Communicator::getSlot(uint8_t deviceId) {
#endif
  for (uint8_t i = 0; i < numDevices; i++) {
    if (slots[i].deviceId == deviceId) return slots + i;
  }
  #ifdef STATIC_SLOT_ASSIGN
    #ifdef DEBUG_PRINT
      Serial.printf("Unexpected device id: %d\r\n", deviceId);
    #endif
    return nullptr;
  #else
    #ifdef DEBUG_PRINT
      if (receiving) {
        Serial.printf("Device ids:  {");
        for (int i = 0; i < numDevices; i++) {
          Serial.print(slots[i].deviceId);
          if (i < numDevices - 1) {
            Serial.print(", ");
          }
        }
        Serial.println("}");
        Serial.printf("Assigning new device [%d] slot {%d}\n", deviceId, nextMessageSlot % numDevices);
      }
    #endif
    uint8_t messageIdx = nextMessageSlot % numDevices;
    slots[messageIdx].setId(deviceId);
    nextMessageSlot++;
    return slots + messageIdx;
  #endif
}

#ifdef DEBUG_PRINT
bool Communicator::send(uint8_t deviceId, uint8_t dataType, int8_t* data, uint8_t dataLength, bool manualSend=false) {
#else
bool Communicator::send(uint8_t deviceId, uint8_t dataType, int8_t* data, uint8_t dataLength) {
#endif
  #ifdef TEMP_MEM_SLOT
  slot = getSlot(deviceId);
  #else
  MessageSlot* slot = getSlot(deviceId);
  #endif
  #ifdef STATIC_SLOT_ASSIGN
    if (!slot) return false;
  #endif
  #ifdef ENABLE_PING
  if (dataType == PING && slot->pingStart != -1) return false;
  if (dataType != ACK && dataType != PING) {
  #elif defined(ENABLE_HOLD)
  if (dataType == HOLD) {
    if ((data[0] == SIGNAL_HOLD && slot->receiveHold) || slot->waitingForReleaseAck != -1) return false;
  } else if (dataType != ACK) {
  #else
  if (dataType != ACK) {
  #endif
    if (!slot->ready()) return false;
    // Store data for resending if necessary
    slot->sendType = dataType;
    slot->sendSize = dataLength;
    memcpy(slot->sendBuffer, data, dataLength * sizeof(int8_t));
  } else if (dataType == ACK) {
    // Clear data available when ack sent
    slot->dataAvailable = 0;
    #ifdef TRACK_STATS
      slot->receiveAttempts++;
    #endif
    #ifdef ENABLE_HOLD
    if (data[0] == SENDANDHOLD_ACK) {
      slot->receiveHold = true;
      #ifdef HOLD_INIT_IGNORE
      slot->ignoreMessageStart = true;
      #endif
    }
    #endif
  }
  int messageId = dataType == ACK && slot->lastReceivedId != NULL_ID ? slot->lastReceivedId : slot->sendCounter;
  #ifdef DEBUG_PRINT
    if (manualSend) Serial.printf("Sending message (%d)\n", messageId);
  #endif
  // Send data
  uint8_t idLORByte = slot->selfId << 5;
  #if defined(DEBUG_PRINT) && defined(PRINT_SEND_BYTES)
    Serial.print("Sending: {");
  #endif
  write(idLORByte | START_BYTE);
  write(idLORByte | (deviceId + DEVICE_ID_OFFSET));
  crc8Write(idLORByte, dataType + DATA_TYPE_OFFSET);
  crc8Write(idLORByte, messageId + MSG_ID_OFFSET);
  #ifdef ENABLE_PING
  if (dataType != PING) {
  #elif defined(ENABLE_HOLD)
  if (true) {
  #else
  if (dataType != ACK) {
  #endif
    nibbleSplitWrite(idLORByte, crc8((uint8_t*)data, dataLength));
    for (int i = 0; i < dataLength; i++) {
      nibbleSplitWrite(idLORByte, (uint8_t)data[i]);
    }
  }
  write(idLORByte | END_BYTE);
  #if defined(DEBUG_PRINT) && defined(PRINT_SEND_BYTES)
    Serial.print("}\n");
  #endif
  #ifdef BUFFERED_WRITE
    strm.write(message, messageIdx);
    messageIdx = 0;
  #endif
  #ifdef ENABLE_PING
  if (dataType == PING) {
    slot->pingStart = millis();
  } else if (dataType != ACK) {
  #elif defined(ENABLE_HOLD)
  if (dataType == HOLD) {
    if (data[0] == SIGNAL_HOLD) {
      slot->receiveHold = true;
      #ifdef HOLD_INIT_IGNORE
      slot->ignoreMessageStart = true;
      #endif
    } else {
      slot->receiveHold = false;
      slot->waitingForReleaseAck = millis();
    }
  } else if (dataType != ACK) {
  #else
  if (dataType != ACK) {
  #endif
    #ifdef TRACK_SEND_STATS
      slot->sendAttempts++;
    #endif
    slot->waitingForAck = millis();
    slot->waitTime = WAIT_TIMEOUT + random(0, WAIT_TIMEOUT);
  }
  return true;
}

#if defined(ENABLE_PING) || defined(ENABLE_HOLD)
void Communicator::ack(uint8_t deviceId, uint8_t ackType) {
  sendAckType = ackType;
  send(deviceId, ACK, &sendAckType, 1);
}
  #ifdef ENABLE_PING
  void Communicator::ping(uint8_t deviceId) {
    send(deviceId, PING, nullptr, 1);
  }
  #else
  void Communicator::hold(uint8_t deviceId, bool enable) {
    sendAckType = enable ? SIGNAL_HOLD : SIGNAL_RELEASE;
    send(deviceId, HOLD, &sendAckType, 1);
  }
  #endif
#else
void Communicator::ack(uint8_t deviceId) {
  send(deviceId, ACK, nullptr, 1);
}
#endif

bool MessageSlot::stagedVerify(uint8_t* currentData, uint8_t newData, uint8_t offset, uint8_t dataMax) {
  if (dataMax > 0 && tempIdx == 0 && (newData < offset || newData >= dataMax)) {
    // Invalid data, reset message
    messagePos = 0;
    tempIdx = 0;
  } else if (tempIdx == 0) {
    *currentData = newData;
    tempIdx++;
  } else if (tempIdx == 1) {
    tempByte = newData - DATA_VALUE_OFFSET;
    tempIdx++;
  } else if (tempIdx == 2 && ((tempByte << 4) | (newData - DATA_VALUE_OFFSET)) == crc8(currentData, 1)) {
    messagePos++; // Move to next message stage
    tempIdx = 0;
    return true;
  } else {
    messagePos = 0;
    tempIdx = 0;
  }
  return false;
}

void MessageSlot::handleAck() {
  #ifdef ENABLE_PING
  if (ackType == SEND_ACK) {
  #elif defined(ENABLE_HOLD)
  if (ackType == SEND_ACK || ackType == SENDANDHOLD_ACK) {
  #else
  if (true) {
  #endif
    #ifdef DEBUG_PRINT
      Serial.printf("Got ack for message (%u)\n", receivedMessageId);
    #endif
    if (receivedMessageId == sendCounter) {
      waitingForAck = -1;
      updateCounter();
      #ifdef TRACK_SEND_STATS
        sendAttempts = 0;
      #endif
      #ifdef DEBUG_PRINT
        Serial.println("Increasing send counter");
      #endif
    }
    #ifdef ENABLE_HOLD
    if (ackType == SENDANDHOLD_ACK) sendHold = true;
    #endif
    messagePos = 0; // Can ignore end byte and data
  }
  #ifdef ENABLE_PING
  else if (ackType == PING_ACK) {
    #ifdef DEBUG_PRINT
      if (pingStart != -1) {
        Serial.printf("Response from device %u after %d ms.\n", deviceId, millis() - pingStart);
      } else {
        Serial.printf("Got unexpected ping ack from device %u.\n", deviceId);
      }
    #endif
    pingStart = -1;
  }
  #elif defined(ENABLE_HOLD)
  else if (ackType == HOLD_ACK) {
    #ifdef DEBUG_PRINT
      if (waitingForReleaseAck != -1) {
        Serial.printf("Device %u back online.\n", deviceId);
      } else {
        Serial.printf("Got unexpected release ack from device %u.\n", deviceId);
      }
    #endif
    waitingForReleaseAck = -1;
  }
  #endif
}

#if defined(ENABLE_PING) || defined(ENABLE_HOLD)
uint8_t MessageSlot::update(uint8_t data) {
#else
bool MessageSlot::update(uint8_t data) {
#endif
  if (data == START_BYTE) {
    // Start collecting data for message
    messagePos = 1;
    tempIdx = 0;
  } else if (data == END_BYTE) {
    #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
    if (messagePos > MSG_DATA && crc == crc8((uint8_t*)dataBuffer, messagePos - MSG_DATA)) {
    #else
    if (messagePos > MSG_DATA && crc == crc8((uint8_t*)receiveBuffer, messagePos - MSG_DATA)) {
    #endif
      #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
      if (dataType == ACK) {
        handleAck();
        return NO_ACK;
      }
      #endif
      #ifdef ENABLE_HOLD
      if (dataType == HOLD) {
        if (ackType == SIGNAL_RELEASE) {
          // Only send ack if device is not currently on hold
          // Sending next data will act as ACK
          // If for any reason receiver did not receive next data arr, it will resend RELEASE
          // Only send ACK if not holding and receiver sends RELEASE
          // - Intended to mitigate receiver resending RELEASE when this device has already processed a RELEASE,
          //   but the send data is not ready so it has not sent data to the receiver to confirm the RELEASE has been processed.
          if (!sendHold) return HOLD_ACK;
          sendHold = false;
          return NO_ACK;
        }
        sendHold = true;
        return NO_ACK;
      }
      #endif
      dataAvailable = messagePos - MSG_DATA;
      #ifdef DEBUG_PRINT
        Serial.printf("Device %d last message: (%u) -> (%u)\n", deviceId, lastReceivedId, receivedMessageId);
      #endif
      lastReceivedId = receivedMessageId;
      #ifdef TRACK_STATS
        messagesReceived++;
        avgAttemptsPerReceive += (float)(receiveAttempts - avgAttemptsPerReceive) / messagesReceived;
        maxReceiveAttempts = max(maxReceiveAttempts, receiveAttempts);
        receiveAttempts = 0;
      #endif
    }
    // Terminate message and mark device as having data ready
    messagePos = 0;
  } else if (messagePos == MSG_RECEIVER) {
    data -= DEVICE_ID_OFFSET;
    if (data < 0 || data >= DATA_TYPE_OFFSET - DEVICE_ID_OFFSET || data != selfId) {
      // Invalid deviceId or message not for this device, reset message
      messagePos = 0;
    #ifdef ENABLE_HOLD
    } else if (receiveHold) { // Received a message from device deviceId, when it should be "off"
      messagePos = 0;
      receiveHold = false; // Allow COMM to resend hold request
      return HOLD_NACK;
    #endif
    } else {
      messagePos++; // Next check recipient id
      #ifdef ENABLE_HOLD
        if (!receiveHold && waitingForReleaseAck >= 0) {
          waitingForReleaseAck = -1;
        }
      #endif
    }
  } else if (messagePos == MSG_TYPE && stagedVerify(&dataType, data, DATA_TYPE_OFFSET, MSG_ID_OFFSET)) {
    dataType -= DATA_TYPE_OFFSET;
    #ifdef ENABLE_PING
      if (dataType == PING)  {
        messagePos = 0;
        return PING_ACK;
      } else if (dataType != ACK) receiveType = dataType;
    #elif defined(ENABLE_HOLD)
      if (dataType == SEND) receiveType = dataType;
    #else
      if (dataType != ACK) receiveType = dataType;
    #endif
  } else if (messagePos == MSG_ID && stagedVerify(&receivedMessageId, data, MSG_ID_OFFSET, DATA_VALUE_OFFSET)) {
    // Confirm that data was intended to be message id
    receivedMessageId -= MSG_ID_OFFSET;
    #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
    dataBuffer = receiveBuffer;
    maxData = receiveSize;
    #endif
    if (dataType == ACK) {
      #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
        dataBuffer = &ackType;
        maxData = 1;
      #else
        handleAck();
      #endif
    }
    #ifdef ENABLE_HOLD
    else if (dataType == HOLD) {
      dataBuffer = &ackType; // Reuse to avoid allocating more resources
      maxData = 1;
    }
    #endif
    else if (lastReceivedId != NULL_ID && lastReceivedId == receivedMessageId || dataAvailable > 0) {
      // Message already acknowledged by main program, send auto ack
      messagePos = 0;
      #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
        if (dataAvailable == 0) return SEND_ACK;
      #else
        if (dataAvailable == 0) return true;
      #endif
    }
  #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
  } else if (messagePos > MSG_ID && (messagePos == MSG_CRC || messagePos - MSG_DATA < maxData)) {
  #else
  } else if (messagePos > MSG_ID && (messagePos == MSG_CRC || messagePos - MSG_DATA < receiveSize)) { // len(data in receive buffer) < bufferSize
  #endif
    if (data < DATA_VALUE_OFFSET || data > DATA_VALUE_MAX) {
      // Corrupted data, discard message
      messagePos = 0;
    } else if (tempIdx % 2) {
      tempByte = ((tempByte - DATA_VALUE_OFFSET) << 4) | (data - DATA_VALUE_OFFSET);
      if (messagePos == MSG_CRC) {
        crc = tempByte;
      } else {
        #if defined(ENABLE_PING) || defined(ENABLE_HOLD) || defined(ENABLE_COMMAND)
          dataBuffer[messagePos - MSG_DATA] = tempByte;
        #else
          receiveBuffer[messagePos - MSG_DATA] = tempByte;
        #endif
      }
      messagePos++;
      tempIdx = 0;
    } else {
      tempByte = data;
      tempIdx++;
    }
  }
  #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
    return NO_ACK;
  #else
    return false;
  #endif
}

#ifdef UPDATE_ON_END
bool Communicator::update() {
#else
void Communicator::update() {
#endif
  #ifndef TEMP_MEM_SLOT
  MessageSlot* slot;
  #endif
  int16_t packet = strm.read();
  uint8_t data;

  #ifdef UPDATE_ON_END
  bool messageEnded = false;
  #endif
  while (packet != -1) {
    data = (uint8_t)packet & 0x1F;
    slot = getSlot((uint8_t)packet >> 5);
    #ifdef UPDATE_ON_END
    if (slot && data == END_BYTE) messageEnded = true;
    #endif
    #ifdef TRACK_INTERLEAVING
    allSlots++;
    if (slot) {
      processedSlots++;
      if (lastDeviceId != NULL_ID && slot->deviceId != lastDeviceId) {
        interleaveApprox++;
      }
      lastDeviceId = (data == END_BYTE) ? NULL_ID : slot->deviceId;
    }
    #endif
    #ifdef ENABLE_PING
    if (slot) {
      uint8_t ackType = slot->update(data);
      if (ackType != NO_ACK) ack(slot->deviceId, ackType);
    }
    #elif defined(ENABLE_HOLD)
    if (slot) {
      uint8_t ackType = slot->update(data);
      if (ackType == HOLD_NACK) {
        // Got data from device, make sure it knows to stop sending
        hold(slot->deviceId, true);
        #ifdef DEBUG_PRINT
          Serial.printf("Resending hold request to device %u\n", slot->deviceId);
        #endif
      } else if (ackType != NO_ACK) ack(slot->deviceId, ackType);
    }
    #else
    if (slot && slot->update(data)) {
      ack(slot->deviceId);
    }
    #endif
    #ifdef DEBUG_PRINT
    else if (!slot) {
      Serial.printf("no slot, strm fill: %d\n", strm.available());
    }
    #endif
    packet = strm.read();
  }

  // Check if resend necessary
  for (int messageIdx = 0; messageIdx < numDevices; messageIdx++) {
    slot = slots + messageIdx;
    #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
      if (slot->deviceId == NULL_ID) continue;
      // Resend data
      if (slot->waitingForAck >= 0 && millis() - slot->waitingForAck >= slot->waitTime) {
        slot->waitingForAck = -1;
        send(slot->deviceId, slot->sendType, slot->sendBuffer, slot->sendSize);
      }
      #ifdef ENABLE_PING
      ping(slot->deviceId);
      if (slot->pingStart >= 0 && millis() - slot->pingStart >= PING_TIMEOUT) {
        #ifdef DEBUG_PRINT
          Serial.printf("No response from device %u after %d ms.\n", slot->deviceId, millis() - slot->pingStart);
        #endif
        slot->pingStart = -1;
      }
      #else
      if (slot->waitingForReleaseAck >= 0 && millis() - slot->waitingForReleaseAck >= HOLD_RELEASE_TIMEOUT) {
        #ifdef DEBUG_PRINT
          Serial.printf("Resending hold release request to device %u\n", slot->deviceId);
        #endif
        slot->waitingForReleaseAck = -1;
        hold(slot->deviceId, false);
      }
      #endif
    #else
      if (slot->deviceId == NULL_ID || slot->waitingForAck < 0 || millis() - slot->waitingForAck < slot->waitTime) continue;
      // Resend data
      slot->waitingForAck = -1;
      send(slot->deviceId, slot->sendType, slot->sendBuffer, slot->sendSize);
    #endif
  }
  #ifdef UPDATE_ON_END
  return messageEnded;
  #endif
}
