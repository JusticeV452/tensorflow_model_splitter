#ifndef COMM_H
#define COMM_H
#include "macros.h"
#include "Arduino.h"

#define START_BYTE (0)
#define END_BYTE (1)
#define DEVICE_ID_OFFSET (2)
#define DATA_TYPE_OFFSET (10)
#define MSG_ID_OFFSET (13)
#define DATA_VALUE_OFFSET (16)
#define DATA_VALUE_RANGE (15)
#define DATA_VALUE_MAX (DATA_VALUE_OFFSET + DATA_VALUE_RANGE)

#ifdef RANDOM_WRITE_DELAY
  #define WR_DELAY_MIN 0
  #define WR_DELAY_MAX 5
  #define MAX_MSG_LEN 50
  #define WAIT_TIMEOUT WR_DELAY_MAX * MAX_MSG_LEN // ms
#else
  #define WAIT_TIMEOUT (10)
#endif
constexpr uint8_t CRC_POLY = 0b00001011; // Polynomial for x^3 + x^1 + 1
#ifdef BUFFERED_WRITE
  constexpr uint8_t MESSAGE_LEN = 10 + 2 * SEND_BUFFER_SIZE;
// Msg size (bytes) 8 + (2 + 2 * max(sendsize)) + 1
#endif

#ifdef ENABLE_PING
  #define PING_TIMEOUT 100
  enum DataTypes {SEND, ACK, PING};
  enum AckTypes {NO_ACK, SEND_ACK, PING_ACK};
#elif defined(ENABLE_HOLD)
  #define HOLD_RELEASE_TIMEOUT 20
  enum DataTypes {SEND, ACK, HOLD};
  enum AckTypes {NO_ACK, SEND_ACK, HOLD_ACK, SENDANDHOLD_ACK, HOLD_NACK};
  enum HoldTypes {SIGNAL_RELEASE, SIGNAL_HOLD};
#else
  enum DataTypes {SEND, ACK};
#endif
enum MessageStates {NO_MSG, MSG_RECEIVER, MSG_TYPE, MSG_ID, MSG_CRC, MSG_DATA};
enum ERROR_CODES {
  FAILED_GENERAL_MEM_ALLOC,
  FAILED_BUFFER_ALLOC
};

uint8_t crc8(const uint8_t* data, size_t length);

class MessageSlot {
  public:
  // Declaration/Allocation order affects program storage space (tested on STM32C0)
  #ifdef SLOT_MALLOC
    MessageSlot(uint8_t sendBufferSize, uint8_t receiveBufferSize);
    ~MessageSlot();
    uint8_t deviceId;
    uint8_t receiveSize;
  #else
    uint8_t deviceId = NULL_ID;
    uint8_t receiveSize = RECEIVE_BUFFER_SIZE;
  #endif
  void setId(uint8_t id);
  void updateCounter();
  void discardMessage();
  uint8_t dataAvailable;
  uint8_t sendType;
  uint8_t sendSize;
  uint8_t receiveType;  
  uint8_t sendCounter = 0;
  uint8_t waitTime;
  uint8_t lastReceivedId;
  uint8_t selfId;
  uint8_t messagePos = 0;
  uint8_t tempIdx = 0;
  uint8_t dataType;
  uint8_t receivedMessageId;
  uint8_t tempByte;
  uint8_t crc;
  #ifdef STATE_MACHINE_COMM
  uint8_t state = NO_MSG;
  #endif
  long waitingForAck;
  #ifdef SLOT_MALLOC
    int8_t* sendBuffer;
  #else
    int8_t sendBuffer[SEND_BUFFER_SIZE];
  #endif
  #ifdef SLOT_MALLOC
    int8_t* receiveBuffer;
  #else
    int8_t receiveBuffer[RECEIVE_BUFFER_SIZE];
  #endif
  #ifdef TRACK_SEND_STATS
    unsigned long sendAttempts = 0;
  #endif
  #ifdef TRACK_STATS
    unsigned long messagesReceived = 0;
    unsigned long receiveAttempts = 0;
    float avgAttemptsPerReceive = 0;
    unsigned long maxReceiveAttempts = 0;
    void updateStats();
  #endif
  bool ready();
  void handleAck();
  #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
  int8_t* dataBuffer;
  int8_t ackType;
  uint8_t maxData;
  #ifdef ENABLE_PING
    long pingStart = -1;
  #else
    bool sendHold = false;
    bool receiveHold = false;
    long waitingForReleaseAck = -1;
  #endif
  uint8_t update(uint8_t data);
  #else
  bool update(uint8_t data);
  #endif
  bool stagedVerify(uint8_t* currentData, uint8_t newData, uint8_t offset, uint8_t dataMax);
};

class Communicator {
  public:
  #ifdef SLOT_MALLOC
    Communicator(uint8_t id, uint8_t numDevices, uint8_t sendBufferSize, uint8_t receiveBufferSize, Stream& stream);
  #else
    Communicator(uint8_t id, uint8_t numDevices, Stream& stream);
  #endif
  ~Communicator();
  int numDevices;
  MessageSlot* slots;
  #ifdef BUFFERED_WRITE
  uint8_t message[MESSAGE_LEN];
  uint8_t messageIdx = 0;
  #endif
  bool ready(uint8_t deviceId);
  #ifdef DEBUG_PRINT
    bool send(uint8_t device_id, uint8_t dataType, int8_t* data, uint8_t data_len, bool manualSend);
  #else
    bool send(uint8_t device_id, uint8_t dataType, int8_t* data, uint8_t data_len);
  #endif
  #ifdef UPDATE_ON_END
  bool update();
  #else
  void update();
  #endif
  int8_t getMessageIdx(uint8_t deviceId);
  #if defined(ENABLE_PING) || defined(ENABLE_HOLD)
    int8_t sendAckType;
    void ack(uint8_t deviceId, uint8_t ackType=SEND_ACK);
    #ifdef ENABLE_PING
      void ping(uint8_t deviceId);
    #else
      void hold(uint8_t deviceId, bool enable=true);
    #endif
  #else
    void ack(uint8_t deviceId);
  #endif
  #ifdef DEBUG_PRINT
    MessageSlot* getSlot(uint8_t deviceId, bool receiving=false);
  #else
    MessageSlot* getSlot(uint8_t deviceId);
  #endif
  #ifdef TRACK_INTERLEAVING
  uint8_t lastDeviceId = NULL_ID;
  unsigned long interleaveApprox = 0;
  unsigned long processedSlots = 0;
  unsigned long allSlots = 0;
  #endif
  private:
  Stream& strm;
  #ifndef STATIC_SLOT_ASSIGN
    uint8_t nextMessageSlot;
  #endif
  #ifdef TEMP_MEM_SLOT
  MessageSlot* slot;
  #endif
  void write(uint8_t byte);
  void crc8Write(uint8_t orByte, uint8_t sendByte);
  void nibbleSplitWrite(uint8_t orByte, uint8_t sendByte);
};
#endif
