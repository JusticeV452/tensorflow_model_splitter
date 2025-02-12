#ifndef MACROS_H
#define MACROS_H
#include "Arduino.h"

// #define TWO_STAGE_VERIFY
#define STATIC_SLOT_ASSIGN
// #define RANDOM_WRITE_DELAY
// #define SLOT_MALLOC
#ifndef SLOT_MALLOC
#define SEND_BUFFER_SIZE 12
#define RECEIVE_BUFFER_SIZE 1
#endif

#define STM32C0116_DK
#define ADD_REDUCE
// #define DEBUG
// #define DEBUG_PRINT
#define BUFFERED_WRITE
// #define TRACK_STATS
// #define TRACK_SEND_STATS
// #define TRACK_INTERLEAVING
// #define ENABLE_PING
#define ENABLE_HOLD
#define USE_HOLD
#define HARDWARE_CRC
#define INPUT_NODE
// #define OUTPUT_NODE
#define UPDATE_ON_END
// #define DMA_UART
// #define STATE_MACHINE_COMM
// #define USE_ACC
// #define TEMP_MEM_SLOT
constexpr int DEVICE_ID = 1;
constexpr int NUM_DEVICES = 2;
constexpr int NUM_THRESHOLDS = 0;
constexpr bool IS_MULTI_INPUT = NUM_THRESHOLDS > 0;
constexpr int INPUT_THRESHOLDS[] = {};
constexpr bool IS_RECURRENT = false;
constexpr uint8_t RECEIVE_ORDER[] = {2};
constexpr uint8_t OUT_DEVICE = 2;
constexpr bool IS_INPUT_NODE = true;
constexpr bool IS_OUTPUT_NODE = false;

#endif