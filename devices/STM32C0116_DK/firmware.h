#ifndef FIRMWARE_H
#define FIRMWARE_H
#include <HardwareSerial.h>

constexpr int COMM_RX_PIN = PA3;
constexpr int COMM_TX_PIN = PA2;
constexpr int _SDA = PB7;
constexpr int _SCL = PB6;
constexpr int BUTTON_PIN = PA8;
constexpr int OUT_LED1 = PA4;
constexpr int OUT_LED2 = PA6;
constexpr int OUT_LED3 = PA7;

HardwareSerial COMMStream(COMM_RX_PIN, COMM_TX_PIN);

#ifdef HARDWARE_CRC
uint8_t crc8(const uint8_t* data, size_t length) {
  // Reset CRC peripheral before each calculation
  CRC->CR = CRC_CR_RESET;

  for (size_t i = 0; i < length; i++) {
    *(uint8_t*)&CRC->DR = data[i];
  }
  
  // lower 8 bits are contain CRC-8
  return (uint8_t)(CRC->DR);
}
#endif

void deviceSetup() {
  const int _RX = PA10_R;
  const int _TX = PA9_R;
  Serial.setRx(_RX);
  Serial.setTx(_TX);
  #ifdef HARDWARE_CRC
    // Enable the CRC peripheral clock
    RCC->AHBENR |= RCC_AHBENR_CRCEN;

    // Reset the CRC peripheral
    CRC->CR = CRC_CR_RESET;

    // Configure CRC for 8-bit
    CRC->INIT = 0xFF;           // Initial CRC value
    CRC->POL = 0x7;
  #endif
  COMMStream.begin(115200);
}
#endif

