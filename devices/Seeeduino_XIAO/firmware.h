#include "wiring_private.h"
// https://sigmdel.ca/michel/ha/xiao/seeeduino_xiao_3usarts_en.html
constexpr int COMM_RX_PIN = A3;
constexpr int COMM_TX_PIN = A2;
Uart COMMStream(&sercom2, COMM_RX_PIN, COMM_TX_PIN, SERCOM_RX_PAD_3, UART_TX_PAD_2);

extern "C" {
  void SERCOM2_Handler(void) {
    COMMStream.IrqHandler();
  }
}

void deviceSetup() {
  COMMStream.begin(115200);
  if (pinPeripheral(COMM_TX_PIN, PIO_SERCOM_ALT) || pinPeripheral(COMM_RX_PIN, PIO_SERCOM_ALT)) {
    Serial.println("Failed to initialize COMMStream.");
    while (true);
  } else {
    Serial.println("Initialized COMMStream.");
  }
}
