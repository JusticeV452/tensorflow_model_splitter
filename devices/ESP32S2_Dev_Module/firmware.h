#include <HardwareSerial.h>
constexpr int COMM_RX_PIN = 20;
constexpr int COMM_TX_PIN = 19;
constexpr int BUTTON_PIN = 45;
constexpr int OUT_LED1 = 10;
constexpr int OUT_LED2 = 12;
constexpr int OUT_LED3 = 11;
HardwareSerial COMMStream = Serial1;

void deviceSetup() {
  pinMode(COMM_TX_PIN, OUTPUT);
  pinMode(COMM_RX_PIN, INPUT);
  COMMStream.begin(115200, SERIAL_8N1, COMM_RX_PIN, COMM_TX_PIN);
}
