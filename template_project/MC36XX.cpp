/******************************************************************************
 *
 * Copyright (c) 2018 mCube, Inc.  All rights reserved.
 *
 * This source is subject to the mCube Software License.
 * This software is protected by Copyright and the information and source code
 * contained herein is confidential. The software including the source code
 * may not be copied and the information contained herein may not be used or
 * disclosed except with the written permission of mCube Inc.
 *
 * All other rights reserved.
 *****************************************************************************/

/**
 * @file    MC36XX.c
 * @author  mCube
 * @date    10 May 2018
 * @brief   Driver interface header file for accelerometer mc36xx series.
 * @see     http://www.mcubemems.com
 */

#include "MC36XX.h"

#ifdef MC36XX_CFG_BUS_I2C
    #define MC36XX_CFG_I2C_ADDR        (0x4C)
#endif
#define MC36XX_CFG_MODE_DEFAULT                 MC36XX_MODE_STANDBY
#define MC36XX_CFG_SAMPLE_RATE_CWAKE_DEFAULT    MC36XX_CWAKE_SR_DEFAULT_54Hz
#define MC36XX_CFG_SAMPLE_RATE_SNIFF_DEFAULT    MC36XX_SNIFF_SR_105Hz//MC36XX_SNIFF_SR_DEFAULT_7Hz
#define MC36XX_CFG_RANGE_DEFAULT                MC36XX_RANGE_8G
#define MC36XX_CFG_RESOLUTION_DEFAULT           MC36XX_RESOLUTION_12BIT
#define MC36XX_CFG_ORIENTATION_MAP_DEFAULT      ORIENTATION_TOP_RIGHT_UP

uint8_t CfgRange, CfgResolution, CfgFifo, CfgINT;

// Read 8-bit from register
uint8_t MC36XX::readRegister8(uint8_t reg)
{
    uint8_t value;

#ifdef MC36XX_CFG_BUS_I2C
    //Wire.beginTransmission(MC36XX_CFG_I2C_ADDR);
    //Wire.write(reg);
    //endTransmission but keep the connection active
    //Wire.endTransmission(false);
    //Once done, bus is released by default
    //Wire.requestFrom(MC36XX_CFG_I2C_ADDR, 1);
    Wire.requestFrom(MC36XX_CFG_I2C_ADDR, 1, reg, 1, true);
    value = Wire.read();
#else  //Reads an 8-bit register with the SPI port.
    //Set active-low CS low to start the SPI cycle
    digitalWrite(chipSelectPin, LOW);
    //Send the register address
    SPI.transfer(reg | 0x80 | 0x40);
    //Read the value from the register
    value = SPI.transfer(0x00);
    //Raise CS
    digitalWrite(chipSelectPin, HIGH);
#endif

    return value;
}

// Repeated Read Byte(s) from register
void MC36XX::readRegisters(uint8_t reg, byte *buffer, uint8_t len)
{
#ifdef MC36XX_CFG_BUS_I2C
    Wire.beginTransmission(MC36XX_CFG_I2C_ADDR);
    Wire.write(reg);
    //endTransmission but keep the connection active
    Wire.endTransmission(false);
    //Ask for bytes, once done, bus is released by default
    Wire.requestFrom(MC36XX_CFG_I2C_ADDR, len);

    //Hang out until we get the # of bytes we expect
    while(Wire.available() < len);
    for(int x = 0 ; x < len ; x++)
        buffer[x] = Wire.read();
#else
    //Set active-low CS low to start the SPI cycle
    digitalWrite(chipSelectPin, LOW);
    //send the device the register you want to read
    SPI.transfer(reg | 0x80 | 0x40);

    //Prepare to clock in the data to be read
    for(int x = 0 ; x < len ; x++)
        buffer[x] = SPI.transfer(0x00);
    //Raise CS
    digitalWrite(chipSelectPin, HIGH);
#endif

}

// Write 8-bit to register
void MC36XX::writeRegister8(uint8_t reg, uint8_t value)
{
#ifdef MC36XX_CFG_BUS_I2C
    Wire.beginTransmission(MC36XX_CFG_I2C_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
#else
    //Set active-low CS low to start the SPI cycle
    digitalWrite(chipSelectPin, LOW);
    //Send the register address
    SPI.transfer(reg | 0x40);
    //Send value to write into register
    SPI.transfer(value);
    //Raise CS
    digitalWrite(chipSelectPin, HIGH);
#endif
}

//Initialize the MC36XX sensor and set as the default configuration
bool MC36XX::start(void)
{

#ifdef MC36XX_CFG_BUS_I2C
    // Initialize I2C
    // #ifdef MY_STM32C0116_DK
    Wire.setSDA(PB7);
    Wire.setSCL(PB6);
    // #endif
    Wire.begin();
#else
    //Set active-low CS low to start the SPI cycle
    digitalWrite(chipSelectPin, HIGH);
    pinMode(chipSelectPin, OUTPUT);
// SPI.setMOSI(PA7);
// SPI.setSCLK(PA5);
    // #ifdef STM32C0116_DK
    //   SPI.setMOSI(PA7);
    //   SPI.setSCLK(PA5);
    // #endif
    SPI.begin();
    SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
#endif

    //Init Reset
    reset();
    SetMode(MC36XX_MODE_STANDBY);

    /* Check I2C connection */
    uint8_t id = readRegister8(MC36XX_REG_PROD);
    if (id != 0x71)
    {
        /* No MC36XX detected ... return false */
        Serial.println("No MC36XX detected!");
        Serial.println(id, HEX);
        return false;
    }

//Running mc36xx in high speed SPI (8MHz)
#ifdef SPI_HS
    uint8_t value;
    value = readRegister8(MC36XX_REG_FEATURE_CTL);
    value &= 0b00111111;
    value |= 0x80 ;
    writeRegister8(MC36XX_REG_FEATURE_CTL, value);

    value = readRegister8(MC36XX_REG_PMCR);
    value |= 0x80 ;
    writeRegister8(MC36XX_REG_PMCR, value);
    SPI.beginTransaction(SPISettings(8000000, MSBFIRST, SPI_MODE3));
    delay(50);
#endif

    //Range: 8g
    SetRangeCtrl(MC36XX_CFG_RANGE_DEFAULT);
    //Resolution: 12bit
    SetResolutionCtrl(MC36XX_CFG_RESOLUTION_DEFAULT);
    //Sampling Rate: 50Hz by default
    SetCWakeSampleRate(MC36XX_CFG_SAMPLE_RATE_CWAKE_DEFAULT);
	  //Sampling Rate: 7Hz by default
    // SetSniffSampleRate(MC36XX_CFG_SAMPLE_RATE_SNIFF_DEFAULT);
    //Mode: Active
    SetMode(MC36XX_MODE_CWAKE);

    delay(50);

    return true;
}

void MC36XX::wake()
{
    //Set mode as wake
    SetMode(MC36XX_MODE_CWAKE);
}

void MC36XX::stop()
{
    //Set mode as Sleep
    SetMode(MC36XX_MODE_STANDBY);
}

//Initial reset
void MC36XX::reset()
{
    writeRegister8(0x10, 0x01);

    delay(10);

    writeRegister8(0x24, 0x40);

    delay(50);

    writeRegister8(0x09, 0x00);
    delay(10);
    writeRegister8(0x0F, 0x42);
    delay(10);
    writeRegister8(0x20, 0x01);
    delay(10);
    writeRegister8(0x21, 0x80);
    delay(10);
    writeRegister8(0x28, 0x00);
    delay(10);
    writeRegister8(0x1a, 0x00);

    delay(50);

    uint8_t _bRegIO_C = 0;

    _bRegIO_C = readRegister8(0x0D);

    #ifdef MC36XX_CFG_BUS_I2C
        _bRegIO_C &= 0x3F;
        _bRegIO_C |= 0x40;
    #else
        _bRegIO_C &= 0x3F;
        _bRegIO_C |= 0x80;
    #endif

    writeRegister8(0x0D, _bRegIO_C);

    delay(50);

    writeRegister8(0x10, 0x01);

    delay(10);
}

//Set the operation mode
void MC36XX::SetMode(MC36XX_mode_t mode)
{
    uint8_t value;
	  uint8_t cfgfifovdd = 0x42;
	
    value = readRegister8(MC36XX_REG_MODE_C);
    value &= 0b11110000;
    value |= mode;
	
	  writeRegister8(MC36XX_REG_PWR_CONTROL, cfgfifovdd);
    writeRegister8(MC36XX_REG_MODE_C, value);
}

//Set the range control
void MC36XX::SetRangeCtrl(MC36XX_range_t range)
{
    uint8_t value;
    CfgRange = range;
    SetMode(MC36XX_MODE_STANDBY);
    value = readRegister8(MC36XX_REG_RANGE_C);
    value &= 0b00000111;
    value |= (range << 4)&0x70 ;
    writeRegister8(MC36XX_REG_RANGE_C, value);
}

//Set the resolution control
void MC36XX::SetResolutionCtrl(MC36XX_resolution_t resolution)
{
    uint8_t value;
    CfgResolution = resolution;
    SetMode(MC36XX_MODE_STANDBY);
    value = readRegister8(MC36XX_REG_RANGE_C);
    value &= 0b01110000;
    value |= resolution;
    writeRegister8(MC36XX_REG_RANGE_C, value);
}

//Set the sampling rate
void MC36XX::SetCWakeSampleRate(MC36XX_cwake_sr_t sample_rate)
{
    uint8_t value;
    SetMode(MC36XX_MODE_STANDBY);
    value = readRegister8(MC36XX_REG_WAKE_C);
    value &= 0b00000000;
    value |= sample_rate;
    writeRegister8(MC36XX_REG_WAKE_C, value);
}

//Read the raw counts and SI units measurement data
void MC36XX::readRawAccel(void)
{
    int16_t faResolution[6] = {32, 64, 128, 512, 2048, 8192};

    byte rawData[6];
    // Read the six raw data registers into data array
    readRegisters(MC36XX_REG_XOUT_LSB, rawData, 6);
    x = (short) ((((unsigned short) rawData[1]) << 8) | rawData[0]);
    y = (short) ((((unsigned short) rawData[3]) << 8) | rawData[2]);
    z = (short) ((((unsigned short) rawData[5]) << 8) | rawData[4]);

    readOutput[0] = ((x * 128) / faResolution[CfgResolution]) > 127 ? 127 : (x * 128) / faResolution[CfgResolution];
    readOutput[1] = ((x * 128) / faResolution[CfgResolution]) > 127 ? 127 : (y * 128) / faResolution[CfgResolution];
    readOutput[2] = ((x * 128) / faResolution[CfgResolution]) > 127 ? 127 : (z * 128) / faResolution[CfgResolution];
    delay(5);
}
