// Funshield constants
// v1.0.1 18.2.2024

#ifndef FUNSHIELD_CONSTANTS_H__
#define FUNSHIELD_CONSTANTS_H__

// convenience constants for switching on/off
constexpr int ON = LOW;
constexpr int OFF = HIGH;

// 7-segs
constexpr int latch_pin = 4;
constexpr int clock_pin = 7;
constexpr int data_pin = 8;

// buzzer
constexpr int beep_pin = 3;

// LEDs
constexpr int led1_pin = 13;
constexpr int led2_pin = 12;
constexpr int led3_pin = 11;
constexpr int led4_pin = 10;

// buttons
constexpr int button1_pin = A1;
constexpr int button2_pin = A2;
constexpr int button3_pin = A3;

// trimmer
constexpr int trimmer_pin = A0;

// numerical digits for 7-segs
// constexpr int digits[10] = { 0xc0, 0xf9, 0xa4, 0xb0, 0x99, 0x92, 0x82, 0xf8, 0x80, 0x90 };
constexpr int empty_glyph = 0xff;
constexpr int digits[11] = { 0xc0, 0xf9, 0xa4, 0xb0, 0x99, 0x92, 0x82, 0xf8, 0x80, 0x90, 0xff };
constexpr int digit_muxpos[4] = { 0x01, 0x02, 0x04, 0x08 };

constexpr byte letterGlyph[] {
  0b10001000,   // A
  0b10000011,   // b
  0b11000110,   // C
  0b10100001,   // d
  0b10000110,   // E
  0b10001110,   // F
  0b10000010,   // G
  0b10001001,   // H
  0b11111001,   // I
  0b11100001,   // J
  0b10000101,   // K
  0b11000111,   // L
  0b11001000,   // M
  0b10101011,   // n
  0b10100011,   // o
  0b10001100,   // P
  0b10011000,   // q
  0b10101111,   // r
  0b10010010,   // S
  0b10000111,   // t
  0b11000001,   // U
  0b11100011,   // v
  0b10000001,   // W
  0b10110110,   // X
  0b10010001,   // Y
  0b10100100,   // Z
};
constexpr byte emptyGlyph = 0b11111111;
byte LED[4] = {emptyGlyph,emptyGlyph,emptyGlyph,emptyGlyph};

#endif

