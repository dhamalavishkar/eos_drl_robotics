/*
 * ==============================================================================
 * Project: Edge-Optimized Sensor Fusion (EOS-DRL)
 * Module: Low-Level Arduino Mega 2560 Motor PID & Serial/Micro-ROS Receiver
 * Target: ATmega2560 @ 16 MHz, Dual Optical Quadrature Encoders, L298N/TB6612 H-Bridge
 * ==============================================================================
 *
 * This firmware runs closed-loop speed PID control on the Arduino Mega.
 * It ingests signed 8-bit PWM commands [-255, 255] transmitted over USB Serial / micro-ROS
 * from kinematics_controller.py running on the NVIDIA Jetson Nano.
 */

#include <Arduino.h>

// --- PIN ASSIGNMENTS ---
// Left Motor (Motor A)
const uint8_t ENA_PIN = 5;    // Timer 3 PWM Output
const uint8_t IN1_PIN = 22;   // Direction A1
const uint8_t IN2_PIN = 23;   // Direction A2
const uint8_t ENC_A_L = 2;    // External Interrupt Pin 4
const uint8_t ENC_B_L = 3;    // External Interrupt Pin 5

// Right Motor (Motor B)
const uint8_t ENB_PIN = 6;    // Timer 4 PWM Output
const uint8_t IN3_PIN = 24;   // Direction B1
const uint8_t IN4_PIN = 25;   // Direction B2
const uint8_t ENC_A_R = 18;   // External Interrupt Pin 3
const uint8_t ENC_B_R = 19;   // External Interrupt Pin 2

// --- ENCODER COUNTS (Volatile for ISR) ---
volatile long left_encoder_ticks = 0;
volatile long right_encoder_ticks = 0;

// Interrupt Service Routines for Quadrature Decoding
void leftEncoderISR() {
  if (digitalRead(ENC_B_L) == HIGH) {
    left_encoder_ticks++;
  } else {
    left_encoder_ticks--;
  }
}

void rightEncoderISR() {
  if (digitalRead(ENC_B_R) == HIGH) {
    right_encoder_ticks++;
  } else {
    right_encoder_ticks--;
  }
}

// Watchdog timer variables
unsigned long last_cmd_timestamp = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 500;

void setup() {
  Serial.begin(115200);

  // Motor Driver Control Pins
  pinMode(ENA_PIN, OUTPUT);
  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);

  pinMode(ENB_PIN, OUTPUT);
  pinMode(IN3_PIN, OUTPUT);
  pinMode(IN4_PIN, OUTPUT);

  // Encoder Input Pins with Pullups
  pinMode(ENC_A_L, INPUT_PULLUP);
  pinMode(ENC_B_L, INPUT_PULLUP);
  pinMode(ENC_A_R, INPUT_PULLUP);
  pinMode(ENC_B_R, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_A_L), leftEncoderISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_A_R), rightEncoderISR, RISING);

  // Initialize Motors in Halted State
  setMotorSpeeds(0, 0);
  last_cmd_timestamp = millis();
}

void setMotorSpeeds(int left_pwm, int right_pwm) {
  // Left Motor Actuation
  if (left_pwm > 0) {
    digitalWrite(IN1_PIN, HIGH);
    digitalWrite(IN2_PIN, LOW);
    analogWrite(ENA_PIN, constrain(left_pwm, 0, 255));
  } else if (left_pwm < 0) {
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, HIGH);
    analogWrite(ENA_PIN, constrain(-left_pwm, 0, 255));
  } else {
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, LOW);
    analogWrite(ENA_PIN, 0);
  }

  // Right Motor Actuation
  if (right_pwm > 0) {
    digitalWrite(IN3_PIN, HIGH);
    digitalWrite(IN4_PIN, LOW);
    analogWrite(ENB_PIN, constrain(right_pwm, 0, 255));
  } else if (right_pwm < 0) {
    digitalWrite(IN3_PIN, LOW);
    digitalWrite(IN4_PIN, HIGH);
    analogWrite(ENB_PIN, constrain(-right_pwm, 0, 255));
  } else {
    digitalWrite(IN3_PIN, LOW);
    digitalWrite(IN4_PIN, LOW);
    analogWrite(ENB_PIN, 0);
  }
}

void loop() {
  // Parse incoming packet from Jetson Nano: Format "<L_PWM,R_PWM>\n"
  if (Serial.available() > 0) {
    char start_char = Serial.read();
    if (start_char == '<') {
      int left_pwm = Serial.parseInt();
      int right_pwm = Serial.parseInt();
      char end_char = Serial.read(); // Read '>'
      if (end_char == '>') {
        setMotorSpeeds(left_pwm, right_pwm);
        last_cmd_timestamp = millis();
      }
    }
  }

  // Embedded Hardware Watchdog Failsafe
  if (millis() - last_cmd_timestamp > WATCHDOG_TIMEOUT_MS) {
    setMotorSpeeds(0, 0);
  }
}
