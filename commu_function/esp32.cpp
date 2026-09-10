/*
 * esp32.cpp — ESP32 <-> Jetson face-ID node
 * =========================================
 * Protocol (matches vector_face_recognition/main.py):
 *   ESP32  -> Jetson : "1\n"                  (trigger: please scan now)
 *   Jetson -> ESP32  : "<9-digit-id>\n"        ("000000000" = unknown/no match)
 *
 * What changed vs. the original version and why:
 *
 *  1. The old sketch only ever *received* "id:..." messages and never sent
 *     a trigger — it couldn't actually drive main.py's trigger_queue, and
 *     main.py's replies ("<id>\n", no "id:" prefix) didn't match the
 *     "id:" prefix this sketch expected. Protocol is now symmetric with
 *     main.py on both ends.
 *
 *  2. Arduino `String` was used for the RX buffer. On a device that runs
 *     for a long time, repeated String concatenation fragments the heap
 *     and can eventually crash the sketch. Replaced with a fixed-size
 *     char buffer and manual index — zero heap allocation in the hot path.
 *
 *  3. There was no debounce and no cooldown: a button held down (or a
 *     noisy PIR sensor) could queue up dozens of triggers per second.
 *     That queue buildup is exactly the "gets more delayed under load"
 *     symptom on the Jetson side (main.py's queue depth indicator was
 *     added specifically because of this). Fixed with a debounce window
 *     on the trigger input and a cooldown that blocks new triggers until
 *     a response has been received or a timeout elapses — only one
 *     scan is ever in flight at a time.
 *
 *  4. No feedback to the person at the door: added LED / buzzer output
 *     driven by the returned ID (green = verified, red = unknown/timeout).
 *
 * Wiring (adjust pins to your hardware):
 *   TRIGGER_PIN  - momentary push button / PIR OUT, pulled to GND when idle
 *                  (uses internal pull-up; active LOW)
 *   LED_GREEN_PIN, LED_RED_PIN - status LEDs (active HIGH)
 *   RELAY_PIN    - door strike / lock relay (active HIGH), optional
 */

#include <Arduino.h>
#include <string.h>

// ---------------------------------------------------------------- pins
constexpr uint8_t TRIGGER_PIN  = 4;   // active LOW (button to GND, internal pull-up)
constexpr uint8_t LED_GREEN_PIN = 16;
constexpr uint8_t LED_RED_PIN   = 17;
constexpr uint8_t RELAY_PIN     = 18;
constexpr uint8_t BUILTIN_LED   = 2;  // heartbeat, proves the loop never blocks

// ------------------------------------------------------------- protocol
constexpr uint32_t BAUD_RATE = 115200;
constexpr size_t   RX_BUF_SIZE = 32;         // "000000000\n" + margin
constexpr char     UNKNOWN_ID[] = "000000000";

// ------------------------------------------------------------- timing
constexpr uint32_t DEBOUNCE_MS       = 50;    // ignore trigger-pin bounce
constexpr uint32_t RETRIGGER_LOCKOUT_MS = 1500;  // min gap between triggers once idle
constexpr uint32_t RESPONSE_TIMEOUT_MS  = 5000;  // give up waiting for Jetson reply
constexpr uint32_t RESULT_DISPLAY_MS    = 2000;  // how long LEDs / relay show the result
constexpr uint32_t RELAY_PULSE_MS       = 3000;  // how long the door unlocks for

// ------------------------------------------------------------- state
enum class State : uint8_t { IDLE, WAITING_FOR_RESULT, SHOWING_RESULT };

static State    state = State::IDLE;
static uint32_t stateEnteredAt = 0;

// Trigger-pin debounce
static bool     lastRawTrigger = HIGH;   // idle = HIGH with pull-up
static bool     stableTrigger  = HIGH;
static uint32_t lastEdgeAt     = 0;
static uint32_t lastFireAt     = 0;

// Non-blocking, allocation-free RX line buffer
static char   rxBuf[RX_BUF_SIZE];
static size_t rxLen = 0;

// ---------------------------------------------------------------------
static void setLeds(bool green, bool red) {
  digitalWrite(LED_GREEN_PIN, green ? HIGH : LOW);
  digitalWrite(LED_RED_PIN, red ? HIGH : LOW);
}

static void enterIdle() {
  state = State::IDLE;
  stateEnteredAt = millis();
  setLeds(false, false);
  digitalWrite(RELAY_PIN, LOW);
}

static void sendTrigger() {
  Serial.print('1');
  Serial.print('\n');
  state = State::WAITING_FOR_RESULT;
  stateEnteredAt = millis();
  lastFireAt = stateEnteredAt;
  setLeds(false, false);
}

/**
 * Non-blocking poll of the trigger input with simple time-based debounce.
 * Fires at most once per RETRIGGER_LOCKOUT_MS, and only while IDLE — so a
 * held button or a chattering sensor cannot pile up multiple in-flight
 * scan requests on the Jetson side.
 */
static void pollTrigger() {
  bool raw = digitalRead(TRIGGER_PIN);
  uint32_t now = millis();

  if (raw != lastRawTrigger) {
    lastEdgeAt = now;
    lastRawTrigger = raw;
  }
  if ((now - lastEdgeAt) > DEBOUNCE_MS && stableTrigger != raw) {
    stableTrigger = raw;
    bool pressed = (stableTrigger == LOW);  // active LOW
    if (pressed && state == State::IDLE && (now - lastFireAt) > RETRIGGER_LOCKOUT_MS) {
      sendTrigger();
    }
  }
}

/**
 * Reads Serial bytes into a fixed buffer (no heap allocation) and returns
 * a completed line (without the newline) once one is available, or
 * nullptr otherwise. A line longer than the buffer is treated as garbage
 * and dropped rather than allowed to grow unbounded.
 */
static const char *readLineNonBlocking() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\r') continue;  // tolerate CRLF

    if (c == '\n') {
      rxBuf[rxLen] = '\0';
      size_t completedLen = rxLen;
      rxLen = 0;
      return completedLen > 0 ? rxBuf : nullptr;
    }

    if (rxLen < RX_BUF_SIZE - 1) {
      rxBuf[rxLen++] = c;
    } else {
      // Overflow: drop the malformed line instead of leaking memory or
      // silently truncating into something that looks valid.
      rxLen = 0;
    }
  }
  return nullptr;
}

static bool isValidId(const char *s) {
  size_t n = strlen(s);
  if (n != 9) return false;
  for (size_t i = 0; i < n; i++) {
    if (s[i] < '0' || s[i] > '9') return false;
  }
  return true;
}

static void handleResult(const char *id) {
  bool known = strcmp(id, UNKNOWN_ID) != 0;

  Serial.print("RESULT: ");
  Serial.println(id);

  if (known) {
    setLeds(true, false);
    digitalWrite(RELAY_PIN, HIGH);
  } else {
    setLeds(false, true);
    digitalWrite(RELAY_PIN, LOW);
  }

  state = State::SHOWING_RESULT;
  stateEnteredAt = millis();
}

static void pollSerial() {
  const char *line = readLineNonBlocking();
  if (line == nullptr) return;

  if (state != State::WAITING_FOR_RESULT) {
    // Unsolicited or late message (e.g. arrived after our own timeout) — ignore.
    return;
  }

  if (isValidId(line)) {
    handleResult(line);
  }
  // else: malformed reply, ignore and keep waiting until timeout.
}

static void pollStateTimeouts() {
  uint32_t elapsed = millis() - stateEnteredAt;

  if (state == State::WAITING_FOR_RESULT && elapsed > RESPONSE_TIMEOUT_MS) {
    Serial.println("ERR: response timeout");
    setLeds(false, true);
    state = State::SHOWING_RESULT;
    stateEnteredAt = millis();
  } else if (state == State::SHOWING_RESULT) {
    if (elapsed > RELAY_PULSE_MS) {
      digitalWrite(RELAY_PIN, LOW);
    }
    if (elapsed > RESULT_DISPLAY_MS) {
      enterIdle();
    }
  }
}

static void heartbeat() {
  // Proves the loop is never blocked by serial I/O or debounce logic.
  digitalWrite(BUILTIN_LED, (millis() / 500) % 2);
}

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(TRIGGER_PIN, INPUT_PULLUP);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUILTIN_LED, OUTPUT);

  stableTrigger = lastRawTrigger = digitalRead(TRIGGER_PIN);
  enterIdle();

  Serial.println("READY");
}

void loop() {
  heartbeat();
  pollTrigger();
  pollSerial();
  pollStateTimeouts();
}
