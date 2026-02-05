#include <Arduino_LED_Matrix.h>
#include <stdint.h>

Arduino_LED_Matrix matrix;

static const unsigned long IDLE_TIMEOUT_MS = 2000;
static const unsigned long DECAY_RATE_MS   = 40;
static const uint8_t ON = 7;

// CB Text
uint8_t frame_cb[] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 7, 7, 7, 7, 0, 0, 0, 7, 7, 7, 7, 0,
  0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0,
  0, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 0,
  0, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 0,
  0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0,
  0, 7, 7, 7, 7, 0, 0, 0, 7, 7, 7, 7, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// State
static unsigned long lastAudioTime = 0;
static unsigned long lastDecayTime = 0;
static bool isIdle = true;
static bool wasIdle = true;
static uint8_t heights[13] = {0};   // 0..8 bars with peak/decay

static inline void clearFrame(uint8_t *f) {
  for (int i = 0; i < 104; i++) f[i] = 0;
}

// x: 0..12, y: 0..7 (0,0 top-left)
static inline void setPx(uint8_t *f, int x, int y, uint8_t v) {
  if (x < 0 || x > 12 || y < 0 || y > 7) return;
  f[y * 13 + x] = v;
}

static void drawSpectrum() {
  uint8_t f[104];
  clearFrame(f);

  // 13 columns, bars rise from bottom (row 7)
  for (int x = 0; x < 13; x++) {
    int h = heights[x];
    if (h > 8) h = 8;
    for (int k = 0; k < h; k++) {
      int y = 7 - k;
      setPx(f, x, y, ON);
    }
  }
  matrix.draw(f);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Serial1.begin(115200);
}

void loop() {
  unsigned long now = millis();
  bool redraw = false;

  // Receive: 'V' + 13 bytes (0..8)
  if (Serial1.available() >= 14) {
    if (Serial1.read() == 'V') {
      uint8_t b[13];
      Serial1.readBytes(b, 13);

      int total = 0;
      for (int i = 0; i < 13; i++) {
        if (b[i] > 8) b[i] = 8;
        total += b[i];
        if (b[i] > heights[i]) heights[i] = b[i]; // instant rise
      }

      if (total > 5) {               // audio active threshold
        lastAudioTime = now;
        isIdle = false;
        redraw = true;
      }
    }
  }

  // Decay
  if (now - lastDecayTime >= DECAY_RATE_MS) {
    lastDecayTime = now;
    for (int i = 0; i < 13; i++) {
      if (heights[i] > 0) {
        heights[i]--;
        redraw = true;
      }
    }
  }

  // Idle transition
  if (!isIdle && (now - lastAudioTime > IDLE_TIMEOUT_MS)) {
    isIdle = true;
    for (int i = 0; i < 13; i++) heights[i] = 0;
  }

  // Render
  if (isIdle) {
    if (!wasIdle) {
      matrix.clear(); 
      wasIdle = true;
    }
    matrix.draw(frame_cb);
  } else {
    wasIdle = false;
    if (redraw) drawSpectrum();
  }
}

