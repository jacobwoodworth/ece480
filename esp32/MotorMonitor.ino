/*
  ============================================================================
   Smart Motor Health Monitor  -  ESP32 (ESP-WROOM-32 dev board)
   ECE 480 Team 15  |  MSU E-RESILIENCY Lab
  ============================================================================
   Hardware
     GPIO34 (ADC1_CH6)  <- SCT-013-005 current sensor (5 A : 1 V), biased to 1.65 V
     GPIO35 (ADC1_CH7)  <- ZMPT101B voltage sensor module,        biased to ~1.65 V
     Both sensors' outputs must sit on a ~1.65 V DC level (ADC counts ~2048).

   What it does
     * Samples both channels at 2 kS/s (hardware timer + dedicated task, core 1)
     * Computes RMS current/voltage, real power, power factor, crest factor,
       harmonics and mechanical sidebands once per 166 ms window
     * Learns a "healthy" baseline in the first 2 minutes of running
     * Flags Warning / Fault with persistence (no false alarms from a blip)
     * Optional teach mode: record a fault condition once, and the monitor
       names it when it sees it again (tiny nearest-centroid classifier)
     * Creates its own WiFi network. Phone: join the network below, then open
       http://192.168.4.1

   Arduino IDE setup
     Board: "ESP32 Dev Module"  (Espressif "esp32" core 2.0.x or 3.x)
     No extra libraries needed.  Serial monitor: 115200 baud (prints a CSV log).

   SAFETY: the current sensor clamps on ONE conductor only; all mains wiring
   must be inside a fused enclosure and inspected by lab staff before power-up.
  ============================================================================
*/
#include <WiFi.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <WebServer.h>
#include <Preferences.h>
#include "esp_timer.h"

// ---------------------------- user settings ---------------------------------
static const int   PIN_I   = 34;                 // SCT-013-005 (current)
static const int   PIN_V   = 35;                 // ZMPT101B   (voltage)
static const char* AP_SSID = "MotorMonitor-T15"; // WiFi network the ESP32 creates
static const char* AP_PASS = "motor-t15";        // at least 8 characters
// ----------------------------------------------------------------------------

// =============================================================================
//  DETECTION ENGINE  (portable C++; unit-tested off-target)
// =============================================================================
// =============================================================================
//  MotorCore  -  portable DSP + detection engine (no Arduino dependencies)
//
//  Pipeline (per channel, 2 kS/s):
//    raw ADC -> DC removal -> 2nd-order Butterworth low-pass (200 Hz)
//    -> 333-sample windows (~10 mains cycles) -> RMS, real power, PF, crest,
//       harmonics + sidebands (Goertzel) -> 6 windows = 1 s "epoch"
//    -> baseline learning (mean/sigma) -> z-scores -> anomaly score
//    -> % deviation rule + score rule -> persistence state machine
//    -> optional "teach mode" nearest-centroid classifier for fault types
// =============================================================================
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

namespace mm {

// ----------------------------- Tunable settings -----------------------------
constexpr float FS              = 2000.0f;  // samples per second, per channel
constexpr int   WIN             = 333;      // samples per window (166.5 ms = ~10 mains cycles)
constexpr int   WIN_PER_EPOCH   = 6;        // windows per decision epoch (~1 s)
constexpr int   WARMUP_SAMPLES  = 4000;     // discard first 2 s (filters / DC settle)
constexpr float DC_ALPHA        = 0.001f;   // DC tracker (~0.3 Hz high-pass)
constexpr float LP_CUTOFF_HZ    = 200.0f;   // low-pass for RMS / power
constexpr float ZC_HYST         = 30.0f;    // zero-crossing hysteresis (ADC counts)

constexpr float IDLE_A          = 0.10f;    // below this the motor is "off"
constexpr int   STARTUP_EPOCHS  = 4;        // ignore inrush after start
constexpr int   LEARN_EPOCHS    = 120;      // baseline learning time (s)
constexpr int   RECORD_EPOCHS   = 30;       // teach-mode recording time (s)

constexpr float WARN_PCT        = 10.0f;    // current deviation -> Warning
constexpr float FAULT_PCT       = 25.0f;    // current deviation -> Fault
constexpr float SCORE_WARN      = 3.0f;     // multi-feature anomaly score -> Warning
constexpr float SCORE_FAULT     = 6.0f;     // multi-feature anomaly score -> Fault
constexpr int   CONFIRM_EPOCHS  = 3;        // persistence before declaring Warning/Fault (s)
constexpr int   WARN_HOLD_EPOCHS= 30;       // Warning held this long -> Fault (s)
constexpr int   RECOVER_EPOCHS  = 10;       // time back in band before returning to Normal (s)
constexpr float SUPPLY_BAND     = 0.07f;    // |Vrms - baseline| / baseline flagged as supply event

constexpr int   NF       = 7;               // number of features
constexpr int   NCLASS   = 4;               // learnable fault patterns
constexpr int   HIST_N   = 600;             // history length (1 record per second)
constexpr int   EVT_N    = 16;              // event log length
constexpr int   EVT_MSG_LEN = 96;           // event / status message length

enum Feat : uint8_t { F_IRMS = 0, F_PWR, F_PF, F_CREST, F_THD, F_SIDE, F_RIPPLE };
enum State : uint8_t { ST_IDLE = 0, ST_STARTING, ST_LEARNING, ST_NORMAL, ST_WARNING, ST_FAULT };

static const char* const FEAT_NAME[NF] = {
  "Current", "Real power", "Power factor", "Waveform shape", "Harmonic distortion",
  "Mechanical sidebands", "Current ripple"};
static const char* const CLASS_NAME[NCLASS] = {
  "Overload", "Airflow blocked", "Misalignment", "Other fault"};
static const char* const STATE_NAME[6] = {
  "IDLE", "STARTING", "LEARNING", "NORMAL", "WARNING", "FAULT"};

// Minimum sigma per feature so a very steady baseline does not make tiny
// changes look huge:  sigma_eff = max(sigma, FLOOR_REL*|mean|, FLOOR_ABS)
static const float FLOOR_REL[NF] = {0.02f, 0.03f, 0.015f, 0.02f, 0.0f,  0.0f,  0.0f};
static const float FLOOR_ABS[NF] = {0.01f, 0.50f, 0.010f, 0.02f, 0.010f, 0.004f, 0.004f};

// Default scale factors (ADC counts -> amps / volts).  Replace by calibrating
// against a plug-in power meter (web page -> Calibration).
constexpr float DEFAULT_I_SCALE = (3.3f / 4095.0f) * 5.0f;  // SCT-013-005: 5 A = 1 V
constexpr float DEFAULT_V_SCALE = 0.12f;                    // V per count (ZMPT101B, pot dependent)

// ------------------------------ Data structures -----------------------------
struct Cal { float iScale; float vScale; uint8_t calibrated; uint8_t pad[3]; };
struct Baseline { uint8_t valid; uint8_t pad[3]; float mu[NF]; float sig[NF]; float vMean; uint16_t n; uint16_t pad2; };
struct ClassRec { uint8_t valid; uint8_t pad; uint16_t n; float c[NF]; };
struct Persist { uint32_t magic; Cal cal; Baseline base; ClassRec cls[NCLASS]; };
constexpr uint32_t PERSIST_MAGIC = 0x4D4D5431;  // "MMT1"

struct Status {
  uint32_t t;                       // seconds since start (epoch counter)
  uint8_t  state;
  float    irms, vrms, pwr, pf, crest, thd, side, ripple, freq;
  float    devPct;                  // current deviation from baseline (%)
  float    score;                   // multi-feature anomaly score
  float    health;                  // 0..100, -1 if not monitoring
  float    baseI, baseV;
  uint8_t  baseValid, calibrated, supply;
  int8_t   label;                   // -1 none, 0..NCLASS-1 learned match, -2 unknown pattern
  int8_t   recSlot; uint16_t recDone;
  uint16_t learnDone;
  uint8_t  confirm;                 // epochs a deviation has been seen (0..CONFIRM_EPOCHS)
  float    irmsCounts, vrmsCounts;  // RMS in ADC counts (for calibration)
  float    biasI, biasV;            // mean raw ADC counts
  int32_t  ppI, ppV;                // peak-to-peak raw counts
  uint8_t  clipI, clipV;
  float    fsMeasured;
  uint32_t overruns;
  char     msg[96];
};
struct HistRec { uint32_t t; float x[NF]; float v; float score; uint8_t st; };
struct Evt { uint32_t t; uint8_t from, to; char msg[EVT_MSG_LEN]; };

// ------------------------------- DSP helpers --------------------------------
struct Biquad {
  float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
  void design(float fc, float fs) {            // 2nd-order Butterworth low-pass
    const float k = tanf(3.14159265f * fc / fs);
    const float s2 = 1.41421356f;
    const float a0 = 1.0f + s2 * k + k * k;
    b0 = k * k / a0; b1 = 2.0f * b0; b2 = b0;
    a1 = 2.0f * (k * k - 1.0f) / a0;
    a2 = (1.0f - s2 * k + k * k) / a0;
    z1 = z2 = 0;
  }
  inline float run(float x) {
    float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
};

struct Goertzel {
  float coeff = 0, s1 = 0, s2 = 0;
  void init(int k) { coeff = 2.0f * cosf(2.0f * 3.14159265f * (float)k / (float)WIN); s1 = s2 = 0; }
  inline void feed(float x) { float s = x + coeff * s1 - s2; s2 = s1; s1 = s; }
  float power() const { return s1 * s1 + s2 * s2 - coeff * s1 * s2; }
  void reset() { s1 = s2 = 0; }
};

// Goertzel bins (bin spacing = FS/WIN = 6.006 Hz, so 10 = 60 Hz mains)
//   0..2  : 24, 30, 36 Hz   (lower sidebands, shaft-rate region)
//   3     : 60 Hz           (fundamental)
//   4..6  : 84, 90, 96 Hz   (upper sidebands)
//   7     : 120 Hz, 8: 180 Hz, 9: 300 Hz  (2nd, 3rd, 5th harmonics)
static const int GBINS[10] = {4, 5, 6, 10, 14, 15, 16, 20, 30, 50};

// --------------------------------- Monitor ----------------------------------
class Monitor {
 public:
  Persist pst;
  bool persistDirty = false;
  Status st;
  HistRec hist[HIST_N];
  uint16_t histHead = 0, histCount = 0;
  Evt evt[EVT_N];
  uint8_t evtHead = 0, evtCount = 0;

  void begin() {
    memset(&st, 0, sizeof(st));
    memset(&pst, 0, sizeof(pst));
    pst.magic = PERSIST_MAGIC;
    pst.cal.iScale = DEFAULT_I_SCALE;
    pst.cal.vScale = DEFAULT_V_SCALE;
    st.label = -1; st.recSlot = -1; st.health = -1;
    lpI.design(LP_CUTOFF_HZ, FS);
    lpV.design(LP_CUTOFF_HZ, FS);
    for (int i = 0; i < 10; i++) gz[i].init(GBINS[i]);
    resetWindow(); resetEpoch();
    setState(ST_IDLE, "boot", false);
  }

  // Load persisted data (returns false if invalid)
  bool load(const Persist& p) {
    if (p.magic != PERSIST_MAGIC) return false;
    if (!(p.cal.iScale > 0 && p.cal.vScale > 0)) return false;
    pst = p;
    updateStatusFlags();
    return true;
  }

  // ---- called once per sample pair (from the sampler task) -----------------
  // Returns true when a full 1-second epoch is ready -> call processEpoch().
  bool pushSample(uint16_t rawI, uint16_t rawV) {
    const float xi = (float)rawI, xv = (float)rawV;
    if (!dcInit) { dcI = xi; dcV = xv; dcInit = true; }
    dcI += DC_ALPHA * (xi - dcI);
    dcV += DC_ALPHA * (xv - dcV);
    const float ci = xi - dcI, cv = xv - dcV;

    // raw diagnostics
    if (rawI < rminI) rminI = rawI;  if (rawI > rmaxI) rmaxI = rawI;
    if (rawV < rminV) rminV = rawV;  if (rawV > rmaxV) rmaxV = rawV;
    sumRawI += xi; sumRawV += xv; nRaw++;

    const float fi = lpI.run(ci), fv = lpV.run(cv);
    if (warm < WARMUP_SAMPLES) { warm++; prevV = fv; return false; }

    for (int i = 0; i < 10; i++) gz[i].feed(ci);        // harmonics from unfiltered current

    sumI2 += fi * fi; sumV2 += fv * fv; sumVI += fv * fi;
    const float a = fabsf(fi); if (a > pkI) pkI = a;

    // mains frequency from rising zero crossings of the voltage channel
    if (zcArmed && fv >= 0.0f && prevV < 0.0f) {
      const float frac = -prevV / (fv - prevV);
      const float pos = (float)(nWin - 1) + frac;
      if (zcCount == 0) zcFirst = pos;
      zcLast = pos; zcCount++; zcArmed = false;
    }
    if (fv < -ZC_HYST) zcArmed = true;
    prevV = fv;

    if (++nWin >= WIN) {
      finishWindow();
      if (++wCount >= WIN_PER_EPOCH) { finishEpoch(); return true; }
    }
    return false;
  }

  void setTiming(float fsMeasured, uint32_t overruns) { st.fsMeasured = fsMeasured; st.overruns = overruns; }

  // ---- commands (call with the shared lock held) ----------------------------
  void cmdRelearn() { invalidateBaseline(false); }
  void cmdRecord(int slot) {
    if (slot < 0 || slot >= NCLASS || !pst.base.valid) return;
    st.recSlot = (int8_t)slot; st.recDone = 0; recN = 0;
    for (int f = 0; f < NF; f++) recAcc[f] = 0;
  }
  void cmdStopRecord() { st.recSlot = -1; st.recDone = 0; recN = 0; }
  void cmdClearClass(int slot) {
    if (slot < 0 || slot >= NCLASS) return;
    memset(&pst.cls[slot], 0, sizeof(ClassRec)); persistDirty = true;
  }
  // Calibrate scale factors using readings from a plug-in power meter.
  bool cmdCalibrate(float refA, float refV) {
    bool ch = false;
    if (refA > 0.05f && st.irmsCounts > 5.0f) { pst.cal.iScale = refA / st.irmsCounts; ch = true; }
    if (refV > 20.0f && st.vrmsCounts > 5.0f) { pst.cal.vScale = refV / st.vrmsCounts; ch = true; }
    if (ch) { pst.cal.calibrated = 1; invalidateBaseline(true); persistDirty = true; }
    return ch;
  }

  // ---- history / events access ----------------------------------------------
  const HistRec& histAt(int i) const { return hist[(histHead + HIST_N - histCount + i) % HIST_N]; }
  const Evt& evtAt(int i) const { return evt[(evtHead + EVT_N - evtCount + i) % EVT_N]; }

  // ---- once-per-second decision logic ----------------------------------------
  void processEpoch() {
    epochN++;
    const float* x = epX;
    st.t = epochN;
    st.irms = x[F_IRMS]; st.pwr = x[F_PWR]; st.pf = x[F_PF]; st.crest = x[F_CREST];
    st.thd = x[F_THD]; st.side = x[F_SIDE]; st.ripple = x[F_RIPPLE];
    st.vrms = epV; st.freq = epF;
    st.irmsCounts = epIc; st.vrmsCounts = epVc;
    st.biasI = epBiasI; st.biasV = epBiasV; st.ppI = epPpI; st.ppV = epPpV;
    st.clipI = epClipI; st.clipV = epClipV;
    updateStatusFlags();

    float z[NF]; for (int f = 0; f < NF; f++) z[f] = 0;
    float score = 0, dev = 0;
    int sev = 0;
    bool monitoring = false;

    const bool running = x[F_IRMS] >= IDLE_A;
    if (!running) {
      if (++idleCnt >= 2 && st.state != ST_IDLE) setState(ST_IDLE, "No motor current - motor off or stopped", true);
      c1 = c2 = cOk = 0;
    } else {
      idleCnt = 0;
      if (st.state == ST_IDLE) {
        setState(ST_STARTING, "Motor started - ignoring start-up surge", true);
        startGuard = STARTUP_EPOCHS;
      } else if (st.state == ST_STARTING) {
        if (--startGuard <= 0) {
          if (pst.base.valid) { setState(ST_NORMAL, "Monitoring against learned baseline", true); }
          else { setState(ST_LEARNING, "Learning baseline - keep motor at normal load", true); }
          c1 = c2 = cOk = 0; warnAge = 0;
        }
      } else if (st.state == ST_LEARNING) {
        learnStep(x);
      } else {
        monitoring = true;
        // ---- z-scores and anomaly score ------------------------------------
        float acc = 0;
        for (int f = 0; f < NF; f++) {
          float zz = (x[f] - pst.base.mu[f]) / pst.base.sig[f];
          if (zz > 20) zz = 20; if (zz < -20) zz = -20;
          z[f] = zz; acc += zz * zz;
        }
        score = sqrtf(acc / (float)NF);
        const float bi = pst.base.mu[F_IRMS];
        dev = bi > 1e-3f ? 100.0f * (x[F_IRMS] - bi) / bi : 0.0f;
        const float ad = fabsf(dev);
        const int sevPct = ad >= FAULT_PCT ? 2 : (ad >= WARN_PCT ? 1 : 0);
        const int sevScore = score >= SCORE_FAULT ? 2 : (score >= SCORE_WARN ? 1 : 0);
        sev = sevPct > sevScore ? sevPct : sevScore;

        if (sev >= 1) c1++; else c1 = 0;
        if (sev >= 2) c2++; else c2 = 0;
        if (sev == 0) cOk++; else cOk = 0;

        // ---- classification of the deviation -------------------------------
        int label = -1;
        if (sev >= 1) label = classify(z);
        st.label = (int8_t)label;
        buildCause(z, dev, label, sev);

        // ---- persistence state machine -------------------------------------
        if (st.state == ST_NORMAL) {
          if (c2 >= CONFIRM_EPOCHS) { setState(ST_FAULT, st.msg, true); }
          else if (c1 >= CONFIRM_EPOCHS) { setState(ST_WARNING, st.msg, true); warnAge = 0; }
        } else if (st.state == ST_WARNING) {
          warnAge++;
          if (c2 >= CONFIRM_EPOCHS) { setState(ST_FAULT, st.msg, true); }
          else if (warnAge >= WARN_HOLD_EPOCHS && sev >= 1) {
            setState(ST_FAULT, "Warning condition persisted 30 s", true);
          } else if (cOk >= RECOVER_EPOCHS) { setState(ST_NORMAL, "Back within baseline", true); c1 = c2 = 0; }
        } else if (st.state == ST_FAULT) {
          if (cOk >= RECOVER_EPOCHS) { setState(ST_NORMAL, "Back within baseline", true); c1 = c2 = 0; }
        }
        st.confirm = (uint8_t)(st.state == ST_NORMAL ? (c1 > CONFIRM_EPOCHS ? CONFIRM_EPOCHS : c1) : 0);
        if (sev == 0 && st.state == ST_NORMAL)
          snprintf(st.msg, sizeof(st.msg), "Operating within learned baseline");
        else if (st.state == ST_NORMAL)
          snprintf(st.msg, sizeof(st.msg), "Deviation seen (%d/%d s) - confirming", (int)st.confirm, CONFIRM_EPOCHS);
        else if (sev == 0)
          snprintf(st.msg, sizeof(st.msg), "Recovering - back in band %d/%d s", cOk, RECOVER_EPOCHS);

        // ---- teach mode recording ------------------------------------------
        if (st.recSlot >= 0) {
          for (int f = 0; f < NF; f++) recAcc[f] += x[f];
          recN++; st.recDone = (uint16_t)recN;
          if (recN >= RECORD_EPOCHS) {
            ClassRec& c = pst.cls[st.recSlot];
            c.valid = 1; c.n = (uint16_t)recN;
            for (int f = 0; f < NF; f++) c.c[f] = recAcc[f] / (float)recN;
            persistDirty = true;
            char m[56]; snprintf(m, sizeof(m), "Learned pattern: %s", CLASS_NAME[st.recSlot]);
            pushEvt(st.state, st.state, m);
            st.recSlot = -1; st.recDone = 0; recN = 0;
          }
        }
      }
    }

    if (!monitoring) {
      st.label = -1; st.confirm = 0;
      if (st.state == ST_LEARNING)
        snprintf(st.msg, sizeof(st.msg), "Learning baseline: %d of %d s - keep motor at normal load", (int)st.learnDone, LEARN_EPOCHS);
      else if (st.state == ST_IDLE)
        snprintf(st.msg, sizeof(st.msg), "No motor current detected");
      else if (st.state == ST_STARTING)
        snprintf(st.msg, sizeof(st.msg), "Start-up - ignoring inrush (%d s)", startGuard);
    }
    st.devPct = dev; st.score = score;
    if (monitoring) {
      float r1 = fabsf(dev) / FAULT_PCT, r2 = score / SCORE_FAULT;
      float r = r1 > r2 ? r1 : r2;
      float h = 100.0f * (1.0f - r); if (h < 0) h = 0; if (h > 100) h = 100;
      st.health = h;
    } else st.health = -1;

    // history record
    HistRec& hr = hist[histHead];
    hr.t = epochN; for (int f = 0; f < NF; f++) hr.x[f] = x[f];
    hr.v = epV; hr.score = score; hr.st = st.state;
    histHead = (histHead + 1) % HIST_N; if (histCount < HIST_N) histCount++;
  }

 private:
  // filters & accumulators
  Biquad lpI, lpV; Goertzel gz[10];
  float dcI = 0, dcV = 0; bool dcInit = false; int warm = 0;
  float sumI2 = 0, sumV2 = 0, sumVI = 0, pkI = 0, prevV = 0;
  bool zcArmed = false; float zcFirst = 0, zcLast = 0; int zcCount = 0;
  int nWin = 0, wCount = 0;
  // raw diagnostics (per epoch)
  uint16_t rminI = 4095, rmaxI = 0, rminV = 4095, rmaxV = 0; float sumRawI = 0, sumRawV = 0; int nRaw = 0;
  // epoch accumulators
  float eSum[NF]; float wIrms[WIN_PER_EPOCH]; float eV = 0, eF = 0, eIc = 0, eVc = 0; int eFn = 0;
  // epoch result
  float epX[NF]; float epV = 0, epF = 0, epIc = 0, epVc = 0, epBiasI = 0, epBiasV = 0;
  int32_t epPpI = 0, epPpV = 0; uint8_t epClipI = 0, epClipV = 0;
  // state machine
  uint32_t epochN = 0; int idleCnt = 0, startGuard = 0, c1 = 0, c2 = 0, cOk = 0, warnAge = 0;
  // learning
  int learnN = 0; float lmean[NF], lm2[NF]; float lv = 0;
  int recN = 0; float recAcc[NF];

  void resetWindow() {
    sumI2 = sumV2 = sumVI = pkI = 0; nWin = 0; zcCount = 0; zcArmed = false;
    for (int i = 0; i < 10; i++) gz[i].reset();
  }
  void resetEpoch() {
    for (int f = 0; f < NF; f++) eSum[f] = 0;
    eV = eF = eIc = eVc = 0; eFn = 0; wCount = 0;
    rminI = rminV = 4095; rmaxI = rmaxV = 0; sumRawI = sumRawV = 0; nRaw = 0;
  }

  void finishWindow() {
    const float iS = pst.cal.iScale, vS = pst.cal.vScale;
    const float ic = sqrtf(sumI2 / WIN), vc = sqrtf(sumV2 / WIN);
    const float iA = ic * iS, vV = vc * vS;
    const float P = (sumVI / WIN) * iS * vS;
    float pf = (iA > 1e-4f && vV > 1e-3f) ? P / (iA * vV) : 0.0f;
    if (pf > 1) pf = 1; if (pf < -1) pf = -1;
    const float crest = ic > 1e-3f ? pkI / ic : 0.0f;
    const float fund = gz[3].power();
    float thd = 0, side = 0;
    if (fund > 1e6f) {
      thd = sqrtf((gz[7].power() + gz[8].power() + gz[9].power()) / fund);
      side = sqrtf((gz[0].power() + gz[1].power() + gz[2].power() + gz[4].power() + gz[5].power() + gz[6].power()) / fund);
    }
    if (zcCount >= 5 && zcLast > zcFirst) { eF += (float)(zcCount - 1) * FS / (zcLast - zcFirst); eFn++; }
    eSum[F_IRMS] += iA; eSum[F_PWR] += P; eSum[F_PF] += pf; eSum[F_CREST] += crest;
    eSum[F_THD] += thd; eSum[F_SIDE] += side;
    wIrms[wCount] = iA; eV += vV; eIc += ic; eVc += vc;
    resetWindow();
  }

  void finishEpoch() {
    const float n = (float)WIN_PER_EPOCH;
    for (int f = 0; f < F_RIPPLE; f++) epX[f] = eSum[f] / n;
    float m = epX[F_IRMS], var = 0;
    for (int i = 0; i < WIN_PER_EPOCH; i++) { float d = wIrms[i] - m; var += d * d; }
    epX[F_RIPPLE] = m > 1e-3f ? sqrtf(var / n) / m : 0.0f;
    epV = eV / n; epIc = eIc / n; epVc = eVc / n;
    epF = eFn > 0 ? eF / (float)eFn : 0.0f;
    epBiasI = nRaw ? sumRawI / nRaw : 0; epBiasV = nRaw ? sumRawV / nRaw : 0;
    epPpI = (int32_t)rmaxI - (int32_t)rminI; epPpV = (int32_t)rmaxV - (int32_t)rminV;
    epClipI = (rminI <= 5 || rmaxI >= 4090); epClipV = (rminV <= 5 || rmaxV >= 4090);
    resetEpoch();
  }

  void updateStatusFlags() {
    st.baseValid = pst.base.valid; st.calibrated = pst.cal.calibrated;
    st.baseI = pst.base.valid ? pst.base.mu[F_IRMS] : 0; st.baseV = pst.base.valid ? pst.base.vMean : 0;
    st.supply = (pst.base.valid && pst.base.vMean > 20.0f &&
                 fabsf(st.vrms - pst.base.vMean) / pst.base.vMean > SUPPLY_BAND) ? 1 : 0;
  }

  void invalidateBaseline(bool clearClasses) {
    pst.base.valid = 0; learnN = 0; lv = 0;
    for (int f = 0; f < NF; f++) { lmean[f] = 0; lm2[f] = 0; }
    st.learnDone = 0; st.recSlot = -1; st.recDone = 0;
    if (clearClasses) for (int c = 0; c < NCLASS; c++) memset(&pst.cls[c], 0, sizeof(ClassRec));
    persistDirty = true;
    if (st.state == ST_NORMAL || st.state == ST_WARNING || st.state == ST_FAULT)
      setState(ST_LEARNING, "Relearning baseline", true);
    updateStatusFlags();
  }

  void learnStep(const float* x) {
    learnN++;
    for (int f = 0; f < NF; f++) {
      const float d = x[f] - lmean[f];
      lmean[f] += d / (float)learnN;
      lm2[f] += d * (x[f] - lmean[f]);
    }
    lv += (epV - lv) / (float)learnN;
    st.learnDone = (uint16_t)learnN;
    if (learnN >= LEARN_EPOCHS) {
      for (int f = 0; f < NF; f++) {
        float sg = sqrtf(lm2[f] / (float)(learnN - 1));
        float fl = FLOOR_REL[f] * fabsf(lmean[f]);
        if (fl < FLOOR_ABS[f]) fl = FLOOR_ABS[f];
        if (sg < fl) sg = fl;
        pst.base.mu[f] = lmean[f]; pst.base.sig[f] = sg;
      }
      pst.base.vMean = lv; pst.base.n = (uint16_t)learnN; pst.base.valid = 1;
      persistDirty = true; c1 = c2 = cOk = 0;
      updateStatusFlags();
      setState(ST_NORMAL, "Baseline learned - monitoring", true);
    }
  }

  // nearest learned fault pattern in baseline-normalised (z) space
  int classify(const float* z) const {
    int best = -1; float bestD = 1e30f, bestN = 0;
    for (int c = 0; c < NCLASS; c++) {
      if (!pst.cls[c].valid) continue;
      float d2 = 0, n2 = 0;
      for (int f = 0; f < NF; f++) {
        float zc = (pst.cls[c].c[f] - pst.base.mu[f]) / pst.base.sig[f];
        if (zc > 20) zc = 20; if (zc < -20) zc = -20;
        const float d = z[f] - zc; d2 += d * d; n2 += zc * zc;
      }
      if (d2 < bestD) { bestD = d2; best = c; bestN = n2; }
    }
    if (best < 0) return -2;
    float thr = 0.5f * sqrtf(bestN); if (thr < 2.5f) thr = 2.5f;
    return sqrtf(bestD) <= thr ? best : -2;
  }

  void buildCause(const float* z, float dev, int label, int sev) {
    if (sev == 0) return;
    char tmp[96];
    if (label >= 0) {
      snprintf(tmp, sizeof(tmp), "Matches learned pattern: %s (current %+.0f%%)", CLASS_NAME[label], dev);
    } else {
      int bi = 0; float bz = 0;
      for (int f = 0; f < NF; f++) if (fabsf(z[f]) > bz) { bz = fabsf(z[f]); bi = f; }
      if (fabsf(dev) >= WARN_PCT)
        snprintf(tmp, sizeof(tmp), "Current %s %.0f%% vs baseline", dev > 0 ? "high" : "low", fabsf(dev));
      else
        snprintf(tmp, sizeof(tmp), "%s %s (%.1f sigma)", FEAT_NAME[bi], z[bi] > 0 ? "high" : "low", fabsf(z[bi]));
    }
    if (st.supply) { size_t l = strlen(tmp); snprintf(tmp + l, sizeof(tmp) - l, " [supply voltage off]"); }
    snprintf(st.msg, sizeof(st.msg), "%s", tmp);
  }

  void pushEvt(uint8_t from, uint8_t to, const char* m) {
    Evt& e = evt[evtHead]; e.t = epochN; e.from = from; e.to = to;
    snprintf(e.msg, sizeof(e.msg), "%s", m);
    evtHead = (evtHead + 1) % EVT_N; if (evtCount < EVT_N) evtCount++;
  }

  void setState(uint8_t s, const char* m, bool log) {
    const uint8_t prev = st.state;
    st.state = s;
    if (log && prev != s) pushEvt(prev, s, m);
    if (m != st.msg) snprintf(st.msg, sizeof(st.msg), "%s", m);
  }
};

}  // namespace mm


// ---------------------------- phone dashboard (HTML) ----------------------------
static const char INDEX_HTML[] PROGMEM = R"HTMLUI(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0a0f1c">
<title>Motor Health Monitor</title>
<style>
:root{--bg:#0a0f1c;--card:#111a2e;--card2:#16213a;--line:#24324f;--tx:#e8eef9;--mu:#8a9bb8;
--ok:#2ecc71;--warn:#f5b301;--bad:#ff5a5f;--info:#4cc3ff;--idle:#6b7a93;--sc:var(--idle);
--r:16px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;background:var(--bg);color:var(--tx)}
body{padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom);min-height:100vh;
background:radial-gradient(1200px 500px at 50% -150px,color-mix(in srgb,var(--sc) 18%,transparent),transparent 70%),var(--bg)}
body[data-st="0"]{--sc:var(--idle)}body[data-st="1"],body[data-st="2"]{--sc:var(--info)}
body[data-st="3"]{--sc:var(--ok)}body[data-st="4"]{--sc:var(--warn)}body[data-st="5"]{--sc:var(--bad)}
.wrap{max-width:760px;margin:0 auto;padding:14px 14px 40px}
header{display:flex;align-items:center;gap:12px;padding:6px 2px 14px}
.logo{width:38px;height:38px;border-radius:11px;background:linear-gradient(145deg,#1c2b4d,#101a31);
border:1px solid var(--line);display:grid;place-items:center;flex:none}
header h1{font-size:17px;line-height:1.15;margin:0;font-weight:650;letter-spacing:.2px}
header small{color:var(--mu);font-size:12px}
.live{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:12px;color:var(--mu)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 0 rgba(46,204,113,.6);animation:pulse 2s infinite}
.dot.off{background:var(--bad);animation:none}
@keyframes pulse{70%{box-shadow:0 0 0 8px rgba(46,204,113,0)}100%{box-shadow:0 0 0 0 rgba(46,204,113,0)}}
.card{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:var(--r);
padding:16px;margin-bottom:14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--mu);margin:0 0 12px;font-weight:650}
.hero{display:flex;gap:18px;align-items:center;border-color:color-mix(in srgb,var(--sc) 45%,var(--line))}
.ring{position:relative;width:138px;height:138px;flex:none}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring .bg{stroke:var(--line)}.ring .fg{stroke:var(--sc);transition:stroke-dashoffset .8s ease,stroke .4s}
.ring .c{position:absolute;inset:0;display:grid;place-content:center;text-align:center}
.ring b{font-size:38px;font-weight:700;line-height:1}.ring span{font-size:10.5px;color:var(--mu);letter-spacing:.14em;margin-top:5px}
.hs{min-width:0}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;font-weight:700;font-size:13px;letter-spacing:.06em;
color:var(--sc);background:color-mix(in srgb,var(--sc) 14%,transparent);border:1px solid color-mix(in srgb,var(--sc) 45%,transparent)}
.pill i{width:8px;height:8px;border-radius:50%;background:var(--sc)}
.msg{margin:10px 0 0;font-size:14.5px;line-height:1.4}
.sub{color:var(--mu);font-size:12.5px;margin-top:6px}
.tiles{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px}
@media(min-width:520px){.tiles{grid-template-columns:repeat(3,1fr)}}
.tile{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:14px;padding:13px 14px}
.tile .l{font-size:11.5px;color:var(--mu);letter-spacing:.04em}
.tile .v{font-size:27px;font-weight:680;margin-top:4px;font-variant-numeric:tabular-nums}
.tile .v u{text-decoration:none;font-size:13px;color:var(--mu);font-weight:500;margin-left:3px}
.tile .s{font-size:11.5px;color:var(--mu);margin-top:3px;min-height:15px}
.tile .s.w{color:var(--warn)}.tile .s.b{color:var(--bad)}
.tabs{display:flex;gap:6px;margin-bottom:10px}
.tab{flex:1;padding:8px 0;border-radius:10px;border:1px solid var(--line);background:transparent;color:var(--mu);font:600 13px inherit;font-family:inherit}
.tab.on{background:color-mix(in srgb,var(--info) 16%,transparent);color:var(--tx);border-color:color-mix(in srgb,var(--info) 55%,transparent)}
canvas{width:100%;height:210px;display:block;touch-action:pan-y}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:11.5px;color:var(--mu);margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.row{display:flex;align-items:center;gap:10px;padding:10px 0;border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.row .n{flex:1;min-width:0}.row .n b{font-size:14px;font-weight:620}.row .n div{font-size:12px;color:var(--mu);margin-top:2px}
button.btn{appearance:none;border:1px solid var(--line);background:var(--card2);color:var(--tx);padding:9px 14px;border-radius:11px;
font:600 13px inherit;font-family:inherit;cursor:pointer}
button.btn:active{transform:scale(.97)}
button.pri{background:color-mix(in srgb,var(--info) 22%,var(--card2));border-color:color-mix(in srgb,var(--info) 55%,transparent)}
button.dng{color:var(--bad)}
button:disabled{opacity:.4}
.bar{height:8px;border-radius:99px;background:var(--line);overflow:hidden;margin-top:8px}
.bar i{display:block;height:100%;width:0;background:var(--info);border-radius:99px;transition:width .6s}
.note{font-size:12px;color:var(--mu);line-height:1.5;margin:10px 0 0}
.ev{display:flex;gap:10px;font-size:13px;padding:7px 0;border-top:1px solid var(--line)}
.ev:first-of-type{border-top:0}.ev time{color:var(--mu);font-variant-numeric:tabular-nums;width:54px;flex:none}
.ev i{width:9px;height:9px;border-radius:50%;margin-top:5px;flex:none}
details summary{cursor:pointer;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--mu);font-weight:650;list-style:none}
details summary::before{content:"\25B8  "}details[open] summary::before{content:"\25BE  "}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
td,th{padding:6px 4px;border-top:1px solid var(--line);text-align:left}th{color:var(--mu);font-weight:500}
td.r{text-align:right;font-variant-numeric:tabular-nums}.bad{color:var(--bad)}.good{color:var(--ok)}
.f{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
input{background:var(--bg);border:1px solid var(--line);color:var(--tx);border-radius:10px;padding:9px 10px;width:120px;font:inherit;font-size:14px}
label{font-size:12px;color:var(--mu);display:block;margin-bottom:4px}
.banner{background:color-mix(in srgb,var(--warn) 12%,transparent);border:1px solid color-mix(in srgb,var(--warn) 40%,transparent);
color:var(--warn);padding:9px 12px;border-radius:12px;font-size:12.5px;margin-bottom:14px;display:none}
footer{text-align:center;color:var(--mu);font-size:11.5px;margin-top:18px;line-height:1.6}
a.dl{color:var(--info);text-decoration:none;font-size:13px;font-weight:600}
</style>
</head>
<body data-st="0">
<div class="wrap">
<header>
  <div class="logo"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#4cc3ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h2.5l2-5 4 10 3-8 1.5 3H21"/></svg></div>
  <div><h1>Motor Health Monitor</h1><small>Team 15 &middot; MSU E-RESILIENCY Lab</small></div>
  <div class="live"><span class="dot off" id="dot"></span><span id="conn">connecting</span></div>
</header>

<div class="banner" id="calBanner">Readings are <b>not calibrated</b> yet &mdash; absolute values are approximate. Fault detection still works (it compares against the learned baseline). See <i>Signal check &amp; calibration</i> at the bottom.</div>

<section class="card hero">
  <div class="ring"><svg viewBox="0 0 140 140"><circle class="bg" cx="70" cy="70" r="60" fill="none" stroke-width="11"/>
  <circle class="fg" id="arc" cx="70" cy="70" r="60" fill="none" stroke-width="11" stroke-linecap="round" stroke-dasharray="377" stroke-dashoffset="377"/></svg>
  <div class="c"><b id="hv">--</b><span>HEALTH</span></div></div>
  <div class="hs">
    <div class="pill"><i></i><span id="stn">--</span></div>
    <p class="msg" id="msg">Waiting for data&hellip;</p>
    <div class="sub" id="upt"></div>
  </div>
</section>

<section class="tiles">
  <div class="tile"><div class="l">CURRENT</div><div class="v"><span id="t_i">--</span><u>A</u></div><div class="s" id="s_i"></div></div>
  <div class="tile"><div class="l">VOLTAGE</div><div class="v"><span id="t_v">--</span><u>V</u></div><div class="s" id="s_v"></div></div>
  <div class="tile"><div class="l">REAL POWER</div><div class="v"><span id="t_p">--</span><u>W</u></div><div class="s" id="s_p"></div></div>
  <div class="tile"><div class="l">POWER FACTOR</div><div class="v"><span id="t_pf">--</span></div><div class="s" id="s_pf"></div></div>
  <div class="tile"><div class="l">LINE FREQUENCY</div><div class="v"><span id="t_f">--</span><u>Hz</u></div><div class="s" id="s_f"></div></div>
  <div class="tile"><div class="l">ANOMALY SCORE</div><div class="v"><span id="t_sc">--</span></div><div class="s">warn 3 &middot; fault 6</div></div>
</section>

<section class="card">
  <h2>Trend &middot; last 5 minutes</h2>
  <div class="tabs"><button class="tab on" data-k="i">Current</button><button class="tab" data-k="p">Power</button><button class="tab" data-k="s">Anomaly score</button></div>
  <canvas id="cv"></canvas>
  <div class="legend"><span><i style="background:#2ecc71"></i>Normal</span><span><i style="background:#f5b301"></i>Warning</span><span><i style="background:#ff5a5f"></i>Fault</span><span><i style="background:#4cc3ff"></i>Learning / start</span><span id="lg_b"></span></div>
</section>

<section class="card">
  <h2>Learning &amp; fault patterns</h2>
  <div class="row"><div class="n"><b id="bl_t">Baseline</b><div id="bl_s">Not learned</div><div class="bar"><i id="bl_bar"></i></div></div>
  <button class="btn pri" id="b_relearn">Relearn</button></div>
  <div id="cls"></div>
  <p class="note">The baseline is what &ldquo;healthy&rdquo; looks like for this motor and load: it is learned automatically over the first 2 minutes of running. Run the motor at its normal load and airflow while it learns. <b>Teach a fault pattern</b> by creating that condition (for example partly blocking the duct) and pressing Record; the monitor then names that condition when it sees it again.</p>
</section>

<section class="card">
  <h2>Event log</h2>
  <div id="evs"><div class="sub">No events yet.</div></div>
</section>

<section class="card">
  <details>
    <summary>Signal check &amp; calibration</summary>
    <table>
      <tr><th></th><th class="r">Current (GPIO34)</th><th class="r">Voltage (GPIO35)</th></tr>
      <tr><td>Mean level (ideal ~2048)</td><td class="r" id="r_bi">--</td><td class="r" id="r_bv">--</td></tr>
      <tr><td>Peak-to-peak counts</td><td class="r" id="r_pi">--</td><td class="r" id="r_pv">--</td></tr>
      <tr><td>Clipping</td><td class="r" id="r_ci">--</td><td class="r" id="r_cv">--</td></tr>
      <tr><td>Sample rate</td><td class="r" colspan="2" id="r_fs">--</td></tr>
      <tr><td>Missed samples</td><td class="r" colspan="2" id="r_ov">--</td></tr>
    </table>
    <p class="note">Good signals: mean near 2048, peak-to-peak of at least a few hundred counts while running (voltage: roughly 1000&ndash;3000), and no clipping. If voltage clips, turn the ZMPT101B trimmer down.</p>
    <div class="f">
      <div><label>Meter current (A)</label><input id="c_a" type="number" step="0.01" inputmode="decimal" placeholder="e.g. 1.05"></div>
      <div><label>Meter voltage (V)</label><input id="c_v" type="number" step="0.1" inputmode="decimal" placeholder="e.g. 119.8"></div>
      <div style="align-self:flex-end"><button class="btn pri" id="b_cal">Apply</button></div>
    </div>
    <p class="note">With the motor running, enter the current and voltage shown on a plug-in power meter (leave one blank to skip it). Calibrating restarts baseline learning, so calibrate <b>first</b>.</p>
  </details>
</section>

<section class="card" style="display:flex;justify-content:space-between;align-items:center">
  <div><b style="font-size:14px">Download data</b><div class="sub">Last 10 minutes, one row per second (CSV)</div></div>
  <a class="dl" href="/api/csv" download="motor_log.csv">Download &darr;</a>
</section>

<footer>ECE 480 Team 15 &middot; Smart Motor Health Monitoring and Early-Warning System<br>Connected to the monitor directly &mdash; no internet required</footer>
</div>

<script>
"use strict";
var $=function(s){return document.querySelector(s)};
var STL=["MOTOR OFF","STARTING","LEARNING","NORMAL","WARNING","FAULT"];
var COL=["#6b7a93","#4cc3ff","#4cc3ff","#2ecc71","#f5b301","#ff5a5f"];
var CLS=["Overload","Airflow blocked","Misalignment","Other fault"];
var MAXH=300, H={i:[],p:[],s:[],st:[]}, tab="i", base={i:0,p:0}, lastEp=-1, lastSt=-1, D=null, hover=-1, evT=0;

function jget(u){return fetch(u,{cache:"no-store"}).then(function(r){if(!r.ok)throw 0;return r.json()})}
function cmd(q){return fetch("/api/cmd?"+q,{cache:"no-store"}).then(function(r){return r.json()}).then(function(){poll(true)})}
function fmt(n,d){return (n===undefined||n===null||isNaN(n))?"--":Number(n).toFixed(d)}
function clock(s){var h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return (h?h+":":"")+(h&&m<10?"0":"")+m+":"+(x<10?"0":"")+x}

function online(ok){$("#dot").className="dot"+(ok?"":" off");$("#conn").textContent=ok?"live":"offline"}

function push(d){
  H.i.push(d.i);H.p.push(d.p);H.s.push(d.sc);H.st.push(d.st);
  for(var k in H) if(H[k].length>MAXH) H[k].shift();
}

function render(d){
  D=d; base.i=d.bi||0; base.p=d.bp||0;
  document.body.setAttribute("data-st",d.st);
  $("#stn").textContent=STL[d.st];
  $("#msg").textContent=d.msg;
  $("#upt").textContent="Running "+clock(d.up)+(d.fs?" · "+d.fs.toFixed(0)+" samples/s":"");
  var h=d.h; $("#hv").textContent=h<0?"--":Math.round(h);
  $("#arc").style.strokeDashoffset=377*(1-(h<0?0:h/100));
  $("#t_i").textContent=fmt(d.i,2); $("#t_v").textContent=fmt(d.v,1); $("#t_p").textContent=fmt(d.p,1);
  $("#t_pf").textContent=fmt(d.pf,2); $("#t_f").textContent=d.f>0?fmt(d.f,2):"--"; $("#t_sc").textContent=fmt(d.sc,1);
  var si=$("#s_i"); if(d.bv){si.textContent="baseline "+fmt(d.bi,2)+" A · "+(d.dev>=0?"+":"")+fmt(d.dev,1)+"%"; si.className="s"+(Math.abs(d.dev)>=25?" b":Math.abs(d.dev)>=10?" w":"")}else{si.textContent="no baseline yet";si.className="s"}
  var sv=$("#s_v"); if(d.bv){sv.textContent=d.sup?"supply off baseline ("+fmt(d.bvv,1)+" V)":"baseline "+fmt(d.bvv,1)+" V";sv.className="s"+(d.sup?" w":"")}else sv.textContent="";
  $("#s_p").textContent=d.bv?"baseline "+fmt(d.bp,1)+" W":""; $("#s_pf").textContent=d.bv?"baseline "+fmt(d.bpf,2):"";
  $("#s_f").textContent=d.f>0?"mains":"no voltage signal";
  $("#calBanner").style.display=d.cal?"none":"block";
  // learning
  var bt=$("#bl_t"),bs=$("#bl_s"),bar=$("#bl_bar");
  if(d.bv){bt.textContent="Baseline learned";bs.textContent="Healthy current "+fmt(d.bi,2)+" A at "+fmt(d.bvv,0)+" V";bar.style.width="100%";bar.style.background="var(--ok)"}
  else{bt.textContent="Learning baseline";bs.textContent=d.st===2?(d.ld+" of "+d.lt+" s"):"Waiting for the motor to run";bar.style.width=(d.st===2?100*d.ld/d.lt:0)+"%";bar.style.background="var(--info)"}
  var rec=d.rs>=0, html="";
  for(var c=0;c<4;c++){
    var on=d.cls[c], isRec=(rec&&d.rs===c);
    html+='<div class="row"><div class="n"><b>'+CLS[c]+'</b><div>'+(isRec?("Recording… "+d.rd+" of "+d.rt+" s — keep this condition steady"):(on?"Pattern learned":"Not taught"))+'</div></div>'+
      (isRec?'<button class="btn" data-a="stop">Cancel</button>':
        '<button class="btn'+(on?'':' pri')+'" data-a="rec" data-s="'+c+'"'+((!d.bv||(rec&&!isRec))?' disabled':'')+'>'+(on?'Re-record':'Record')+'</button>'+
        (on?'<button class="btn dng" data-a="clr" data-s="'+c+'">Clear</button>':''))+'</div>';
  }
  var ce=$("#cls"); if(ce.getAttribute("data-h")!==html){ce.innerHTML=html;ce.setAttribute("data-h",html)}
  // signal check
  $("#r_bi").textContent=fmt(d.raw.bi,0);$("#r_bv").textContent=fmt(d.raw.bv,0);
  $("#r_pi").textContent=d.raw.pi;$("#r_pv").textContent=d.raw.pv;
  $("#r_ci").innerHTML=d.raw.ci?'<span class="bad">CLIPPING</span>':'<span class="good">ok</span>';
  $("#r_cv").innerHTML=d.raw.cv?'<span class="bad">CLIPPING</span>':'<span class="good">ok</span>';
  $("#r_fs").textContent=fmt(d.fs,1)+" samples/s per channel"; $("#r_ov").textContent=d.ov;
  if(d.ep!==lastEp){lastEp=d.ep;push(d)}
  if(d.st!==lastSt||Date.now()-evT>8000){lastSt=d.st;loadEvents()}
  draw();
}

function loadEvents(){
  evT=Date.now();
  jget("/api/events").then(function(a){
    if(!a.length){$("#evs").innerHTML='<div class="sub">No events yet.</div>';return}
    var h="";for(var k=a.length-1;k>=0;k--){var e=a[k];h+='<div class="ev"><time>'+clock(e.t)+'</time><i style="background:'+COL[e.s]+'"></i><div>'+esc(e.m)+'</div></div>'}
    $("#evs").innerHTML=h;
  }).catch(function(){});
}
function esc(s){return String(s).replace(/[&<>]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;"}[c]})}

function poll(once){
  jget("/api/status").then(function(d){online(true);render(d)}).catch(function(){online(false)}).then(function(){if(!once)setTimeout(poll,1000)});
}

/* ---------- chart ---------- */
function draw(){
  var cv=$("#cv"),dpr=window.devicePixelRatio||1,W=cv.clientWidth,Ht=cv.clientHeight; if(!W)return;
  if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(Ht*dpr)){cv.width=Math.round(W*dpr);cv.height=Math.round(Ht*dpr)}
  var c=cv.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,W,Ht);
  var y=H[tab],n=y.length,L=40,R=8,T=8,B=26,pw=W-L-R,ph=Ht-T-B;
  var mut="#8a9bb8",grid="#24324f",unit=tab==="i"?"A":tab==="p"?"W":"";
  var b=tab==="i"?base.i:tab==="p"?base.p:0;
  var lo=1e9,hi=-1e9;for(var k=0;k<n;k++){if(y[k]<lo)lo=y[k];if(y[k]>hi)hi=y[k]}
  if(!n){lo=0;hi=1}
  if(tab==="s"){lo=0;hi=Math.max(8,hi*1.1)}
  else if(b>0){lo=Math.min(lo,b*0.72);hi=Math.max(hi,b*1.28)}
  else{var pad=(hi-lo)*0.2||Math.max(0.05,Math.abs(hi)*0.1);lo-=pad;hi+=pad}
  if(hi-lo<1e-6)hi=lo+1;
  var X=function(k){return L+pw*((MAXH-n)+k)/(MAXH-1)};
  var Y=function(v){return T+ph*(1-(v-lo)/(hi-lo))};
  c.font="11px -apple-system,Segoe UI,Roboto,sans-serif";c.textAlign="right";c.textBaseline="middle";
  for(var g=0;g<=4;g++){var v=lo+(hi-lo)*g/4,yy=Y(v);c.strokeStyle=grid;c.lineWidth=1;c.beginPath();c.moveTo(L,yy);c.lineTo(W-R,yy);c.stroke();c.fillStyle=mut;c.fillText(v.toFixed(tab==="p"?0:tab==="s"?1:2),L-6,yy)}
  c.textAlign="left";c.textBaseline="alphabetic";c.fillStyle=mut;c.fillText("-5 min",L,Ht-8);c.textAlign="right";c.fillText("now",W-R,Ht-8);
  function band(a,bb,col){var y1=Y(Math.min(bb,hi)),y2=Y(Math.max(a,lo));c.fillStyle=col;c.fillRect(L,y1,pw,y2-y1)}
  function hl(v,col,dash){if(v<lo||v>hi)return;c.strokeStyle=col;c.setLineDash(dash||[]);c.lineWidth=1;c.beginPath();c.moveTo(L,Y(v));c.lineTo(W-R,Y(v));c.stroke();c.setLineDash([])}
  if(tab==="s"){hl(3,"rgba(245,179,1,.8)",[5,4]);hl(6,"rgba(255,90,95,.8)",[5,4]);$("#lg_b").innerHTML='<i style="background:#f5b301"></i>thresholds 3 / 6'}
  else if(b>0){
    band(b*1.25,hi,"rgba(255,90,95,.10)");band(lo,b*0.75,"rgba(255,90,95,.10)");
    band(b*1.10,b*1.25,"rgba(245,179,1,.10)");band(b*0.75,b*0.90,"rgba(245,179,1,.10)");
    hl(b,"rgba(46,204,113,.75)",[2,3]);
    $("#lg_b").innerHTML='<i style="background:rgba(46,204,113,.8)"></i>baseline &plusmn;10% / &plusmn;25%';
  }else $("#lg_b").textContent="";
  // state strip
  for(var k2=0;k2<n;k2++){c.fillStyle=COL[H.st[k2]]+"cc";c.fillRect(X(k2)-pw/MAXH/2,T+ph+4,Math.max(pw/MAXH+0.6,1.5),5)}
  // area + line
  if(n>1){
    var gr=c.createLinearGradient(0,T,0,T+ph);gr.addColorStop(0,"rgba(76,195,255,.22)");gr.addColorStop(1,"rgba(76,195,255,0)");
    c.beginPath();c.moveTo(X(0),Y(y[0]));for(var k3=1;k3<n;k3++)c.lineTo(X(k3),Y(y[k3]));c.lineTo(X(n-1),T+ph);c.lineTo(X(0),T+ph);c.closePath();c.fillStyle=gr;c.fill();
    c.lineWidth=2;c.lineJoin="round";c.lineCap="round";
    for(var k4=1;k4<n;k4++){c.strokeStyle=H.st[k4]>=4?COL[H.st[k4]]:"#4cc3ff";c.beginPath();c.moveTo(X(k4-1),Y(y[k4-1]));c.lineTo(X(k4),Y(y[k4]));c.stroke()}
  }
  if(hover>=0&&hover<n){
    var hx=X(hover),hy=Y(y[hover]);c.strokeStyle="rgba(232,238,249,.35)";c.lineWidth=1;c.beginPath();c.moveTo(hx,T);c.lineTo(hx,T+ph);c.stroke();
    c.fillStyle="#e8eef9";c.beginPath();c.arc(hx,hy,4,0,7);c.fill();
    var tx=y[hover].toFixed(tab==="s"?1:2)+" "+unit+"  ·  "+(n-1-hover)+" s ago";c.font="600 12px -apple-system,Segoe UI,Roboto,sans-serif";
    var tw=c.measureText(tx).width+16,bx=Math.min(Math.max(hx-tw/2,L),W-R-tw);c.fillStyle="#0a0f1c";c.strokeStyle="#24324f";c.beginPath();c.rect(bx,T+2,tw,22);c.fill();c.stroke();c.fillStyle="#e8eef9";c.textAlign="left";c.fillText(tx,bx+8,T+17);
  }
}
(function(){
  var cv=$("#cv");
  function mv(e){var r=cv.getBoundingClientRect(),x=(e.touches?e.touches[0].clientX:e.clientX)-r.left,L=40,pw=r.width-L-8,n=H.i.length;
    var k=Math.round((x-L)/pw*(MAXH-1))-(MAXH-n);hover=(k>=0&&k<n)?k:-1;draw()}
  cv.addEventListener("pointermove",mv);cv.addEventListener("pointerdown",mv);
  cv.addEventListener("pointerleave",function(){hover=-1;draw()});cv.addEventListener("pointerup",function(e){if(e.pointerType!=="mouse"){hover=-1;draw()}});
  window.addEventListener("resize",draw);
})();
document.querySelectorAll(".tab").forEach(function(b){b.addEventListener("click",function(){
  document.querySelectorAll(".tab").forEach(function(x){x.classList.remove("on")});b.classList.add("on");tab=b.getAttribute("data-k");draw()})});

/* ---------- actions ---------- */
$("#b_relearn").addEventListener("click",function(){if(confirm("Relearn the baseline?\nRun the motor at its normal load for the next 2 minutes."))cmd("c=relearn")});
$("#cls").addEventListener("click",function(e){var t=e.target.closest("button");if(!t)return;var a=t.getAttribute("data-a"),s=t.getAttribute("data-s");
  if(a==="rec"){if(confirm("Record “"+CLS[s]+"” for 30 s?\nCreate that condition now and hold it steady."))cmd("c=rec&s="+s)}
  else if(a==="stop")cmd("c=stop");else if(a==="clr"){if(confirm("Clear the learned “"+CLS[s]+"” pattern?"))cmd("c=clr&s="+s)}});
$("#b_cal").addEventListener("click",function(){var a=$("#c_a").value,v=$("#c_v").value;if(!a&&!v){alert("Enter the meter current and/or voltage.");return}
  if(confirm("Apply calibration?\nThis restarts baseline learning."))cmd("c=cal&a="+(a||0)+"&v="+(v||0))});

jget("/api/hist?n="+MAXH).then(function(h){H.i=h.i;H.p=h.p;H.s=h.s;H.st=h.st;draw()}).catch(function(){}).then(function(){poll()});
</script>
</body>
</html>
)HTMLUI";


static mm::Monitor        mon;
static WebServer          server(80);
static Preferences        prefs;
static SemaphoreHandle_t  mtx;
static TaskHandle_t       samplerHandle = nullptr;
static volatile uint32_t  overruns = 0;
static mm::HistRec        snap[mm::HIST_N];      // snapshot buffer for history / CSV
static mm::Evt            esnap[mm::EVT_N];

static inline float fin(float v) { return (v != v || v > 3.0e38f || v < -3.0e38f) ? 0.0f : v; }
static inline void lock()   { xSemaphoreTake(mtx, portMAX_DELAY); }
static inline void unlock() { xSemaphoreGive(mtx); }

// ------------------------------ sampling ------------------------------------
static void onTick(void*) {                       // esp_timer: every 500 us
  if (samplerHandle) xTaskNotifyGive(samplerHandle);
}

static void samplerTask(void*) {
  uint32_t lastUs = micros();
  bool first = true;
  for (;;) {
    const uint32_t n = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (n > 1) overruns += (n - 1);
    const uint16_t ri = analogRead(PIN_I);
    const uint16_t rv = analogRead(PIN_V);
    if (mon.pushSample(ri, rv)) {
      const uint32_t now = micros();
      const float dt = (float)(now - lastUs) * 1e-6f;
      lastUs = now;
      const float fs = (mm::WIN * mm::WIN_PER_EPOCH) / (dt > 0 ? dt : 1.0f);
      lock();
      mon.processEpoch();
      if (!first) mon.setTiming(fs, overruns);
      first = false;
      unlock();
    }
  }
}

// ------------------------- chunked JSON / CSV writer -------------------------
struct Out {
  char b[480]; size_t n;
  Out() : n(0) {}
  void flush() { if (n) { b[n] = 0; server.sendContent(b); n = 0; } }
  void put(const char* s) {
    const size_t l = strlen(s);
    if (l >= sizeof(b)) { flush(); server.sendContent(s); return; }
    if (n + l >= sizeof(b)) flush();
    memcpy(b + n, s, l); n += l;
  }
  void f(float v, int d) { char t[28]; snprintf(t, sizeof(t), "%.*f", d, fin(v)); put(t); }
  void u(unsigned long v) { char t[16]; snprintf(t, sizeof(t), "%lu", v); put(t); }
};

static void beginChunked(const char* type) {
  server.sendHeader("Cache-Control", "no-store");
  server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server.send(200, type, "");
}
static void endChunked() { server.sendContent(""); }

static int snapshotHistory(int want) {            // copy newest `want` records
  lock();
  int n = mon.histCount; if (want < n) n = want;
  const int skip = mon.histCount - n;
  for (int k = 0; k < n; k++) snap[k] = mon.histAt(skip + k);
  unlock();
  return n;
}

// -------------------------------- web handlers ------------------------------
static void handleRoot() {
  server.sendHeader("Cache-Control", "no-cache");
  server.send_P(200, "text/html; charset=utf-8", INDEX_HTML);
}

static void handleStatus() {
  mm::Status s; mm::Baseline b; uint8_t cv[mm::NCLASS];
  lock();
  s = mon.st; b = mon.pst.base;
  for (int c = 0; c < mm::NCLASS; c++) cv[c] = mon.pst.cls[c].valid;
  unlock();
  char buf[960];
  snprintf(buf, sizeof(buf),
    "{\"up\":%lu,\"ep\":%lu,\"st\":%u,\"i\":%.3f,\"v\":%.1f,\"p\":%.1f,\"pf\":%.3f,\"cr\":%.2f,\"f\":%.2f,"
    "\"thd\":%.4f,\"sb\":%.4f,\"rp\":%.4f,\"dev\":%.1f,\"sc\":%.2f,\"h\":%.0f,"
    "\"bv\":%u,\"bi\":%.3f,\"bp\":%.1f,\"bpf\":%.3f,\"bvv\":%.1f,\"cal\":%u,\"sup\":%u,\"lab\":%d,"
    "\"ld\":%u,\"lt\":%d,\"rs\":%d,\"rd\":%u,\"rt\":%d,\"cf\":%u,\"cls\":[%u,%u,%u,%u],"
    "\"raw\":{\"bi\":%.0f,\"bv\":%.0f,\"pi\":%ld,\"pv\":%ld,\"ci\":%u,\"cv\":%u},\"fs\":%.1f,\"ov\":%lu,\"msg\":\"%s\"}",
    (unsigned long)(millis() / 1000), (unsigned long)s.t, (unsigned)s.state,
    fin(s.irms), fin(s.vrms), fin(s.pwr), fin(s.pf), fin(s.crest), fin(s.freq),
    fin(s.thd), fin(s.side), fin(s.ripple), fin(s.devPct), fin(s.score), fin(s.health),
    (unsigned)s.baseValid, fin(b.mu[mm::F_IRMS]), fin(b.mu[mm::F_PWR]), fin(b.mu[mm::F_PF]), fin(b.vMean),
    (unsigned)s.calibrated, (unsigned)s.supply, (int)s.label,
    (unsigned)s.learnDone, mm::LEARN_EPOCHS, (int)s.recSlot, (unsigned)s.recDone, mm::RECORD_EPOCHS, (unsigned)s.confirm,
    (unsigned)cv[0], (unsigned)cv[1], (unsigned)cv[2], (unsigned)cv[3],
    fin(s.biasI), fin(s.biasV), (long)s.ppI, (long)s.ppV, (unsigned)s.clipI, (unsigned)s.clipV,
    fin(s.fsMeasured), (unsigned long)s.overruns, s.msg);
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", buf);
}

static void handleHist() {
  int want = server.hasArg("n") ? server.arg("n").toInt() : 300;
  if (want < 1) want = 1; if (want > mm::HIST_N) want = mm::HIST_N;
  const int n = snapshotHistory(want);
  beginChunked("application/json");
  Out o;
  o.put("{\"i\":[");  for (int k = 0; k < n; k++) { o.f(snap[k].x[mm::F_IRMS], 3); if (k < n - 1) o.put(","); }
  o.put("],\"p\":["); for (int k = 0; k < n; k++) { o.f(snap[k].x[mm::F_PWR], 1);  if (k < n - 1) o.put(","); }
  o.put("],\"s\":["); for (int k = 0; k < n; k++) { o.f(snap[k].score, 2);         if (k < n - 1) o.put(","); }
  o.put("],\"st\":["); for (int k = 0; k < n; k++) { o.u(snap[k].st);               if (k < n - 1) o.put(","); }
  o.put("]}");
  o.flush(); endChunked();
}

static void handleEvents() {
  int n;
  lock();
  n = mon.evtCount;
  for (int k = 0; k < n; k++) esnap[k] = mon.evtAt(k);
  unlock();
  beginChunked("application/json");
  Out o;
  o.put("[");
  for (int k = 0; k < n; k++) {
    o.put("{\"t\":"); o.u(esnap[k].t);
    o.put(",\"f\":"); o.u(esnap[k].from);
    o.put(",\"s\":"); o.u(esnap[k].to);
    o.put(",\"m\":\"");
    char m[mm::EVT_MSG_LEN]; size_t j = 0;
    for (const char* p = esnap[k].msg; *p && j < sizeof(m) - 1; p++) if (*p != '"' && *p != '\\') m[j++] = *p;
    m[j] = 0; o.put(m); o.put("\"}");
    if (k < n - 1) o.put(",");
  }
  o.put("]");
  o.flush(); endChunked();
}

static void handleCsv() {
  const int n = snapshotHistory(mm::HIST_N);
  server.sendHeader("Content-Disposition", "attachment; filename=motor_log.csv");
  beginChunked("text/csv");
  Out o;
  o.put("t_s,state,irms_A,vrms_V,power_W,pf,crest,thd,sideband,ripple,score\n");
  for (int k = 0; k < n; k++) {
    const mm::HistRec& r = snap[k];
    o.u(r.t); o.put(","); o.put(mm::STATE_NAME[r.st < 6 ? r.st : 0]); o.put(",");
    o.f(r.x[mm::F_IRMS], 3); o.put(","); o.f(r.v, 1); o.put(",");
    o.f(r.x[mm::F_PWR], 1); o.put(",");  o.f(r.x[mm::F_PF], 3); o.put(",");
    o.f(r.x[mm::F_CREST], 2); o.put(","); o.f(r.x[mm::F_THD], 4); o.put(",");
    o.f(r.x[mm::F_SIDE], 4); o.put(",");  o.f(r.x[mm::F_RIPPLE], 4); o.put(",");
    o.f(r.score, 2); o.put("\n");
  }
  o.flush(); endChunked();
}

static void handleCmd() {
  const String c = server.arg("c");
  bool ok = true;
  lock();
  if (c == "relearn")      mon.cmdRelearn();
  else if (c == "rec")     mon.cmdRecord(server.arg("s").toInt());
  else if (c == "stop")    mon.cmdStopRecord();
  else if (c == "clr")     mon.cmdClearClass(server.arg("s").toInt());
  else if (c == "cal")     ok = mon.cmdCalibrate(server.arg("a").toFloat(), server.arg("v").toFloat());
  else                     ok = false;
  unlock();
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", ok ? "{\"ok\":1}" : "{\"ok\":0}");
}

static void handleNotFound() { server.send(404, "text/plain", "Not found"); }
static void handleNoContent() { server.send(204, "text/plain", ""); }

// ---------------------------------- setup -----------------------------------
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("Smart Motor Health Monitor - Team 15");

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);                  // ~0 - 3.1 V input range

  mtx = xSemaphoreCreateMutex();
  mon.begin();

  prefs.begin("mm", false);
  if (prefs.getBytesLength("p") == sizeof(mm::Persist)) {
    mm::Persist p;
    prefs.getBytes("p", &p, sizeof(p));
    if (mon.load(p)) Serial.println("Loaded saved calibration / baseline / fault patterns.");
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS, 6, 0, 4);
  Serial.print("WiFi network: "); Serial.print(AP_SSID);
  Serial.print("   password: "); Serial.println(AP_PASS);
  Serial.print("Open http://"); Serial.println(WiFi.softAPIP().toString());

  server.on("/", handleRoot);
  server.on("/api/status", handleStatus);
  server.on("/api/hist", handleHist);
  server.on("/api/events", handleEvents);
  server.on("/api/csv", handleCsv);
  server.on("/api/cmd", handleCmd);
  server.on("/favicon.ico", handleNoContent);
  server.onNotFound(handleNotFound);
  server.begin();

  xTaskCreatePinnedToCore(samplerTask, "sampler", 6144, nullptr, 5, &samplerHandle, 1);
  esp_timer_create_args_t targs = {};
  targs.callback = &onTick;
  targs.name = "samp";
  esp_timer_handle_t th;
  esp_timer_create(&targs, &th);
  esp_timer_start_periodic(th, 500);               // 2 kHz

  Serial.println("t_s,state,irms_A,vrms_V,power_W,pf,crest,thd,sideband,ripple,freq_Hz,dev_pct,score");
}

// ----------------------------------- loop -----------------------------------
void loop() {
  server.handleClient();

  // Serial CSV log, one line per second
  static uint32_t lastT = 0;
  if (mon.st.t != lastT) {
    mm::Status s;
    lock(); s = mon.st; unlock();
    lastT = s.t;
    Serial.printf("%lu,%s,%.3f,%.1f,%.1f,%.3f,%.2f,%.4f,%.4f,%.4f,%.2f,%.1f,%.2f\n",
                  (unsigned long)s.t, mm::STATE_NAME[s.state < 6 ? s.state : 0],
                  fin(s.irms), fin(s.vrms), fin(s.pwr), fin(s.pf), fin(s.crest), fin(s.thd),
                  fin(s.side), fin(s.ripple), fin(s.freq), fin(s.devPct), fin(s.score));
  }

  // Save calibration / baseline / learned patterns when they change
  if (mon.persistDirty) {
    mm::Persist p;
    lock(); p = mon.pst; mon.persistDirty = false; unlock();
    prefs.putBytes("p", &p, sizeof(p));
    Serial.println("Saved settings to flash.");
  }

  // Serial commands:  r = relearn baseline,  R = erase saved settings and reboot
  while (Serial.available()) {
    const int ch = Serial.read();
    if (ch == 'r') { lock(); mon.cmdRelearn(); unlock(); Serial.println("Relearning baseline."); }
    if (ch == 'R') { prefs.clear(); Serial.println("Settings erased - rebooting."); delay(200); ESP.restart(); }
  }
  delay(2);
}
