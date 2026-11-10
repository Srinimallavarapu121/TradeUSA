// Pure, dependency-free indicator/calibration functions shared by stock_details.html
// (loaded via <script src="indicators.js">) and indicators.test.js (via require()).
// Keep this file DOM-free so it stays testable under plain Node.

// Empirical raw->actual calibration curve derived from a 7-day/10,000-point causal
// backtest (see backtest.py) -- corrects the raw model's overconfidence on rare/large moves.
const CALIBRATION_CURVE = [[0.0,0.0], [3.7,7.6], [15.2,20.5], [24.8,26.3], [34.9,34.6], [45.1,44.0], [55.0,55.3], [65.1,63.3], [75.0,72.9], [83.7,74.7], [92.1,77.1], [100.0,77.1]];

function calibrateProb(raw) {
  const pts = CALIBRATION_CURVE;
  if (raw <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    if (raw <= pts[i][0]) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      const t = x1 > x0 ? (raw - x0) / (x1 - x0) : 0;
      return y0 + t * (y1 - y0);
    }
  }
  return pts[pts.length - 1][1];
}

// Wilder's RSI, causal (each value depends only on itself and prior closes).
function rollingRSI(closes, period = 14) {
  const n = closes.length;
  const rsi = new Array(n).fill(50);
  const gains = new Array(n).fill(0);
  const losses = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const diff = closes[i] - closes[i - 1];
    gains[i] = Math.max(diff, 0);
    losses[i] = Math.max(-diff, 0);
  }
  let avgGain = 0, avgLoss = 0;
  for (let i = 1; i < n; i++) {
    if (i < period) continue;
    if (i === period) {
      let sg = 0, sl = 0;
      for (let k = 1; k <= period; k++) { sg += gains[k]; sl += losses[k]; }
      avgGain = sg / period;
      avgLoss = sl / period;
    } else {
      avgGain = (avgGain * (period - 1) + gains[i]) / period;
      avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    }
    rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return rsi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { CALIBRATION_CURVE, calibrateProb, rollingRSI };
}
