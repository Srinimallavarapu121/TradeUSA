// Unit tests for indicators.js (rollingRSI, calibrateProb).
// Run with: node indicators.test.js   (uses Node's built-in test runner, no deps needed)
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { calibrateProb, rollingRSI, CALIBRATION_CURVE } = require('./indicators');

function closeTo(actual, expected, delta = 1e-9) {
  assert.ok(
    Math.abs(actual - expected) <= delta,
    `expected ${actual} to be within ${delta} of ${expected}`
  );
}

test('calibrateProb: clamps at the curve endpoints', () => {
  assert.equal(calibrateProb(0), 0.0);
  assert.equal(calibrateProb(-10), 0.0); // below range -> first point's y
  assert.equal(calibrateProb(100), 77.1);
  assert.equal(calibrateProb(150), 77.1); // above range -> last point's y
});

test('calibrateProb: returns exact y at a known curve point', () => {
  assert.equal(calibrateProb(55.0), 55.3);
});

test('calibrateProb: linearly interpolates between two curve points', () => {
  // Between (3.7, 7.6) and (15.2, 20.5): t = (10-3.7)/(15.2-3.7)
  closeTo(calibrateProb(10), 14.666956521739131, 1e-6);
});

test('calibrateProb: is monotonically non-decreasing (matches CALIBRATION_CURVE shape)', () => {
  let prev = -Infinity;
  for (let raw = 0; raw <= 100; raw += 5) {
    const y = calibrateProb(raw);
    assert.ok(y >= prev - 1e-9, `calibrateProb(${raw})=${y} should be >= previous ${prev}`);
    prev = y;
  }
  assert.equal(CALIBRATION_CURVE[0][0], 0.0);
  assert.equal(CALIBRATION_CURVE[CALIBRATION_CURVE.length - 1][0], 100.0);
});

test('rollingRSI: output length matches input length', () => {
  const closes = [10, 11, 10, 13, 12, 14];
  assert.equal(rollingRSI(closes, 3).length, closes.length);
});

test('rollingRSI: defaults to 50 before the period warms up', () => {
  const closes = [10, 11, 10, 13, 12, 14];
  const rsi = rollingRSI(closes, 3);
  assert.equal(rsi[0], 50);
  assert.equal(rsi[1], 50);
  assert.equal(rsi[2], 50);
});

test('rollingRSI: matches hand-computed Wilder smoothing for a small period', () => {
  // closes: 10,11,10,13,12,14 with period=3
  // diffs:  +1,-1,+3,-1,+2 -> gains [1,0,3,0,2], losses [0,1,0,1,0]
  // i=3 (seed): avgGain=4/3, avgLoss=1/3 -> rs=4       -> rsi=100-100/5=80
  // i=4:        avgGain=8/9, avgLoss=5/9 -> rs=1.6     -> rsi=100-100/2.6=61.53846153846154
  // i=5:        avgGain=34/27,avgLoss=10/27 -> rs=3.4  -> rsi=100-100/4.4=77.27272727272727
  const closes = [10, 11, 10, 13, 12, 14];
  const rsi = rollingRSI(closes, 3);
  closeTo(rsi[3], 80, 1e-9);
  closeTo(rsi[4], 61.53846153846154, 1e-9);
  closeTo(rsi[5], 77.27272727272727, 1e-9);
});

test('rollingRSI: a strictly rising series saturates at 100 (no losses)', () => {
  const closes = Array.from({ length: 20 }, (_, i) => 100 + i); // +1 every bar
  const rsi = rollingRSI(closes, 14);
  closeTo(rsi[19], 100, 1e-9);
});

test('rollingRSI: a strictly falling series saturates at 0 (no gains)', () => {
  const closes = Array.from({ length: 20 }, (_, i) => 200 - i); // -1 every bar
  const rsi = rollingRSI(closes, 14);
  closeTo(rsi[19], 0, 1e-9);
});

test('rollingRSI: stays within [0, 100] for a noisy series', () => {
  const closes = [100];
  let seed = 42;
  const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (let i = 1; i < 200; i++) closes.push(closes[i - 1] + (rand() - 0.5) * 4);
  const rsi = rollingRSI(closes, 14);
  for (const v of rsi) {
    assert.ok(v >= 0 && v <= 100, `RSI value ${v} out of [0,100] range`);
  }
});
