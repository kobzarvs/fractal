import assert from 'node:assert/strict';
import test from 'node:test';
import { FrameRateMeter } from '../src/frame-rate.ts';

test('counts rendered frames, not reads from an idle animation loop', () => {
  const meter = new FrameRateMeter();
  assert.equal(meter.sample(0), null);
  for (let time = 0; time <= 1000; time += 10) {
    if (time % 50 === 0) meter.record(time);
    meter.sample(time);
  }
  assert.equal(meter.sample(1000), 20);
});

test('does not report a rate from one frame or a tiny startup interval', () => {
  const meter = new FrameRateMeter();
  meter.record(0);
  assert.equal(meter.sample(0), null);
  meter.record(10);
  assert.equal(meter.sample(10), null);
});

test('includes dropped frames and ages out an idle scene', () => {
  const meter = new FrameRateMeter();
  for (let time = 0; time <= 500; time += 50) meter.record(time);
  assert.equal(meter.sample(1000), 10);
  assert.equal(meter.sample(1600), 0);
});

test('recent cadence replaces old slow frames in the rolling window', () => {
  const meter = new FrameRateMeter();
  for (let time = 0; time <= 1000; time += 100) meter.record(time);
  assert.equal(meter.sample(1000), 10);
  for (let time = 1020; time <= 2000; time += 20) meter.record(time);
  assert.equal(meter.sample(2000), 50);
});

test('reset excludes the previous engine and time spent in a hidden tab', () => {
  const meter = new FrameRateMeter();
  for (let time = 0; time <= 1000; time += 20) meter.record(time);
  meter.reset();
  assert.equal(meter.sample(10000), null);
  for (let time = 10000; time <= 11000; time += 50) meter.record(time);
  assert.equal(meter.sample(11000), 20);
});

test('resuming after an idle gap does not count time outside the last second', () => {
  const meter = new FrameRateMeter();
  for (let time = 0; time <= 500; time += 50) meter.record(time);
  assert.equal(meter.sample(4000), 0);
  for (let time = 5000; time <= 5304; time += 16) meter.record(time);
  // Twenty renders in the last second, including the idle part of that second.
  assert.equal(meter.sample(5304), 20);
});
