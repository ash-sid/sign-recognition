/**
 * web/tests/motion.test.ts
 *
 * Checks the segmentation rules that decide when the app answers at all.
 *
 * These are thresholds rather than a contract, so nothing here pins a number.
 * What it pins is the behaviour those numbers exist for: that an absent hand
 * does not read as movement, that a moment of stillness inside a sign does not
 * split it in two, and that a burst too short to be a sign yields nothing
 * rather than a guess.
 */
import { describe, expect, it } from "vitest";

import {
  BurstTracker,
  MIN_BURST_FRAMES,
  MOVING_THRESHOLD,
  STILL_THRESHOLD,
  frameMovement,
} from "../src/motion";
import { HAND_LANDMARKS, NUM_COORDS, TARGET_LEN, at } from "../src/preprocessing";

function sequence(fill: (t: number, l: number, c: number) => number): Float32Array {
  const out = new Float32Array(TARGET_LEN * 50 * NUM_COORDS);
  for (let t = 0; t < TARGET_LEN; t++) {
    for (let l = 0; l < 50; l++) {
      for (let c = 0; c < NUM_COORDS; c++) out[at(t, l, c)] = fill(t, l, c);
    }
  }
  return out;
}

describe("frameMovement", () => {
  it("is zero when nothing changed between the last two frames", () => {
    expect(frameMovement(sequence(() => 0.4), TARGET_LEN)).toBe(0);
  });

  it("rises with how far the hands travelled", () => {
    const small = frameMovement(sequence((t) => 0.4 + t * 0.001), TARGET_LEN);
    const large = frameMovement(sequence((t) => 0.4 + t * 0.05), TARGET_LEN);
    expect(small).toBeGreaterThan(0);
    expect(large).toBeGreaterThan(small);
  });

  it("ignores a hand that is absent, rather than reading it as a jump", () => {
    // The left hand is zeroed throughout, which is what an unused hand looks
    // like after preprocessing; only the right hand moves.
    const data = sequence((t, l) => (l < HAND_LANDMARKS ? 0 : 0.4 + t * 0.02));
    const movement = frameMovement(data, TARGET_LEN);
    expect(movement).toBeGreaterThan(0);
    expect(Number.isFinite(movement)).toBe(true);
  });
});

describe("BurstTracker", () => {
  it("does not start on a still hand", () => {
    const burst = new BurstTracker();
    expect(burst.observe(STILL_THRESHOLD / 2, true)).toBe(false);
    expect(burst.active).toBe(false);
  });

  it("does not start when there are no hands, however much moved", () => {
    const burst = new BurstTracker();
    expect(burst.observe(MOVING_THRESHOLD * 10, false)).toBe(false);
  });

  it("holds through a lull between the two thresholds", () => {
    const burst = new BurstTracker();
    burst.observe(MOVING_THRESHOLD * 1.5, true);
    const between = (MOVING_THRESHOLD + STILL_THRESHOLD) / 2;
    expect(burst.observe(between, true)).toBe(true);
    expect(burst.active).toBe(true);
  });

  it("ends once movement drops below the lower threshold", () => {
    const burst = new BurstTracker();
    burst.observe(MOVING_THRESHOLD * 1.5, true);
    expect(burst.observe(STILL_THRESHOLD / 2, true)).toBe(false);
  });

  it("yields nothing for a burst too short to be a sign", () => {
    const burst = new BurstTracker();
    burst.observe(MOVING_THRESHOLD * 1.5, true);
    for (let i = 0; i < MIN_BURST_FRAMES - 1; i++) burst.add(Float32Array.from([0.2, 0.8]));
    expect(burst.take()).toBeUndefined();
  });

  it("averages the scores it saw across a long enough burst", () => {
    const burst = new BurstTracker();
    burst.observe(MOVING_THRESHOLD * 1.5, true);
    // An even count, so an exact half-and-half average is the right answer
    // whatever the minimum happens to be set to.
    for (let i = 0; i < 2 * MIN_BURST_FRAMES; i++) {
      burst.add(Float32Array.from(i % 2 === 0 ? [1, 0] : [0, 1]));
    }
    const averaged = burst.take();
    expect(averaged).toBeDefined();
    expect(Array.from(averaged!)).toEqual([0.5, 0.5]);
  });

  it("clears itself once taken", () => {
    const burst = new BurstTracker();
    burst.observe(MOVING_THRESHOLD * 1.5, true);
    for (let i = 0; i < MIN_BURST_FRAMES; i++) burst.add(Float32Array.from([1, 0]));
    burst.take();
    expect(burst.active).toBe(false);
    expect(burst.length).toBe(0);
  });
});
