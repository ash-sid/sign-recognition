/**
 * web/tests/slots.test.ts
 *
 * Checks the browser's slot assignment against the hand-written cases in
 * tests/fixtures/slots.json. tests/test_slots.py checks the offline
 * extractor's port against the same file, so the two agree with each other by
 * each agreeing with cases neither of them produced.
 *
 * Coordinates are rounded to float32 on the way in, as the landmarker's are,
 * and the arithmetic after that is in doubles, as it is in the Python port.
 * One case is a near-tie that float32 arithmetic resolves the other way, so a
 * change of precision on either side fails here rather than showing up as a
 * handful of misplaced hands in a cache of millions of frames.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

import { HAND_LANDMARKS, LEFT_HAND_START, RIGHT_HAND_START } from "../src/preprocessing";
import { assignSlots } from "../src/tracking";

interface SlotCase {
  name: string;
  hands: [number, number][];
  pose_wrists: { left: [number, number]; right: [number, number] } | null;
  handedness: string[];
  expected: ("left" | "right")[];
}

const FIXTURE = new URL("./fixtures/slots.json", import.meta.url);
const cases: SlotCase[] = JSON.parse(readFileSync(FIXTURE, "utf-8")).cases;

const POSE_LANDMARKS = 33;
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;

function point([x, y]: [number, number]): NormalizedLandmark {
  return { x: Math.fround(x), y: Math.fround(y), z: 0, visibility: 0 };
}

function hand(wrist: [number, number]): NormalizedLandmark[] {
  return Array.from({ length: HAND_LANDMARKS }, () => point(wrist));
}

function pose(wrists: SlotCase["pose_wrists"]): NormalizedLandmark[] | undefined {
  if (!wrists) return undefined;
  const out = Array.from({ length: POSE_LANDMARKS }, () => point([NaN, NaN]));
  out[POSE_LEFT_WRIST] = point(wrists.left);
  out[POSE_RIGHT_WRIST] = point(wrists.right);
  return out;
}

const NAMES = new Map([
  [LEFT_HAND_START, "left"],
  [RIGHT_HAND_START, "right"],
]);

describe("assignSlots", () => {
  it("has cases to check", () => {
    expect(cases.length).toBeGreaterThan(0);
  });

  for (const c of cases) {
    it(c.name, () => {
      const hands = c.hands.map(hand);
      const slots = assignSlots(hands, pose(c.pose_wrists), c.handedness).slice(0, hands.length);
      expect(slots.map((s) => (s === undefined ? "none" : NAMES.get(s)))).toEqual(c.expected);
    });
  }
});
