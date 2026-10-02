/**
 * web/tests/sequenceSlots.test.ts
 *
 * Checks the browser's assignSequenceSlots against the cases
 * tests/test_sequence_slots.py checks the Python implementation against:
 * hand-written cases whose slots were reasoned out by hand, and generated
 * cases, some found by search, where float32 arithmetic would decide
 * differently and the expected slots are what src/tracking.py assigned.
 *
 * Coordinates are rounded to float32 on the way in, as the landmarker's are.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

import { HAND_LANDMARKS, LEFT_HAND_START, RIGHT_HAND_START } from "../src/preprocessing";
import { assignSequenceSlots, type SequenceFrame } from "../src/tracking";

interface SequenceCase {
  name: string;
  aspect: number;
  frames: {
    hands: number[][];
    pose_wrists: { left: [number, number]; right: [number, number] } | null;
  }[];
  expected: ("left" | "right")[][];
}

const POSE_LANDMARKS = 33;
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;
const NAMES = new Map([
  [LEFT_HAND_START, "left"],
  [RIGHT_HAND_START, "right"],
]);

function point(x: number, y: number, z = 0): NormalizedLandmark {
  return { x: Math.fround(x), y: Math.fround(y), z: Math.fround(z), visibility: 0 };
}

/** [x, y] stands for 21 landmarks at one point; otherwise 21 x 3 flattened. */
function hand(values: number[]): NormalizedLandmark[] {
  if (values.length === 2) return Array.from({ length: HAND_LANDMARKS }, () => point(values[0], values[1]));
  return Array.from({ length: HAND_LANDMARKS }, (_, i) =>
    point(values[3 * i], values[3 * i + 1], values[3 * i + 2]),
  );
}

function pose(wrists: SequenceCase["frames"][number]["pose_wrists"]): NormalizedLandmark[] | undefined {
  if (!wrists) return undefined;
  const out = Array.from({ length: POSE_LANDMARKS }, () => point(0, 0));
  out[POSE_LEFT_WRIST] = point(...wrists.left);
  out[POSE_RIGHT_WRIST] = point(...wrists.right);
  return out;
}

for (const fixture of ["sequence_slots.json", "sequence_slots_generated.json"]) {
  const cases: SequenceCase[] = JSON.parse(
    readFileSync(new URL(`./fixtures/${fixture}`, import.meta.url), "utf-8"),
  ).cases;

  describe(`assignSequenceSlots: ${fixture}`, () => {
    it("has cases to check", () => {
      expect(cases.length).toBeGreaterThan(0);
    });
    for (const c of cases) {
      it(c.name, () => {
        const frames: SequenceFrame[] = c.frames.map((f) => ({
          hands: f.hands.map(hand),
          pose: pose(f.pose_wrists),
        }));
        const slots = assignSequenceSlots(frames, c.aspect);
        expect(slots.map((frame) => frame.map((s) => NAMES.get(s)))).toEqual(c.expected);
      });
    }
  });
}
