/**
 * web/tests/parity.test.ts
 *
 * Holds the TypeScript preprocessing port to the Python implementation, case
 * by case, against the fixture built by src/make_fixture.py.
 *
 * Nothing else in this project fails when the two drift apart. Both sides
 * produce a tensor of the right shape full of plausible numbers, the graph
 * classifies it without complaint, and the only symptom is that the demo is
 * worse than the accuracy figure it advertises. That reads as a model problem
 * and would be debugged as one.
 *
 * Both implementations compute in doubles and narrow once at the end, so their
 * outputs are expected to be equal bit for bit rather than close. The
 * tolerances below are tight enough that a real divergence cannot hide under
 * them, and the failure message reports where the largest one is so a
 * mismatch names the stage it came from.
 *
 * The graph runs here too, on the same tensors, so a runtime that disagrees
 * with the one the published numbers were measured on shows up as its own
 * failure rather than as a preprocessing one.
 *
 * Run: npm test
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it } from "vitest";

import { mirrorSequence } from "../src/mirror";
import { NUM_COORDS, NUM_LANDMARKS, TARGET_LEN, processSequence } from "../src/preprocessing";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = join(HERE, "fixtures");
const GRAPH_PATH = join(HERE, "..", "..", "models", "abl_hands_aug.onnx");

/**
 * Preprocessing is expected to be exact. A non-zero allowance exists only so
 * that a failure reports a magnitude instead of a boolean, and is far below
 * the size of any real disagreement about interpolation or ordering.
 */
const TENSOR_TOLERANCE = 0;

/**
 * The graph runs under a different build of the runtime here than the one the
 * stored predictions were measured with, so its arithmetic is allowed to move
 * in the last bits. What is not allowed to move is which class wins.
 *
 * Relative to the largest score, because one case drives the normalization
 * scale into its clip and comes out with coordinates a few hundred times
 * larger than usual. Its logits are correspondingly large and so is the
 * absolute gap between two runtimes rounding them, while the relative gap is
 * the same 1e-7 as everywhere else. An absolute bound would either fail on
 * that case or be too loose to catch anything on the others.
 */
const LOGIT_TOLERANCE = 1e-6;

interface Block {
  offset: number;
  shape: number[];
}

interface FixtureCase {
  name: string;
  description: string;
  frames: number;
  input: Block;
  expected: Block;
  logits: Block;
  mirrored?: Block;
}

interface Fixture {
  target_len: number;
  num_landmarks: number;
  num_coords: number;
  cases: FixtureCase[];
}

const manifest: Fixture = JSON.parse(readFileSync(join(FIXTURE_DIR, "parity.json"), "utf-8"));
const blobBytes = readFileSync(join(FIXTURE_DIR, "parity.bin"));
const blob = new Float32Array(
  blobBytes.buffer.slice(blobBytes.byteOffset, blobBytes.byteOffset + blobBytes.byteLength),
);

function read(block: Block): Float32Array {
  const count = block.shape.reduce((a, b) => a * b, 1);
  if (block.offset + count > blob.length) {
    throw new Error(`Fixture block runs past the end of parity.bin; the two files disagree.`);
  }
  return blob.subarray(block.offset, block.offset + count);
}

/** Largest absolute difference and where it is, for a message worth reading. */
function worstDifference(
  actual: Float32Array,
  expected: Float32Array,
): { value: number; index: number } {
  let value = 0;
  let index = -1;
  for (let i = 0; i < expected.length; i++) {
    const difference = Math.abs(actual[i] - expected[i]);
    if (difference > value) {
      value = difference;
      index = i;
    }
  }
  return { value, index };
}

function describeIndex(index: number): string {
  if (index < 0) return "nowhere";
  const coord = index % NUM_COORDS;
  const landmark = Math.floor(index / NUM_COORDS) % NUM_LANDMARKS;
  const frame = Math.floor(index / (NUM_COORDS * NUM_LANDMARKS));
  return `frame ${frame}, landmark ${landmark}, coordinate ${"xyz"[coord]}`;
}

function topFive(logits: Float32Array): number[] {
  return Array.from(logits.keys())
    .sort((a, b) => logits[b] - logits[a])
    .slice(0, 5);
}

describe("fixture", () => {
  it("matches the shape constants the port was written against", () => {
    expect(manifest.target_len).toBe(TARGET_LEN);
    expect(manifest.num_landmarks).toBe(NUM_LANDMARKS);
    expect(manifest.num_coords).toBe(NUM_COORDS);
  });

  it("covers the branches that only show up in bad tracking", () => {
    const names = manifest.cases.map((c) => c.name);
    for (const required of [
      "left_hand_unused",
      "right_hand_unused",
      "dropout_midway",
      "dropout_at_edges",
      "landmark_never_tracked",
      "shoulders_coincide",
      "pose_dropout",
    ]) {
      expect(names).toContain(required);
    }
  });
});

describe("preprocessing", () => {
  for (const testCase of manifest.cases) {
    it(`${testCase.name}: ${testCase.description}`, () => {
      const input = Float64Array.from(read(testCase.input));
      const expected = read(testCase.expected);
      const actual = processSequence(input, testCase.frames);

      expect(actual.length).toBe(TARGET_LEN * NUM_LANDMARKS * NUM_COORDS);
      expect(actual.length).toBe(expected.length);
      expect(actual.some(Number.isNaN)).toBe(false);

      const worst = worstDifference(actual, expected);
      expect(
        worst.value,
        `largest difference ${worst.value} at ${describeIndex(worst.index)}`,
      ).toBeLessThanOrEqual(TENSOR_TOLERANCE);
    });
  }

  it("does not modify the array it is given", () => {
    const testCase = manifest.cases[0];
    const input = Float64Array.from(read(testCase.input));
    const copy = Float64Array.from(input);
    processSequence(input, testCase.frames);
    expect(Array.from(input)).toEqual(Array.from(copy));
  });

  it("rejects a window whose length disagrees with its frame count", () => {
    const short = new Float64Array(10 * NUM_LANDMARKS * NUM_COORDS);
    expect(() => processSequence(short, 11)).toThrow();
  });
});

describe("mirror", () => {
  const mirrored = manifest.cases.filter((c) => c.mirrored);

  it("has cases to check", () => {
    expect(mirrored.length).toBeGreaterThan(0);
  });

  for (const testCase of mirrored) {
    it(`${testCase.name}: reflects about the body midline`, () => {
      const actual = mirrorSequence(read(testCase.expected), TARGET_LEN);
      const worst = worstDifference(actual, read(testCase.mirrored!));
      expect(
        worst.value,
        `largest difference ${worst.value} at ${describeIndex(worst.index)}`,
      ).toBeLessThanOrEqual(TENSOR_TOLERANCE);
    });
  }

  it("is its own inverse", () => {
    const once = mirrorSequence(read(manifest.cases[0].expected), TARGET_LEN);
    const twice = mirrorSequence(once, TARGET_LEN);
    const worst = worstDifference(twice, read(manifest.cases[0].expected));
    expect(worst.value).toBeLessThanOrEqual(TENSOR_TOLERANCE);
  });
});

describe("exported graph", () => {
  let runtime: ort.InferenceSession;

  beforeAll(async () => {
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";
    runtime = await ort.InferenceSession.create(readFileSync(GRAPH_PATH), {
      executionProviders: ["wasm"],
    });
  }, 60_000);

  it("takes the documented tensor", () => {
    expect(runtime.inputNames).toEqual(["landmarks"]);
    expect(runtime.outputNames).toEqual(["logits"]);
  });

  for (const testCase of manifest.cases) {
    it(`${testCase.name}: scores as it does in Python`, async () => {
      const tensor = processSequence(Float64Array.from(read(testCase.input)), testCase.frames);
      const output = await runtime.run({
        landmarks: new ort.Tensor("float32", tensor, [1, TARGET_LEN, NUM_LANDMARKS, NUM_COORDS]),
      });
      const actual = output.logits.data as Float32Array;
      const expected = read(testCase.logits);

      expect(Array.from(output.logits.dims)).toEqual([1, expected.length]);
      const scale = Math.max(...Array.from(expected, Math.abs));
      const worst = worstDifference(actual, expected);
      expect(
        worst.value / scale,
        `largest logit difference ${worst.value} against a largest score of ${scale}`,
      ).toBeLessThan(LOGIT_TOLERANCE);
      expect(topFive(actual)).toEqual(topFive(expected));
    });
  }
});
