/**
 * web/src/preprocessing.ts
 *
 * Turns a window of tracked frames into the fixed-shape tensor the exported
 * graph expects. Port of src/preprocessing.py; both implement the interface in
 * reports/contract.md and neither may drift from it alone.
 *
 * The port starts one step later than the Python module does. There, a
 * sequence arrives as rows in a parquet file and has to be pivoted into an
 * array first; here the array is assembled a frame at a time from a
 * landmarker. That step is where the two platforms genuinely differ, and
 * everything after it is the contract. So the boundary is the (T, 50, 3)
 * array with NaN for anything not tracked, and this file implements what
 * happens to it from there.
 *
 * Arithmetic is done in doubles and narrowed to float32 once at the end,
 * which is what the reference implementation does. Narrowing earlier, or
 * reordering an expression for tidiness, moves results in the last bits and
 * turns an exact comparison into an approximate one.
 *
 * Held to the reference by tests/parity.test.ts against a fixture built by
 * src/make_fixture.py. A divergence here does not throw. It yields a slightly
 * wrong tensor, the classifier answers confidently, and the demo looks like a
 * weaker model than the one that was measured.
 */

/** Frames per tensor after resampling. */
export const TARGET_LEN = 70;

/** Landmarks per frame: left hand, right hand, then the pose subset. */
export const NUM_LANDMARKS = 50;

/** Coordinates per landmark: x, y, z. */
export const NUM_COORDS = 3;

export const HAND_LANDMARKS = 21;
export const LEFT_HAND_START = 0;
export const RIGHT_HAND_START = HAND_LANDMARKS;
export const HANDS_END = 2 * HAND_LANDMARKS;
export const POSE_START = HANDS_END;
export const POSE_LANDMARKS = NUM_LANDMARKS - POSE_START;

/**
 * Pose landmark positions within the output ordering. The first two are the
 * left and right shoulder, which the normalization below depends on; the rest
 * are elbows, wrists and hips in left/right pairs.
 */
export const LEFT_SHOULDER = POSE_START + 0;
export const RIGHT_SHOULDER = POSE_START + 1;
export const LEFT_WRIST = POSE_START + 4;
export const RIGHT_WRIST = POSE_START + 5;

/**
 * A hand missing for at least this fraction of a window is taken to be a hand
 * the sign does not use, and is zeroed rather than interpolated. Below it, the
 * gaps are tracking dropout and get filled in.
 */
export const UNUSED_HAND_NAN_THRESHOLD = 0.95;

/** Smallest shoulder width used as a scale, guarding degenerate frames. */
export const MIN_SHOULDER_SCALE = 1e-3;

/** Index of coordinate `c` of landmark `l` in frame `t` of a flat sequence. */
export function at(t: number, landmark: number, coord: number): number {
  return (t * NUM_LANDMARKS + landmark) * NUM_COORDS + coord;
}

/** An empty frame: every landmark absent. */
export function blankFrame(): Float64Array {
  return new Float64Array(NUM_LANDMARKS * NUM_COORDS).fill(NaN);
}

/**
 * Linear interpolation matching the reference implementation's, including its
 * edge behaviour: a query below the first sample takes the first value and one
 * above the last takes the last, rather than extrapolating.
 *
 * `xs` and `queries` are both ascending, so the search index only moves
 * forward. Written as one expression per branch because the arithmetic has to
 * match term for term, not just to within a tolerance.
 */
function interpolate(
  queries: Float64Array,
  xs: Float64Array,
  ys: Float64Array,
  out: (index: number, value: number) => void,
): void {
  const n = xs.length;
  let j = 0;
  for (let q = 0; q < queries.length; q++) {
    const x = queries[q];
    if (x < xs[0]) {
      out(q, ys[0]);
      continue;
    }
    if (x > xs[n - 1]) {
      out(q, ys[n - 1]);
      continue;
    }
    while (j + 1 < n && xs[j + 1] <= x) j++;
    if (j === n - 1 || xs[j] === x) {
      out(q, ys[j]);
      continue;
    }
    const slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
    out(q, slope * (x - xs[j]) + ys[j]);
  }
}

/**
 * Evenly spaced values from 0 to 1 inclusive, with the last one set exactly
 * rather than accumulated. The reference implementation's does the same, and
 * a final sample of 0.9999999999999999 would put the last output frame on the
 * interpolating branch instead of the exact one.
 */
function unitSteps(count: number): Float64Array {
  const out = new Float64Array(count);
  if (count === 1) return out;
  const step = 1 / (count - 1);
  for (let i = 0; i < count; i++) out[i] = i * step;
  out[count - 1] = 1;
  return out;
}

/**
 * Fill gaps along time for one span of landmarks, per landmark and coordinate.
 * A track that is absent for the whole window has nothing to interpolate
 * between and becomes zero.
 */
function interpolateTime(
  data: Float64Array,
  frameCount: number,
  startLandmark: number,
  endLandmark: number,
): void {
  const times = new Float64Array(frameCount);
  for (let t = 0; t < frameCount; t++) times[t] = t;
  const validTimes = new Float64Array(frameCount);
  const validValues = new Float64Array(frameCount);

  for (let l = startLandmark; l < endLandmark; l++) {
    for (let c = 0; c < NUM_COORDS; c++) {
      let valid = 0;
      for (let t = 0; t < frameCount; t++) {
        const value = data[at(t, l, c)];
        if (!Number.isNaN(value)) {
          validTimes[valid] = t;
          validValues[valid] = value;
          valid++;
        }
      }
      if (valid === 0) {
        for (let t = 0; t < frameCount; t++) data[at(t, l, c)] = 0;
      } else if (valid < frameCount) {
        const filled = new Float64Array(frameCount);
        interpolate(
          times,
          validTimes.subarray(0, valid),
          validValues.subarray(0, valid),
          (index, value) => {
            filled[index] = value;
          },
        );
        for (let t = 0; t < frameCount; t++) data[at(t, l, c)] = filled[t];
      }
    }
  }
}

/**
 * Decide, per hand, between "this sign does not use it" and "tracking dropped
 * out", and handle each accordingly. Returns the hands judged unused; their
 * landmarks are a placeholder until normalization has run.
 */
function fillMissing(data: Float64Array, frameCount: number): number[] {
  const unused: number[] = [];
  for (const start of [LEFT_HAND_START, RIGHT_HAND_START]) {
    const end = start + HAND_LANDMARKS;
    let missing = 0;
    for (let t = 0; t < frameCount; t++) {
      for (let l = start; l < end; l++) {
        // Presence is read off the x coordinate alone, as it is in the
        // reference implementation. A landmark with one coordinate tracked
        // and another not does not occur.
        if (Number.isNaN(data[at(t, l, 0)])) missing++;
      }
    }
    const fraction = frameCount > 0 ? missing / (frameCount * HAND_LANDMARKS) : 1;
    if (fraction >= UNUSED_HAND_NAN_THRESHOLD) {
      for (let t = 0; t < frameCount; t++) {
        for (let l = start; l < end; l++) {
          for (let c = 0; c < NUM_COORDS; c++) data[at(t, l, c)] = 0;
        }
      }
      unused.push(start);
    } else {
      interpolateTime(data, frameCount, start, end);
    }
  }
  interpolateTime(data, frameCount, POSE_START, NUM_LANDMARKS);
  return unused;
}

/**
 * Centre each frame on the shoulder midpoint and scale it by shoulder width,
 * which is what makes the tensor independent of where the signer stands and
 * how far from the camera. z is scaled but not translated.
 */
function normalize(data: Float64Array, frameCount: number): void {
  for (let t = 0; t < frameCount; t++) {
    const lx = data[at(t, LEFT_SHOULDER, 0)];
    const ly = data[at(t, LEFT_SHOULDER, 1)];
    const rx = data[at(t, RIGHT_SHOULDER, 0)];
    const ry = data[at(t, RIGHT_SHOULDER, 1)];
    const centerX = (lx + rx) / 2;
    const centerY = (ly + ry) / 2;
    const dx = lx - rx;
    const dy = ly - ry;
    let scale = Math.sqrt(dx * dx + dy * dy);
    if (scale < MIN_SHOULDER_SCALE) scale = MIN_SHOULDER_SCALE;
    for (let l = 0; l < NUM_LANDMARKS; l++) {
      data[at(t, l, 0)] = (data[at(t, l, 0)] - centerX) / scale;
      data[at(t, l, 1)] = (data[at(t, l, 1)] - centerY) / scale;
      data[at(t, l, 2)] = data[at(t, l, 2)] / scale;
    }
  }
}

/** Resample the time axis to TARGET_LEN, in place of nothing when it already is. */
function resample(data: Float64Array, frameCount: number): Float64Array {
  if (frameCount === TARGET_LEN) return data;
  const source = unitSteps(frameCount);
  const target = unitSteps(TARGET_LEN);
  const out = new Float64Array(TARGET_LEN * NUM_LANDMARKS * NUM_COORDS);
  const track = new Float64Array(frameCount);
  for (let l = 0; l < NUM_LANDMARKS; l++) {
    for (let c = 0; c < NUM_COORDS; c++) {
      for (let t = 0; t < frameCount; t++) track[t] = data[at(t, l, c)];
      interpolate(target, source, track, (index, value) => {
        out[at(index, l, c)] = value;
      });
    }
  }
  return out;
}

/**
 * A window of raw frames to the tensor the exported graph takes.
 *
 * `data` is (frameCount, 50, 3) with NaN for anything the landmarker did not
 * report, and is not modified. The result is (70, 50, 3).
 */
export function processSequence(data: Float64Array, frameCount: number): Float32Array {
  const expected = frameCount * NUM_LANDMARKS * NUM_COORDS;
  if (data.length !== expected) {
    throw new Error(`Expected ${expected} values for ${frameCount} frames, got ${data.length}.`);
  }
  if (frameCount < 1) throw new Error("A sequence needs at least one frame.");

  const working = Float64Array.from(data);
  const unused = fillMissing(working, frameCount);
  normalize(working, frameCount);
  // Re-zero unused hands after normalization, not before: normalization
  // subtracts a shared shoulder centre from every landmark, which would push
  // the placeholder off zero and turn an absent hand into a real coordinate.
  for (const start of unused) {
    for (let t = 0; t < frameCount; t++) {
      for (let l = start; l < start + HAND_LANDMARKS; l++) {
        for (let c = 0; c < NUM_COORDS; c++) working[at(t, l, c)] = 0;
      }
    }
  }
  return Float32Array.from(resample(working, frameCount));
}

/** Fraction of frames in a window where the given hand was tracked at all. */
export function trackedFraction(data: Float64Array, frameCount: number, start: number): number {
  if (frameCount === 0) return 0;
  let tracked = 0;
  for (let t = 0; t < frameCount; t++) {
    if (!Number.isNaN(data[at(t, start, 0)])) tracked++;
  }
  return tracked / frameCount;
}
