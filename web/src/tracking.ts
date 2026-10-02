/**
 * web/src/tracking.ts
 *
 * Runs the landmarkers over one video frame and lays their output out in the
 * order the preprocessing contract specifies.
 *
 * Two landmarkers rather than the combined one, because the combined task also
 * runs a face model whose output this project deliberately does not use. Pose
 * is not optional even though the classifier only sees hands: normalization
 * divides by shoulder width, so the shoulders are needed to build the tensor
 * even though they are sliced away inside the graph.
 *
 * **Which hand goes in which slot** is the one decision here that can be wrong
 * without looking wrong. The landmarker reports handedness, but it decides it
 * on the assumption that the image is a mirror, and this app does not mirror
 * the image it hands over. Rather than track that assumption through every
 * change, each detected hand is matched to the nearer pose wrist. That is also
 * how the pipeline that produced the training data associated hands with
 * bodies, so it reproduces the dataset's slotting rather than approximating it.
 * Handedness is used only when the pose is missing, where it is inverted to
 * undo the mirror assumption, and a window without a pose has no usable
 * normalization anyway.
 */
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type ImageSource,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

import {
  HAND_LANDMARKS,
  LEFT_HAND_START,
  NUM_COORDS,
  POSE_START,
  RIGHT_HAND_START,
  blankFrame,
} from "./preprocessing";

/** Pinned so a rebuild months from now loads what was measured. */
export const TASKS_VERSION = "1.0.1";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/wasm`;
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

/**
 * Pose landmark indices in the order the contract stores them: left and right
 * shoulder, elbow, wrist, then hip. The first two carry the normalization.
 */
const POSE_SOURCE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24];
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;

/** Index of the wrist within a hand's own 21 landmarks. */
const HAND_WRIST = 0;

export interface Trackers {
  hands: HandLandmarker;
  pose: PoseLandmarker;
}

export interface TrackedFrame {
  /** One frame laid out as (50, 3), NaN where nothing was tracked. */
  frame: Float64Array;
  leftHandSeen: boolean;
  rightHandSeen: boolean;
  poseSeen: boolean;
  /** What the landmarkers returned before any slot was chosen. */
  detections: Detections;
}

export interface Detections {
  /** Hands in the order the landmarker reported them. */
  hands: NormalizedLandmark[][];
  /** The landmarker's top handedness label and its score, per hand. */
  handedness: { label: string; score: number }[];
  /** All 33 pose landmarks, or undefined when no pose was found. */
  pose: NormalizedLandmark[] | undefined;
}

/**
 * The live page runs on the GPU. The processor delegate exists so that
 * landmarks can be compared with the offline extractor, which has no GPU
 * delegate on Windows, with and without the delegate difference included.
 */
export async function createTrackers(delegate: "GPU" | "CPU" = "GPU"): Promise<Trackers> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
  const [hands, pose] = await Promise.all([
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
    }),
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate },
      runningMode: "VIDEO",
      numPoses: 1,
    }),
  ]);
  return { hands, pose };
}

function squaredDistance(a: NormalizedLandmark, b: NormalizedLandmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

/**
 * Decide which slot each detected hand belongs in. Ported to src/tracking.py
 * for the offline extractor; both are held to web/tests/fixtures/slots.json.
 *
 * With at most two hands there are only two assignments to compare, so the
 * cheapest total distance is found by evaluating both rather than by matching
 * each hand independently. Matching independently sends both hands to the same
 * slot whenever one is closer to the other's wrist, which happens whenever the
 * signer crosses their arms.
 */
export function assignSlots(
  hands: NormalizedLandmark[][],
  poseLandmarks: NormalizedLandmark[] | undefined,
  handedness: string[],
): (number | undefined)[] {
  if (hands.length === 0) return [undefined, undefined];

  if (poseLandmarks) {
    const left = poseLandmarks[POSE_LEFT_WRIST];
    const right = poseLandmarks[POSE_RIGHT_WRIST];
    if (left && right) {
      const cost = hands.map((hand) => ({
        left: squaredDistance(hand[HAND_WRIST], left),
        right: squaredDistance(hand[HAND_WRIST], right),
      }));
      if (hands.length === 1) {
        return [cost[0].left <= cost[0].right ? LEFT_HAND_START : RIGHT_HAND_START];
      }
      const straight = cost[0].left + cost[1].right;
      const crossed = cost[0].right + cost[1].left;
      return straight <= crossed
        ? [LEFT_HAND_START, RIGHT_HAND_START]
        : [RIGHT_HAND_START, LEFT_HAND_START];
    }
  }

  // No pose to match against. The reported handedness assumes a mirrored
  // image and this one is not mirrored, so it means the opposite hand.
  const slots = hands.map((_, i) =>
    handedness[i] === "Left" ? RIGHT_HAND_START : LEFT_HAND_START,
  );
  if (slots.length === 2 && slots[0] === slots[1]) {
    slots[1] = slots[0] === LEFT_HAND_START ? RIGHT_HAND_START : LEFT_HAND_START;
  }
  return slots;
}

function writeLandmark(frame: Float64Array, position: number, landmark: NormalizedLandmark): void {
  const base = position * NUM_COORDS;
  frame[base] = landmark.x;
  frame[base + 1] = landmark.y;
  frame[base + 2] = landmark.z;
}

/**
 * Landmark one video frame. `timestamp` must increase between calls; the
 * landmarkers reject a frame that appears to arrive out of order.
 */
export function trackFrame(
  trackers: Trackers,
  source: ImageSource,
  timestamp: number,
): TrackedFrame {
  const frame = blankFrame();
  const poseResult = trackers.pose.detectForVideo(source, timestamp);
  const handResult = trackers.hands.detectForVideo(source, timestamp);

  const poseLandmarks = poseResult.landmarks[0];
  if (poseLandmarks) {
    POSE_SOURCE_INDICES.forEach((source, offset) => {
      const landmark = poseLandmarks[source];
      if (landmark) writeLandmark(frame, POSE_START + offset, landmark);
    });
  }

  const reported = handResult.handedness.map((categories) => ({
    label: categories[0]?.categoryName ?? "",
    score: categories[0]?.score ?? Number.NaN,
  }));
  const handedness = reported.map((category) => category.label);
  const slots = assignSlots(handResult.landmarks, poseLandmarks, handedness);

  let leftHandSeen = false;
  let rightHandSeen = false;
  handResult.landmarks.forEach((hand, i) => {
    const slot = slots[i];
    if (slot === undefined) return;
    for (let j = 0; j < HAND_LANDMARKS; j++) {
      if (hand[j]) writeLandmark(frame, slot + j, hand[j]);
    }
    if (slot === LEFT_HAND_START) leftHandSeen = true;
    else rightHandSeen = true;
  });

  return {
    frame,
    leftHandSeen,
    rightHandSeen,
    poseSeen: Boolean(poseLandmarks),
    detections: { hands: handResult.landmarks, handedness: reported, pose: poseLandmarks },
  };
}

// --- sequence-level slot assignment --------------------------------------

/**
 * Largest mean landmark distance, in image widths, at which a hand in one
 * frame is taken to be the same hand as one in the frame before. 60 pixels at
 * the 640-pixel width the live page captures.
 */
export const CONTINUITY = 60 / 640;

export interface SequenceFrame {
  /** Detected hands in detection order. */
  hands: NormalizedLandmark[][];
  pose: NormalizedLandmark[] | undefined;
}

/** Mean distance over corresponding landmarks, in image widths. */
function meanDistance(a: NormalizedLandmark[], b: NormalizedLandmark[], aspect: number): number {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    const dx = a[i].x - b[i].x;
    const dy = (a[i].y - b[i].y) * aspect;
    total += Math.sqrt(dx * dx + dy * dy);
  }
  return total / a.length;
}

/** Pair this frame's hands with the previous frame's by least total distance. */
function pairHands(
  current: NormalizedLandmark[][],
  previous: NormalizedLandmark[][],
  aspect: number,
): [number, number, number][] {
  const n = current.length;
  const m = previous.length;
  if (n === 0 || m === 0) return [];
  if (n > 2 || m > 2) throw new Error("at most two hands per frame");
  const d = current.map((c) => previous.map((p) => meanDistance(c, p, aspect)));
  if (n === 1 && m === 1) return [[0, 0, d[0][0]]];
  if (n === 1) {
    const j = d[0][0] <= d[0][1] ? 0 : 1;
    return [[0, j, d[0][j]]];
  }
  if (m === 1) {
    const i = d[0][0] <= d[1][0] ? 0 : 1;
    return [[i, 0, d[i][0]]];
  }
  if (d[0][0] + d[1][1] <= d[0][1] + d[1][0]) {
    return [
      [0, 0, d[0][0]],
      [1, 1, d[1][1]],
    ];
  }
  return [
    [0, 1, d[0][1]],
    [1, 0, d[1][0]],
  ];
}

/** +1 on the left pose wrist, -1 on the right, 0 without a pose. */
function wristMargin(hand: NormalizedLandmark[], pose: NormalizedLandmark[] | undefined, aspect: number): number {
  if (!pose) return 0;
  const wrist = hand[HAND_WRIST];
  const lx = wrist.x - pose[POSE_LEFT_WRIST].x;
  const ly = (wrist.y - pose[POSE_LEFT_WRIST].y) * aspect;
  const rx = wrist.x - pose[POSE_RIGHT_WRIST].x;
  const ry = (wrist.y - pose[POSE_RIGHT_WRIST].y) * aspect;
  const left = lx * lx + ly * ly;
  const right = rx * rx + ry * ry;
  if (left + right === 0) return 0;
  return (right - left) / (right + left);
}

/**
 * Slot start index for every detected hand in every frame of a sequence, one
 * decision per hand for the whole sequence. Port of
 * src/tracking.py:assign_sequence_slots, which explains the rule; both are
 * held to web/tests/fixtures/sequence_slots.json and
 * sequence_slots_generated.json.
 *
 * Hands are linked frame to frame into tracks, and each track takes the slot
 * its summed wrist margin points to. Matching each frame on its own moves a
 * stationary hand between slots whenever the pose wrists are closer together
 * than their own error, as they are when the arms cross. Not yet used by the
 * live page, whose model was trained on frame-by-frame slotting.
 *
 * `aspect` is image height over width.
 */
export function assignSequenceSlots(frames: SequenceFrame[], aspect: number): number[][] {
  const ids: number[][] = [];
  const total: number[] = [];
  let previous: { hand: NormalizedLandmark[]; track: number }[] = [];
  for (const frame of frames) {
    const current: (number | undefined)[] = frame.hands.map(() => undefined);
    const pairs = pairHands(
      frame.hands,
      previous.map((p) => p.hand),
      aspect,
    );
    for (const [i, j, d] of pairs) {
      if (d < CONTINUITY) current[i] = previous[j].track;
    }
    const tracks = current.map((track) => {
      if (track !== undefined) return track;
      total.push(0);
      return total.length - 1;
    });
    frame.hands.forEach((hand, i) => {
      total[tracks[i]] += wristMargin(hand, frame.pose, aspect);
    });
    ids.push(tracks);
    previous = frame.hands.map((hand, i) => ({ hand, track: tracks[i] }));
  }

  const weaker = (a: number, b: number): number =>
    Math.abs(total[b]) < Math.abs(total[a]) || (Math.abs(total[b]) === Math.abs(total[a]) && b > a)
      ? b
      : a;
  const other = (slot: number): number => (slot === LEFT_HAND_START ? RIGHT_HAND_START : LEFT_HAND_START);

  const slot: number[] = total.map((t) => (t >= 0 ? LEFT_HAND_START : RIGHT_HAND_START));
  for (const tracks of ids) {
    if (tracks.length === 2 && slot[tracks[0]] === slot[tracks[1]]) {
      const moved = weaker(tracks[0], tracks[1]);
      slot[moved] = other(slot[moved]);
    }
  }

  return ids.map((tracks) => {
    const slots: number[] = tracks.map((t) => slot[t]);
    if (slots.length === 2 && slots[0] === slots[1]) {
      const i = weaker(tracks[0], tracks[1]) === tracks[0] ? 0 : 1;
      slots[i] = other(slots[i]);
    }
    return slots;
  });
}

/** Hand landmark pairs, for drawing a skeleton over the video. */
export const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
