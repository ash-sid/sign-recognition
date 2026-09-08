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
const TASKS_VERSION = "1.0.1";
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
}

export async function createTrackers(): Promise<Trackers> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
  const [hands, pose] = await Promise.all([
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    }),
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
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
 * Decide which slot each detected hand belongs in.
 *
 * With at most two hands there are only two assignments to compare, so the
 * cheapest total distance is found by evaluating both rather than by matching
 * each hand independently. Matching independently sends both hands to the same
 * slot whenever one is closer to the other's wrist, which happens whenever the
 * signer crosses their arms.
 */
function assignSlots(
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
  video: HTMLVideoElement,
  timestamp: number,
): TrackedFrame {
  const frame = blankFrame();
  const poseResult = trackers.pose.detectForVideo(video, timestamp);
  const handResult = trackers.hands.detectForVideo(video, timestamp);

  const poseLandmarks = poseResult.landmarks[0];
  if (poseLandmarks) {
    POSE_SOURCE_INDICES.forEach((source, offset) => {
      const landmark = poseLandmarks[source];
      if (landmark) writeLandmark(frame, POSE_START + offset, landmark);
    });
  }

  const handedness = handResult.handedness.map((categories) => categories[0]?.categoryName ?? "");
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

  return { frame, leftHandSeen, rightHandSeen, poseSeen: Boolean(poseLandmarks) };
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
