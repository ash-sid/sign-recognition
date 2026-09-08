/**
 * web/src/main.ts
 *
 * Camera in, sign out. Reads frames, keeps a sliding window of landmarks,
 * scores it, and reports what it took to do so.
 *
 * The timings on the panel are there because they decide what to fix. The
 * classifier costs a fraction of a millisecond against a frame budget of about
 * fifty; if this runs slowly it is the landmarkers, and no amount of work on
 * the model will show up.
 *
 * The demo declines to answer when it cannot see the hands, rather than
 * naming whichever of 250 signs a badly tracked window happens to resemble.
 * Tracking quality is worth about ten points of accuracy in the offline
 * evaluation -- more than any modelling decision measured -- so a confident
 * answer over an empty window would be the single most misleading thing this
 * page could show.
 */
import modelUrl from "../../models/abl_hands_aug.onnx?url";
import labelsUrl from "../../models/abl_hands_aug_labels.json?url";

import { FrameWindow } from "./buffer";
import { Classifier, Smoother, median, ranked } from "./classifier";
import { mirrorSequence } from "./mirror";
import {
  LEFT_HAND_START,
  LEFT_SHOULDER,
  NUM_COORDS,
  RIGHT_HAND_START,
  RIGHT_SHOULDER,
  TARGET_LEN,
  processSequence,
  trackedFraction,
} from "./preprocessing";
import { HAND_BONES, createTrackers, trackFrame, type Trackers } from "./tracking";

/**
 * Frames the classifier looks back over. Recorded sequences run a median of
 * about 22 frames and a 95th percentile near 135, so a window of about a
 * second and a half at typical webcam rates sits in the same range as the
 * sequences the model was trained on.
 */
const DEFAULT_WINDOW = 48;

/** Smallest score the demo will name a sign on. */
const DEFAULT_THRESHOLD = 0.35;

/**
 * Fraction of the window that must contain a tracked hand before the demo
 * will answer at all.
 */
const MIN_TRACKED = 0.5;

/** Weight on the newest scores when smoothing. */
const SMOOTHING = 0.35;

/** How many recent timings the medians on the panel are taken over. */
const TIMING_HISTORY = 60;

const CANDIDATES_SHOWN = 5;

const dom = {
  status: document.getElementById("status") as HTMLParagraphElement,
  video: document.getElementById("video") as HTMLVideoElement,
  overlay: document.getElementById("overlay") as HTMLCanvasElement,
  word: document.getElementById("word") as HTMLParagraphElement,
  confidence: document.getElementById("confidence-bar") as HTMLSpanElement,
  verdictNote: document.getElementById("verdict-note") as HTMLParagraphElement,
  candidates: document.getElementById("candidates") as HTMLOListElement,
  fps: document.getElementById("fps") as HTMLElement,
  landmarkMs: document.getElementById("landmark-ms") as HTMLElement,
  modelMs: document.getElementById("model-ms") as HTMLElement,
  windowFill: document.getElementById("window-fill") as HTMLElement,
  tracked: document.getElementById("tracked") as HTMLElement,
  windowLength: document.getElementById("window-length") as HTMLInputElement,
  windowLengthValue: document.getElementById("window-length-value") as HTMLOutputElement,
  threshold: document.getElementById("threshold") as HTMLInputElement,
  thresholdValue: document.getElementById("threshold-value") as HTMLOutputElement,
  reflect: document.getElementById("reflect") as HTMLInputElement,
};

const state = {
  frames: new FrameWindow(DEFAULT_WINDOW),
  smoother: new Smoother(SMOOTHING),
  threshold: DEFAULT_THRESHOLD,
  reflect: false,
  landmarkTimes: [] as number[],
  modelTimes: [] as number[],
  frameTimes: [] as number[],
  lastFrameAt: 0,
  lastTimestamp: 0,
  pending: false,
};

function setStatus(text: string, tone: "loading" | "live" | "blocked"): void {
  dom.status.textContent = text;
  dom.status.dataset.state = tone;
}

function record(history: number[], value: number): void {
  history.push(value);
  if (history.length > TIMING_HISTORY) history.shift();
}

async function startCamera(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  dom.video.srcObject = stream;
  if (dom.video.readyState < HTMLMediaElement.HAVE_METADATA) {
    await new Promise((resolve) => dom.video.addEventListener("loadedmetadata", resolve, { once: true }));
  }
  await dom.video.play();
  dom.overlay.width = dom.video.videoWidth;
  dom.overlay.height = dom.video.videoHeight;
}

function drawOverlay(frame: Float64Array, poseSeen: boolean): void {
  const context = dom.overlay.getContext("2d");
  if (!context) return;
  const { width, height } = dom.overlay;
  context.clearRect(0, 0, width, height);

  const point = (index: number): [number, number] | undefined => {
    const x = frame[index * NUM_COORDS];
    const y = frame[index * NUM_COORDS + 1];
    return Number.isNaN(x) || Number.isNaN(y) ? undefined : [x * width, y * height];
  };

  if (poseSeen) {
    const left = point(LEFT_SHOULDER);
    const right = point(RIGHT_SHOULDER);
    if (left && right) {
      context.strokeStyle = "rgba(244, 247, 247, 0.45)";
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(left[0], left[1]);
      context.lineTo(right[0], right[1]);
      context.stroke();
    }
  }

  context.lineWidth = 2.5;
  for (const start of [LEFT_HAND_START, RIGHT_HAND_START]) {
    context.strokeStyle = start === LEFT_HAND_START ? "#7fd8bf" : "#f0b264";
    context.beginPath();
    for (const [a, b] of HAND_BONES) {
      const from = point(start + a);
      const to = point(start + b);
      if (!from || !to) continue;
      context.moveTo(from[0], from[1]);
      context.lineTo(to[0], to[1]);
    }
    context.stroke();
  }
}

function showCandidates(entries: { label: string; probability: number }[]): void {
  dom.candidates.replaceChildren(
    ...entries.map((entry) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = entry.label;
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = `${(entry.probability * 100).toFixed(1)}%`;
      item.append(name, score);
      return item;
    }),
  );
}

function showNothing(note: string): void {
  dom.word.textContent = "—";
  dom.word.dataset.settled = "false";
  dom.confidence.style.width = "0%";
  dom.verdictNote.textContent = note;
}

function updateTelemetry(tracked: number): void {
  const frameTime = median(state.frameTimes);
  dom.fps.textContent = frameTime > 0 ? `${(1000 / frameTime).toFixed(0)} per second` : "—";
  dom.landmarkMs.textContent = `${median(state.landmarkTimes).toFixed(1)} ms`;
  dom.modelMs.textContent = state.modelTimes.length
    ? `${median(state.modelTimes).toFixed(2)} ms`
    : "—";
  dom.windowFill.textContent = `${state.frames.filled} of ${state.frames.capacity} frames`;
  dom.tracked.textContent = `${(tracked * 100).toFixed(0)}% of the window`;
  dom.tracked.dataset.poor = String(tracked < MIN_TRACKED);
}

async function step(trackers: Trackers, classifier: Classifier): Promise<void> {
  const now = performance.now();
  if (state.lastFrameAt) record(state.frameTimes, now - state.lastFrameAt);
  state.lastFrameAt = now;

  // The landmarkers reject a frame that looks older than the last one, and
  // two calls in the same millisecond can look exactly that way.
  const timestamp = Math.max(now, state.lastTimestamp + 1);
  state.lastTimestamp = timestamp;

  const landmarkStart = performance.now();
  const tracked = trackFrame(trackers, dom.video, timestamp);
  record(state.landmarkTimes, performance.now() - landmarkStart);

  state.frames.push(tracked.frame);
  drawOverlay(tracked.frame, tracked.poseSeen);

  const recent = state.frames.snapshot();
  const frameCount = state.frames.filled;
  const handPresence = Math.max(
    trackedFraction(recent, frameCount, LEFT_HAND_START),
    trackedFraction(recent, frameCount, RIGHT_HAND_START),
  );
  updateTelemetry(handPresence);

  if (!state.frames.full) {
    showNothing("Filling the window");
    return;
  }
  if (handPresence < MIN_TRACKED) {
    state.smoother.clear();
    dom.candidates.replaceChildren(emptyRow("Nothing to rank yet"));
    showNothing("Not enough of the hands is being tracked to call this");
    return;
  }

  let tensor = processSequence(recent, frameCount);
  if (state.reflect) tensor = mirrorSequence(tensor, TARGET_LEN);

  const modelStart = performance.now();
  const probabilities = await classifier.probabilities(tensor);
  record(state.modelTimes, performance.now() - modelStart);

  const smoothed = state.smoother.update(probabilities);
  const top = ranked(smoothed, classifier, CANDIDATES_SHOWN);
  showCandidates(top);

  const best = top[0];
  dom.confidence.style.width = `${Math.min(best.probability * 100, 100)}%`;
  if (best.probability >= state.threshold) {
    dom.word.textContent = best.label;
    dom.word.dataset.settled = "true";
    dom.verdictNote.textContent = `${(best.probability * 100).toFixed(0)}% confident`;
  } else {
    dom.word.textContent = "—";
    dom.word.dataset.settled = "false";
    dom.verdictNote.textContent = "No sign clear enough to call";
  }
}

function emptyRow(text: string): HTMLLIElement {
  const item = document.createElement("li");
  item.className = "empty";
  item.textContent = text;
  return item;
}

function bindControls(): void {
  dom.windowLength.value = String(DEFAULT_WINDOW);
  dom.windowLengthValue.textContent = `${DEFAULT_WINDOW} frames`;
  dom.windowLength.addEventListener("input", () => {
    const length = Number(dom.windowLength.value);
    dom.windowLengthValue.textContent = `${length} frames`;
    state.frames = new FrameWindow(length);
    state.smoother.clear();
  });

  dom.threshold.value = String(Math.round(DEFAULT_THRESHOLD * 100));
  dom.thresholdValue.textContent = `${Math.round(DEFAULT_THRESHOLD * 100)}%`;
  dom.threshold.addEventListener("input", () => {
    state.threshold = Number(dom.threshold.value) / 100;
    dom.thresholdValue.textContent = `${dom.threshold.value}%`;
  });

  dom.reflect.checked = false;
  dom.reflect.addEventListener("change", () => {
    state.reflect = dom.reflect.checked;
    state.smoother.clear();
  });
}

function loop(trackers: Trackers, classifier: Classifier): void {
  const tick = () => {
    if (!state.pending) {
      state.pending = true;
      void step(trackers, classifier).finally(() => {
        state.pending = false;
      });
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function main(): Promise<void> {
  bindControls();
  dom.candidates.replaceChildren(emptyRow("Nothing to rank yet"));

  try {
    setStatus("Loading the models", "loading");
    const [trackers, classifier] = await Promise.all([createTrackers(), loadClassifier()]);
    setStatus("Asking for the camera", "loading");
    await startCamera();
    setStatus(`Watching for ${classifier.vocabulary} signs`, "live");
    showNothing("Filling the window");
    loop(trackers, classifier);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus("Cannot start", "blocked");
    showNothing(message);
  }
}

function loadClassifier(): Promise<Classifier> {
  return Classifier.create(modelUrl, labelsUrl);
}

void main();
