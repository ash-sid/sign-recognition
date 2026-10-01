/**
 * web/src/seam.ts
 *
 * Landmarks a folder of PNG frames with the live page's trackers and downloads
 * the result as JSON, so that the browser's extraction can be compared with
 * src/extract_landmarks.py over identical pixels.
 *
 * Identical pixels is the point. Comparing a live window with a separately
 * recorded clip would measure the video encoder and the frame alignment as
 * well as the extractor, with no way to tell the three apart. PNG is lossless,
 * so both sides can decode the same file to the same bytes -- and rather than
 * assume they do, each frame's RGB bytes are hashed here and in the Python
 * extractor, and the comparison refuses frames whose hashes differ.
 *
 * Decoding asks the browser not to colour-manage or premultiply, and frames
 * are handed to the landmarkers as ImageData so that what was hashed is what
 * was landmarked. Timestamps are round(i * 1000 / fps), the rule the Python
 * extractor uses, because video mode carries tracking and smoothing state
 * between frames and the timestamps are part of its input.
 *
 * Each frame records the slotted (50, 3) layout the live page builds, and also
 * what the landmarkers returned before any slot was chosen: hands in detection
 * order with their handedness label and score, and all 33 pose landmarks. The
 * raw detections let any slot rule be evaluated offline on both sides without
 * running the page again.
 *
 * NaN cannot be written in JSON, so untracked coordinates are written as null.
 */
import { TASKS_VERSION, createTrackers, trackFrame } from "./tracking";

const dom = {
  files: document.getElementById("files") as HTMLInputElement,
  clip: document.getElementById("clip") as HTMLInputElement,
  fps: document.getElementById("fps") as HTMLInputElement,
  delegate: document.getElementById("delegate") as HTMLSelectElement,
  run: document.getElementById("run") as HTMLButtonElement,
  log: document.getElementById("log") as HTMLPreElement,
};

function log(line: string): void {
  dom.log.textContent += `${line}\n`;
}

async function decode(file: File): Promise<ImageData> {
  const bitmap = await createImageBitmap(file, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("no 2D context");
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

/** sha256 of the RGB bytes, alpha dropped, matching the Python extractor. */
async function rgbSha256(image: ImageData): Promise<string> {
  const rgba = image.data;
  const rgb = new Uint8Array((rgba.length / 4) * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  const digest = await crypto.subtle.digest("SHA-256", rgb);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

function flat(landmarks: { x: number; y: number; z: number }[]): number[] {
  return landmarks.flatMap((p) => [p.x, p.y, p.z]);
}

function download(name: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function run(): Promise<void> {
  const files = Array.from(dom.files.files ?? []).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const fps = Number(dom.fps.value);
  const delegate = dom.delegate.value as "GPU" | "CPU";
  const clip = dom.clip.value.trim() || "clip";
  if (files.length === 0 || !(fps > 0)) {
    log("Choose the frames and a frame rate first.");
    return;
  }
  if (!globalThis.crypto?.subtle) {
    // Browsers expose hashing only to secure contexts: https, or localhost.
    log("Hashing is unavailable here. Open this page at localhost on the machine serving it.");
    return;
  }

  dom.log.textContent = "";
  log(`${files.length} frames, ${fps} per second, ${delegate} delegate`);
  // Fresh trackers for every run, so no tracking state carries over from a
  // previous clip.
  const trackers = await createTrackers(delegate);
  const frames = [];
  const counts = { left: 0, right: 0, pose: 0 };
  let size: [number, number] | undefined;
  try {
    for (let i = 0; i < files.length; i++) {
      const image = await decode(files[i]);
      if (!size) size = [image.width, image.height];
      else if (size[0] !== image.width || size[1] !== image.height) {
        throw new Error(`${files[i].name} is ${image.width}x${image.height}, not ${size[0]}x${size[1]}`);
      }
      const timestamp = Math.round((i * 1000) / fps);
      const sha256 = await rgbSha256(image);
      const tracked = trackFrame(trackers, image, timestamp);
      counts.left += Number(tracked.leftHandSeen);
      counts.right += Number(tracked.rightHandSeen);
      counts.pose += Number(tracked.poseSeen);
      frames.push({
        file: files[i].name,
        sha256,
        timestamp,
        landmarks: Array.from(tracked.frame, (v) => (Number.isNaN(v) ? null : v)),
        hands: tracked.detections.hands.map((hand, j) => ({
          landmarks: flat(hand),
          handedness: tracked.detections.handedness[j].label,
          score: tracked.detections.handedness[j].score,
        })),
        pose: tracked.detections.pose ? flat(tracked.detections.pose) : null,
      });
    }
  } finally {
    trackers.hands.close();
    trackers.pose.close();
  }

  log(`left hand ${counts.left}, right hand ${counts.right}, pose ${counts.pose} of ${files.length}`);
  const result = {
    clip,
    delegate,
    tasks_vision_version: TASKS_VERSION,
    user_agent: navigator.userAgent,
    fps,
    size,
    layout: "landmarks: contract (50, 3); hands: detection order, 21 x 3 each; pose: 33 x 3",
    frames,
  };
  const name = `seam_${clip}_${delegate.toLowerCase()}.json`;
  download(name, JSON.stringify(result));
  log(`downloaded ${name}`);
}

dom.run.addEventListener("click", () => {
  dom.run.disabled = true;
  run()
    .catch((error: unknown) => log(`failed: ${error instanceof Error ? error.message : String(error)}`))
    .finally(() => {
      dom.run.disabled = false;
    });
});
