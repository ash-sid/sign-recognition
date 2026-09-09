/**
 * web/src/motion.ts
 *
 * Finds where a sign starts and stops in a continuous stream.
 *
 * The classifier was trained on sequences already trimmed to one sign each. A
 * camera does not trim anything, so scoring a sliding window asks the model a
 * question it was never asked during training: what sign is this two seconds
 * of video, most of which is somebody standing still. The answer drifts as the
 * window fills with the held pose that follows a sign, and it drifts toward
 * whichever class that pose resembles on its own.
 *
 * That drift is not the model being wrong. A diagnostic run measured a network
 * trained and evaluated on randomly permuted frames at 47.68% against 54.83%
 * ordered, so most of this task is solvable from the unordered collection of
 * poses a sequence contains and only a few points come from their order. Hand
 * a window dominated by one held pose to a model that leans that way and the
 * held pose is what it answers.
 *
 * So the stream is segmented by movement instead. Scoring runs while the hands
 * are moving, the scores across one burst of movement are averaged, and the
 * result is held after the hands stop or leave. Hands leaving the frame is the
 * clearest end-of-sign marker a webcam offers, and treating it as a reason to
 * discard the answer -- which is what this app did before -- throws the
 * answer away at the moment it becomes right.
 *
 * Averaging across a burst is not independent evidence. The windows overlap
 * almost entirely and they are views of one performance, not several. What it
 * buys is variance: windows early in a burst have only seen the start of the
 * sign, and averaging keeps any single one of them from deciding the answer.
 */
import { HAND_LANDMARKS, LEFT_HAND_START, NUM_COORDS, RIGHT_HAND_START, at } from "./preprocessing";

/**
 * Mean per-landmark movement between one captured frame and the next, in
 * shoulder-widths, above which the hands count as moving. Measured on
 * normalized coordinates so it does not depend on how close the signer stands.
 *
 * These two numbers are read off the panel, not derived from anything. Both
 * are shown live so a signer whose bursts never start, or never end, can see
 * which one is wrong instead of guessing.
 */
export const MOVING_THRESHOLD = 0.02;

/**
 * The threshold movement has to fall back below to end a burst. Set lower than
 * the one that starts it so that a moment of near-stillness inside a sign does
 * not chop it into two.
 */
export const STILL_THRESHOLD = 0.01;

/**
 * Frames a burst must last before its answer is worth committing. Counted in
 * frames actually scored, which is fewer than the frames a sign takes: scoring
 * only runs above the movement threshold.
 */
export const MIN_BURST_FRAMES = 3;

/**
 * Frames a burst stays open for once it starts, whatever movement does next.
 *
 * The two thresholds above have a gap between them so that a lull inside a
 * sign does not chop it in two, but a gap cannot help when the burst is closed
 * on the very next sample. A quick sign at thirty frames a second registers as
 * a spike rather than a plateau: one frame above the upper threshold, the next
 * already below the lower one. Without a floor that produces a one-frame
 * burst, which is discarded as too short, so signing quickly fails while
 * signing slowly works.
 *
 * Four frames is about an eighth of a second. Long enough to turn a spike into
 * something scorable, short enough that it cannot swallow a second sign.
 */
export const MIN_BURST_DURATION = 4;

export type Phase = "absent" | "still" | "moving";

/**
 * Movement between the last two frames of a window, averaged over the tracked
 * hand landmarks.
 *
 * Measured on the processed tensor rather than raw landmarks, so it is already
 * shoulder-relative and already has absent hands zeroed. Landmarks that are
 * exactly zero are skipped: a hand that appears or disappears between frames
 * would otherwise register as a whole-hand jump and read as motion.
 *
 * The tensor is resampled, so its frames are not captured frames: a 48-frame
 * window stretched to 70 spreads the same movement over half again as many
 * steps, and a threshold set against one window length would mean something
 * different at another. `scale` rescales the result back to movement per
 * captured frame, which is the quantity the thresholds are about.
 */
export function frameMovement(tensor: Float32Array, frameCount: number, scale = 1): number {
  if (frameCount < 2) return 0;
  const previous = frameCount - 2;
  const current = frameCount - 1;
  let total = 0;
  let counted = 0;
  for (const start of [LEFT_HAND_START, RIGHT_HAND_START]) {
    for (let l = start; l < start + HAND_LANDMARKS; l++) {
      let sum = 0;
      let present = false;
      for (let c = 0; c < NUM_COORDS; c++) {
        const a = tensor[at(previous, l, c)];
        const b = tensor[at(current, l, c)];
        if (a !== 0 || b !== 0) present = true;
        const d = b - a;
        sum += d * d;
      }
      if (!present) continue;
      total += Math.sqrt(sum);
      counted++;
    }
  }
  return counted > 0 ? (total / counted) * scale : 0;
}

/**
 * Accumulates scores over one burst of movement and reports the answer when
 * the burst ends.
 */
export class BurstTracker {
  private accumulated: Float32Array | undefined;
  private frames = 0;
  private moving = false;
  private completed = 0;
  private held = 0;

  /** True while a burst is in progress. */
  get active(): boolean {
    return this.moving;
  }

  /**
   * Frames scored: the running count during a burst, and the count the last
   * one finished with afterwards.
   *
   * A burst lasts a fraction of a second, so a panel showing only the live
   * count reads "not moving" by the time anyone looks at it -- which made the
   * one number deciding whether an answer appears the one number nobody could
   * read.
   */
  get length(): number {
    return this.moving ? this.frames : this.completed;
  }

  /**
   * Update the phase from this frame's movement. Returns whether scoring
   * should run: during a burst, and on the frame that starts one.
   */
  observe(movement: number, handsPresent: boolean): boolean {
    if (!handsPresent) return false;
    if (this.moving) {
      if (this.held > 0) this.held--;
      if (this.held === 0 && movement < STILL_THRESHOLD) {
        this.moving = false;
        this.completed = this.frames;
      }
    } else if (movement >= MOVING_THRESHOLD) {
      this.moving = true;
      this.held = MIN_BURST_DURATION;
      this.accumulated = undefined;
      this.frames = 0;
    }
    return this.moving;
  }

  add(probabilities: Float32Array): void {
    if (!this.accumulated || this.accumulated.length !== probabilities.length) {
      this.accumulated = new Float32Array(probabilities.length);
    }
    for (let i = 0; i < probabilities.length; i++) this.accumulated[i] += probabilities[i];
    this.frames++;
  }

  /**
   * The burst's averaged scores, or undefined if it was too short to trust.
   * Clears the accumulator either way, but keeps the frame count so the panel
   * can still say how short "too short" was.
   */
  take(): Float32Array | undefined {
    const accumulated = this.accumulated;
    const frames = this.frames;
    if (frames > 0) this.completed = frames;
    this.accumulated = undefined;
    this.frames = 0;
    this.moving = false;
    this.held = 0;
    if (!accumulated || frames < MIN_BURST_FRAMES) return undefined;
    const out = new Float32Array(accumulated.length);
    for (let i = 0; i < accumulated.length; i++) out[i] = accumulated[i] / frames;
    return out;
  }

  clear(): void {
    this.accumulated = undefined;
    this.frames = 0;
    this.moving = false;
    this.completed = 0;
    this.held = 0;
  }
}
