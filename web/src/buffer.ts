/**
 * web/src/buffer.ts
 *
 * The most recent frames, kept in one allocation.
 *
 * The classifier was trained on whole signs, each resampled to a fixed length
 * from however many frames it originally took. A camera does not hand over
 * whole signs, so the app scores a window that slides along and keeps the
 * length the resampler expects.
 *
 * How long the window is matters more than it looks. Resampling removes how
 * fast a sign was performed, but not how much of the window the sign occupies:
 * a two-second window around a half-second sign is mostly the signer standing
 * still, and that is not what any training sequence looked like. Recorded
 * sequences run a median of about 22 frames, so the default window is set near
 * the length of a sign rather than to the tensor's 70 frames, and most windows
 * are stretched to the target the same way most training sequences were.
 */
import { NUM_COORDS, NUM_LANDMARKS } from "./preprocessing";

const FRAME_SIZE = NUM_LANDMARKS * NUM_COORDS;

export class FrameWindow {
  private readonly data: Float64Array;
  private next = 0;
  private count = 0;

  constructor(readonly capacity: number) {
    if (capacity < 1) throw new Error("A window needs room for at least one frame.");
    this.data = new Float64Array(capacity * FRAME_SIZE);
  }

  /** How many frames the window is holding, up to its capacity. */
  get filled(): number {
    return this.count;
  }

  get full(): boolean {
    return this.count === this.capacity;
  }

  push(frame: Float64Array): void {
    if (frame.length !== FRAME_SIZE) {
      throw new Error(`Expected ${FRAME_SIZE} values per frame, got ${frame.length}.`);
    }
    this.data.set(frame, this.next * FRAME_SIZE);
    this.next = (this.next + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
  }

  /** The window in the order it was recorded, oldest first. */
  snapshot(): Float64Array {
    const out = new Float64Array(this.count * FRAME_SIZE);
    const oldest = this.count === this.capacity ? this.next : 0;
    for (let i = 0; i < this.count; i++) {
      const source = ((oldest + i) % this.capacity) * FRAME_SIZE;
      out.set(this.data.subarray(source, source + FRAME_SIZE), i * FRAME_SIZE);
    }
    return out;
  }

  clear(): void {
    this.next = 0;
    this.count = 0;
  }
}
