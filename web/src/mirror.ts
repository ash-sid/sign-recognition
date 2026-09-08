/**
 * web/src/mirror.ts
 *
 * Reflects a processed sequence about the body midline. Port of the mirror
 * transform in src/augment.py.
 *
 * Only mirroring is ported. The training-time augmentation also warps the time
 * axis and jitters coordinates, and both were measured at no effect once the
 * epoch budget was held constant, so there is nothing here for them to
 * reproduce. Mirroring is not augmentation at this end anyway: it is the
 * transform that converts between a camera the signer faces and a camera that
 * shows them their reflection, and the model does not answer the same way for
 * both.
 *
 * A reflection is two changes, not one. Negating x alone leaves the signer's
 * left hand in the left-hand slot while it is physically on the right, which is
 * an input no signer can produce; the hand blocks and each left/right pose pair
 * have to trade places as well.
 *
 * Valid only on normalized coordinates, where x = 0 is the body midline. On raw
 * landmarker output x runs from 0 to 1 across the image and negating it puts
 * the signer outside the frame.
 */
import { HANDS_END, HAND_LANDMARKS, NUM_COORDS, NUM_LANDMARKS, at } from "./preprocessing";

/** Landmark ordering after a left/right swap. */
export function mirrorPermutation(): Int32Array {
  const order = new Int32Array(NUM_LANDMARKS);
  for (let i = 0; i < NUM_LANDMARKS; i++) order[i] = i;
  for (let i = 0; i < HAND_LANDMARKS; i++) {
    order[i] = HAND_LANDMARKS + i;
    order[HAND_LANDMARKS + i] = i;
  }
  for (let i = HANDS_END; i < NUM_LANDMARKS - 1; i += 2) {
    order[i] = i + 1;
    order[i + 1] = i;
  }
  return order;
}

const PERMUTATION = mirrorPermutation();

/** Reflect a (frameCount, 50, 3) processed sequence. The input is not modified. */
export function mirrorSequence(data: Float32Array, frameCount: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let t = 0; t < frameCount; t++) {
    for (let l = 0; l < NUM_LANDMARKS; l++) {
      const source = PERMUTATION[l];
      out[at(t, l, 0)] = -data[at(t, source, 0)];
      for (let c = 1; c < NUM_COORDS; c++) out[at(t, l, c)] = data[at(t, source, c)];
    }
  }
  return out;
}
