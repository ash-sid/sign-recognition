/**
 * web/src/classifier.ts
 *
 * Runs the exported graph and turns its scores into something worth showing.
 *
 * The graph takes the whole documented tensor, all fifty landmarks. Selecting
 * the hands and flattening them happen inside it, because which landmarks a
 * model was trained on is a property of that model rather than of
 * preprocessing. Nothing here should slice the tensor down; doing so would
 * feed the graph a shape it rejects, which is the point.
 *
 * Class names come from a list written out beside the checkpoint. Position i
 * in the output means whatever the label encoder called class i during
 * training, and the graph does not record that.
 *
 * Single-threaded on purpose. Threads would need the page served with the
 * headers that unlock shared memory, and at a millisecond or two per call
 * there is nothing to gain: the classifier is not what limits the frame rate.
 *
 * The plain WebAssembly build, not the default one. The default carries the
 * GPU-backed kernels as well, which doubles the runtime a visitor downloads to
 * no purpose here -- this graph runs in about a millisecond on a processor and
 * has nothing to offload. Served from the site rather than a third party, so
 * the page depends on nothing it does not ship.
 */
import * as ort from "onnxruntime-web/wasm";

import { NUM_COORDS, NUM_LANDMARKS, TARGET_LEN } from "./preprocessing";

export interface Candidate {
  label: string;
  probability: number;
}

export class Classifier {
  private constructor(
    private readonly runtime: ort.InferenceSession,
    private readonly labels: string[],
  ) {}

  static async create(graphUrl: string, labelsUrl: string): Promise<Classifier> {
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";

    const [graph, labels] = await Promise.all([
      fetch(graphUrl).then((r) => r.arrayBuffer()),
      fetch(labelsUrl).then((r) => r.json() as Promise<string[]>),
    ]);
    const runtime = await ort.InferenceSession.create(new Uint8Array(graph), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new Classifier(runtime, labels);
  }

  get vocabulary(): number {
    return this.labels.length;
  }

  name(index: number): string {
    return this.labels[index] ?? `class ${index}`;
  }

  /** Scores one tensor. Returns probabilities in the label list's order. */
  async probabilities(tensor: Float32Array): Promise<Float32Array> {
    const feeds = {
      landmarks: new ort.Tensor("float32", tensor, [1, TARGET_LEN, NUM_LANDMARKS, NUM_COORDS]),
    };
    const output = await this.runtime.run(feeds);
    return softmax(output.logits.data as Float32Array);
  }
}

export function softmax(logits: Float32Array): Float32Array {
  let largest = -Infinity;
  for (const value of logits) if (value > largest) largest = value;
  const out = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - largest);
    total += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

/**
 * Averages recent scores so a single odd frame cannot change the answer.
 *
 * Consecutive windows overlap almost completely, so their scores are not
 * independent evidence and averaging them is not the same as seeing the sign
 * several times. What it does is suppress the frame-to-frame flicker that
 * comes from one landmark jumping, which is worth doing on its own.
 */
export class Smoother {
  private state: Float32Array | undefined;

  constructor(private readonly weight: number) {}

  update(probabilities: Float32Array): Float32Array {
    if (!this.state || this.state.length !== probabilities.length) {
      this.state = Float32Array.from(probabilities);
      return this.state;
    }
    for (let i = 0; i < probabilities.length; i++) {
      this.state[i] += this.weight * (probabilities[i] - this.state[i]);
    }
    return this.state;
  }

  clear(): void {
    this.state = undefined;
  }
}

/** The highest-scoring classes, best first. */
export function ranked(
  probabilities: Float32Array,
  classifier: Classifier,
  count: number,
): Candidate[] {
  const order = Array.from(probabilities.keys()).sort((a, b) => probabilities[b] - probabilities[a]);
  return order.slice(0, count).map((index) => ({
    label: classifier.name(index),
    probability: probabilities[index],
  }));
}

/** Median of a list of timings, for a latency figure that ignores stalls. */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
