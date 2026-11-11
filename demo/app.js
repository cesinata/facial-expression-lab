// Webcam -> MediaPipe Face Landmarker (face box) -> 48x48 grey crop -> ONNX model in the browser.
// Library versions are pinned. Check them before release (see docs/decisions.md).
import { FaceLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"];
// One emoji per class, keyed by class name. A test keeps these keys equal to CLASSES.
const EMOJI = {
  neutral: "\u{1F610}", happiness: "\u{1F604}", surprise: "\u{1F62E}", sadness: "\u{1F622}",
  anger: "\u{1F620}", disgust: "\u{1F922}", fear: "\u{1F628}", contempt: "\u{1F60F}",
};
const NO_FACE = "\u2754";
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_DIR = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "model.onnx";
const MARGIN = 0.1; // keep equal to fel.landmarks.face_box

const $ = (id) => document.getElementById(id);
const video = $("video"), cropCanvas = $("crop"), ctx = cropCanvas.getContext("2d", { willReadFrequently: true });
const barsEl = $("bars");
const times = [];
let landmarker = null, session = null, running = false, lastVideoTime = -1;
let smoothed = null; // exponential moving average of the class probabilities

for (const c of CLASSES) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<span>${c}</span><div class="track"><div class="bar" id="bar-${c}"></div></div><span id="pct-${c}">0%</span>`;
  barsEl.appendChild(row);
}

function setStatus(msg) { $("status").textContent = msg; }

function softmax(v) {
  const m = Math.max(...v);
  const e = Array.from(v, (x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

async function init() {
  setStatus("Loading models...");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot set the headers that threads need
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  } catch (err) {
    $("notice").textContent = "No model file is published with this page yet (model.onnx). " +
      "The camera will start, but there is nothing to run.";
    console.warn(err);
  }
  const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);
  const base = { modelAssetPath: LANDMARKER_MODEL };
  const opts = { runningMode: "VIDEO", numFaces: 1 };
  try {
    landmarker = await FaceLandmarker.createFromOptions(fileset, { baseOptions: { ...base, delegate: "GPU" }, ...opts });
  } catch (err) {
    landmarker = await FaceLandmarker.createFromOptions(fileset, { baseOptions: base, ...opts });
  }
}

function faceBox(lms, w, h) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of lms) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  const mx = (x1 - x0) * MARGIN, my = (y1 - y0) * MARGIN;
  const bx0 = Math.max(0, x0 - mx) * w, by0 = Math.max(0, y0 - my) * h;
  const bx1 = Math.min(1, x1 + mx) * w, by1 = Math.min(1, y1 + my) * h;
  return [bx0, by0, bx1 - bx0, by1 - by0];
}

function cropToTensor() {
  const d = ctx.getImageData(0, 0, 48, 48).data;
  const out = new Float32Array(48 * 48);
  for (let i = 0; i < 48 * 48; i++) {
    out[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
  }
  return new ort.Tensor("float32", out, [1, 1, 48, 48]);
}

function updateMirror(probs) {
  if (!probs) { smoothed = null; $("mirror").textContent = NO_FACE; return; }
  const n = Number($("smooth").value);
  const alpha = 2 / (n + 1);
  smoothed = smoothed ? smoothed.map((s, i) => alpha * probs[i] + (1 - alpha) * s) : probs;
  $("mirror").textContent = EMOJI[CLASSES[smoothed.indexOf(Math.max(...smoothed))]];
}

$("smooth").addEventListener("input", () => { $("smoothval").textContent = $("smooth").value; });

function show(probs) {
  CLASSES.forEach((c, i) => {
    $("bar-" + c).style.width = (probs[i] * 100).toFixed(1) + "%";
    $("pct-" + c).textContent = (probs[i] * 100).toFixed(0) + "%";
  });
}

async function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = landmarker.detectForVideo(video, performance.now());
    if (res.faceLandmarks && res.faceLandmarks.length && session) {
      const [sx, sy, sw, sh] = faceBox(res.faceLandmarks[0], video.videoWidth, video.videoHeight);
      ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 48, 48);
      const t0 = performance.now();
      const out = await session.run({ input: cropToTensor() });
      times.push(performance.now() - t0);
      if (times.length > 30) times.shift();
      const sorted = [...times].sort((a, b) => a - b);
      $("latency").textContent = sorted[Math.floor(sorted.length / 2)].toFixed(1) + " ms";
      const probs = softmax(out.logits.data);
      show(probs);
      updateMirror(probs);
      setStatus("Face found.");
    } else if (session) {
      updateMirror(null);
      setStatus("No face found.");
    }
  }
  requestAnimationFrame(loop);
}

$("start").addEventListener("click", async () => {
  $("start").disabled = true;
  try {
    if (!landmarker) await init();
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    await video.play();
    running = true;
    setStatus("Running.");
    requestAnimationFrame(loop);
  } catch (err) {
    setStatus("Could not start: " + err.message);
    $("start").disabled = false;
  }
});
