# Browser demo

A static page: webcam, MediaPipe Face Landmarker (finds the face box), then the ONNX model from
Study 3. Nothing is uploaded.

Live page: https://cesinata.github.io/facial-expression-lab/ (live after Pages is enabled).

## Where `model.onnx` comes from

The model file is not in git. It is published only as an attachment to the repository's GitHub
release, and only the FP32 file is published (6.12 MB). The Pages workflow
(`.github/workflows/pages.yml`) downloads `model.onnx` from the latest release on each run and
copies it next to the page. If no release exists, the page loads, says that no model file was
found, and runs nothing. The INT8 file is not published: it failed on PrivateTest (accuracy 0.047
against 0.705 for FP32, measured (host CPU), `results/study3_ferplus.json`). Licence and data
terms for the file are in `docs/release_notes.md`.

## Weakest classes

Per-class recall on FER2013 PrivateTest (measured (host CPU), `results/study3_ferplus.json`):
contempt 0.481, fear 0.516 and neutral 0.599 are the weakest. Expect the mirror to be wrong on
these more often. FER2013 has no subject IDs, so all FER2013 numbers are optimistic. On JAFFE,
which the model was not trained on, accuracy was 0.380 and recall was 0.000 for anger, disgust
and fear (measured (host CPU), `results/study3_ood_jaffe.json`).

## Run locally

Needs a model file next to the page and a local web server, because browsers block camera access
on plain file URLs:

    cp path/to/model.onnx demo/model.onnx     # ignored by git
    python -m http.server 8000 --directory demo
    # open http://localhost:8000

## Emoji mirror

The emoji mirror shows one emoji for the class with the highest smoothed probability. The
smoothing window is a slider; the default of 5 frames was not tuned or tested. The plain
probability bars stay on the page. The mirror is for fun and says nothing about how anyone feels.

## Libraries

Pinned exact versions, checked on 2026-11-15 (details in `docs/decisions.md`):
onnxruntime-web 1.31.1 and @mediapipe/tasks-vision 1.2.0, both from cdn.jsdelivr.net. The
MediaPipe face model is loaded from Google's model storage. Neither version has been run in a
browser by me yet.

## Not measured

Browser latency. The page shows a live median for your device. A number goes in the README only
after you measure it and commit the result (label: measured (browser)).

## Limits

- The crop is resized with the browser canvas, not Pillow, so inputs differ slightly from
  training. The size of that effect has not been measured.
- The page applies a plain softmax with no temperature scaling. Expected calibration error on
  PrivateTest was 0.288 without scaling and 0.232 with it (measured (host CPU)), so the bar
  percentages are not probabilities.
