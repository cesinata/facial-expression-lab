# Browser demo

A static page: webcam, MediaPipe Face Landmarker (finds the face box), then the ONNX model from
Study 3. Nothing is uploaded. The model file `model.onnx` is not in git. The Pages workflow
copies it from the latest GitHub release if one exists.

Run locally (needs a model file next to the page and a local web server, because browsers block
camera access on plain file URLs):

    cp path/to/model.onnx demo/model.onnx     # ignored by git
    python -m http.server 8000 --directory demo
    # open http://localhost:8000

The emoji mirror shows one emoji for the class with the highest smoothed probability. The smoothing window is a slider; the default of 5 frames was not tuned or tested. The plain probability bars stay on the page. The mirror is for fun and says nothing about how anyone feels.

Not measured yet: browser latency. The page shows a live median for your device. A number goes in
the README only after you measure it and commit the result (label: measured (browser)).

Limits to know about:
- The crop is resized with the browser canvas, not Pillow, so inputs differ slightly from training.
  The size of that effect has not been measured.
- The ONNX Runtime Web wasm build may not support every INT8 operator. Publish the FP32 model unless you
  have run the INT8 one in the browser.
