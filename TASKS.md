# Tasks

- [x] Investigate if `classic.py` can be GPU accelerated
- [x] Fix this warning

```
/home/masimms/code/aiml-course-capstone/.venv/lib/python3.12/site-packages/keras/src/layers/layer.py:1039: UserWarning: Layer 'lstm' (of type LSTM) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(
```

- [x] Save the chosen weights out to be used at inference time
- [x] Update the runner training/tuning scripts to fail if any uncomitted git changes, add git hash to the output logs and summary
- [ ] Add OpenTelemetry logging and a local self-contained otel docker-compose instance to visualize
- [ ] Reimplement with PyTorch
- [x] Configure tensorflow to lazy load
- [ ] Do the capstone technical and non-technical writeups
