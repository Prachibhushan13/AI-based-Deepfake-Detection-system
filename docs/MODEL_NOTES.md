# Model Notes

## Current model

The repository contains a PyTorch temporal model trained from cached Celeb-DF
features:

- **Visual backbone:** ImageNet-pretrained EfficientNet-B0 with its classifier
  removed, producing 1,280 features per frame.
- **Temporal head:** a 256-unit projection, two bidirectional LSTM layers
  (`128` and `64` hidden units), and a binary classifier.
- **Input:** 20 RGB frames resized to `224x224`, represented as float values in
  `[0, 1]` and normalized with the standard ImageNet mean and standard
  deviation.
- **Checkpoint:** `backend/artifacts/models/torch/best_checkpoint.pt`.
  Model artifacts are intentionally ignored by Git; place the checkpoint at
  this path locally or configure `TORCH_MODEL_PATH`.

The checkpoint's training configuration records 5,109 training rows and 902
validation rows, with a 20-epoch CPU training run. The saved validation loss
was `0.1535`. On the recorded 518-video held-out evaluation, the model
reported:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7181 |
| Precision | 0.7215 |
| Recall | 0.9294 |
| F1 | 0.8123 |
| ROC-AUC | 0.7073 |

These results are dataset-specific and should not be interpreted as a
production accuracy guarantee.

## Backend configuration

Set the following variables in `backend/.env` to use this model:

```dotenv
INFERENCE_BACKEND=torch
TORCH_MODEL_PATH=backend/artifacts/models/torch/best_checkpoint.pt
TORCH_TEMPORAL_SAMPLING=full_video
SEQUENCE_LENGTH=20
IMAGE_SIZE=224
```

With `full_video`, the video service decodes every frame and the inference
adapter deterministically selects up to 20 evenly spaced frames. The default
`legacy_stride` mode keeps the configured frame stride and then samples from
the decoded frames.

If the configured checkpoint cannot be loaded, the service logs the failure
and continues through its existing model fallback chain. Do not treat a
fallback result as equivalent to the trained temporal model; the API response
identifies the active mode in `modelMode`.

## Limitations and responsible use

- The temporal classifier was trained on cached EfficientNet features, so
  preprocessing, frame order, image normalization, and the EfficientNet
  weights must remain consistent with training.
- The recorded held-out set contains substantially more fake than real
  examples, and the confusion matrix shows many false positives. Thresholds
  should be calibrated for the intended deployment population.
- A result is an automated screening signal, not proof that media is
  authentic or manipulated. Use human review and preserve the original media
  when making consequential decisions.
- The model does not provide a validated audio, lip-sync, source-generator,
  or multi-face forensic assessment. Any additional fields exposed by the
  demo UI should be treated as experimental until backed by independently
  validated models.
