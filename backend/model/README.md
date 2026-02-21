# Model Folder

Place your trained Keras model here:

```
backend/model/
├── model.h5        ← REQUIRED: your trained Keras model
└── labels.json     ← OPTIONAL: class label names
```

## model.h5

Your model must be a Keras model saved in HDF5 format.
Compatible with TensorFlow 2.10.1 / Keras 2.10.0.

Save your model with:
```python
model.save("model.h5")
```

## labels.json

A JSON array of strings where the index corresponds to the class index output by your model.

Example for a 3-class ripeness classifier:
```json
["unripe", "ripe", "overripe"]
```

If `labels.json` is missing, predictions will use `"class_0"`, `"class_1"`, etc.

## Checking model input shape

```python
import keras
m = keras.models.load_model("model.h5", compile=False)
print("Input shape:", m.input_shape)
# Expected format: (None, height, width, channels)
# e.g. (None, 224, 224, 3) for a standard RGB model
m.summary()
```

The backend automatically reads `model.input_shape` and resizes uploaded
images to match. If the shape is `None` or dynamic, it falls back to 224×224
and logs a warning. You can override the default by setting
`DEFAULT_IMG_SIZE = (H, W)` in `app/utils.py`.
