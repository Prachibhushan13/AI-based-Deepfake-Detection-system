from pathlib import Path
import sys
import tensorflow as tf
import tf2onnx

# Add backend directory to sys.path to import app modules
sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings
from app.ml.model import build_cnn_lstm_model

def export_to_onnx():
    model_path = settings.model_artifact_path
    onnx_path = model_path.with_suffix('.onnx')

    print(f"Loading Keras model from {model_path}...")
    if model_path.exists():
        model = tf.keras.models.load_model(model_path)
    else:
        print("Keras model not found. Building untrained architecture for export...")
        model = build_cnn_lstm_model()

    print(f"Converting model to ONNX format...")
    # Get input shape from the model's first layer
    input_shape = model.inputs[0].shape
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    
    # Convert using tf2onnx
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(onnx_path))
    
    print(f"Successfully exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
