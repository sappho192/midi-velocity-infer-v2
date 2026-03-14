"""Convert v2 .h5 model to ONNX format.

Run in a separate TF environment:
    python3 -m venv /tmp/tf_convert_env
    source /tmp/tf_convert_env/bin/activate
    pip install tensorflow tf2onnx
    python scripts/convert_v2_to_onnx.py
    deactivate
"""

from pathlib import Path


def main():
    import tensorflow as tf
    import tf2onnx

    ALPHA = 0.15

    def make_mse_cosine_loss(alpha):
        def mse_cosine_loss(y_true, y_pred):
            return alpha * (1 * tf.keras.losses.cosine_similarity(y_true, y_pred)) + (
                1 - alpha
            ) * tf.keras.losses.mse(y_true, y_pred)

        return mse_cosine_loss

    mse_cosine_loss = make_mse_cosine_loss(ALPHA)

    model_dir = Path(__file__).resolve().parent.parent.parent / "v2-attention" / "saved_models"
    h5_path = model_dir / "mvi-v2-2023-07-20_13-00_56-h4-e5-mse_cosine_loss-alpha0.15-m0.60-LSTM-luong_attention-MAESTRO.h5"
    onnx_path = model_dir / "mvi-v2-MAESTRO.onnx"

    print(f"Loading model from {h5_path}")
    custom_objects = {
        "mse_cosine_loss": mse_cosine_loss,
        "LeakyReLU": tf.keras.layers.LeakyReLU,
    }
    model = tf.keras.models.load_model(str(h5_path), custom_objects=custom_objects)
    model.summary()

    print(f"Converting to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=None)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Saved ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
