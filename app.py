import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("flowers_prediction.h5")

CLASS = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

IMG_SIZE = 50


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image.convert("L")  # Convert to grayscale
    image_array = np.array(image)
    image_array = image_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    image_array = image_array / 255.0
    return image_array


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    try:
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)
        class_name = CLASS[class_index]
        confidence = prediction[0][class_index]

        response = {"class": class_name, "confidence": float(confidence)}
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
