from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_JSON = MODEL_DIR / "hasil_perbandingan.json"
METADATA_JSON = MODEL_DIR / "metadata.json"

IMAGE_SIZE = (128, 128)
ORB_NFEATURES = 500
VOCAB_SIZE = 128

CLASS_LABELS = {
    0: "Belum Berbuah",
    1: "Belum Siap Panen",
    2: "Sebagian Siap Panen",
    3: "Siap Panen",
}

ESTIMATED_HARVEST = {
    0: "40-60 hari lagi",
    1: "20-30 hari lagi",
    2: "7-14 hari lagi",
    3: "Siap dipanen",
}

DESCRIPTIONS = {
    0: "Tanaman belum menghasilkan buah dan diperkirakan membutuhkan waktu sekitar 40-60 hari menuju masa panen.",
    1: "Buah masih didominasi warna hijau sehingga diperkirakan membutuhkan waktu 20-30 hari lagi.",
    2: "Buah memiliki campuran warna hijau, oranye, dan merah. Diperkirakan panen optimal dapat dilakukan dalam 7-14 hari.",
    3: "Buah didominasi warna merah dan telah memasuki fase siap panen.",
}


def calculate_color_ratio(image: np.ndarray) -> tuple[float, float, float]:
    resized = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    total_pixel = hue.shape[0] * hue.shape[1]

    green = np.sum((hue >= 35) & (hue <= 85))
    orange = np.sum((hue >= 10) & (hue < 20))
    red = np.sum(((hue >= 0) & (hue < 10)) | ((hue >= 170) & (hue <= 179)))

    return (
        round(float(green / total_pixel * 100), 2),
        round(float(orange / total_pixel * 100), 2),
        round(float(red / total_pixel * 100), 2),
    )


def get_harvest_analysis(
    prediction_id: int,
    green: float,
    orange: float,
    red: float,
) -> tuple[str, list[str], str]:
    if prediction_id == 0:
        return (
            "BELUM BERBUAH",
            [
                "Tanaman belum menunjukkan adanya buah cabai.",
                "Warna hijau pada citra didominasi oleh daun dan batang tanaman.",
                "Tanaman masih berada pada fase vegetatif.",
                "Berdasarkan ciri visual, tanaman belum memasuki fase generatif.",
            ],
            (
                "Tanaman masih berada pada fase vegetatif dan belum memasuki fase pembentukan buah.\n"
                "Lanjutkan pemeliharaan tanaman dan lakukan pengamatan berkala hingga memasuki fase:\n"
                "  - Mulai Berbuah (Generatif): sekitar 30-50 hari\n"
                "  - Perubahan Warna (Pematangan): sekitar 50-70 hari\n"
                "  - Siap Panen: sekitar 75-90 hari"
            ),
        )

    if prediction_id == 1:
        return (
            "BELUM SIAP PANEN",
            [
                "Buah cabai sudah muncul tetapi masih didominasi warna hijau.",
                f"Persentase warna hijau mencapai {green:.2f}%.",
                f"Warna oranye ({orange:.2f}%) dan merah ({red:.2f}%) masih sangat sedikit.",
                "Berdasarkan ciri visual, buah belum memasuki fase pemasakan secara optimal.",
            ],
            (
                "Buah cabai masih didominasi warna hijau sehingga belum mencapai kematangan optimal.\n"
                "Lanjutkan pemeliharaan dan lakukan pengamatan berkala hingga melewati fase:\n"
                "  - Perubahan Warna (Pematangan): sekitar 50-70 hari\n"
                "  - Siap Panen: sekitar 75-90 hari"
            ),
        )

    if prediction_id == 2:
        return (
            "SEBAGIAN SIAP PANEN",
            [
                "Sebagian buah telah mengalami perubahan warna dari hijau menuju oranye atau merah.",
                f"Persentase warna hijau: {green:.2f}%.",
                f"Persentase warna oranye: {orange:.2f}%.",
                f"Persentase warna merah: {red:.2f}%.",
                "Tanaman sedang berada pada fase pemasakan buah.",
            ],
            (
                "Sebagian buah telah mengalami perubahan warna dari hijau menuju merah.\n"
                "Lakukan pengamatan lebih intensif karena tanaman sedang berada pada fase pemasakan buah.\n"
                "Perkiraan fase yang sedang berlangsung:\n"
                "  - Perubahan Warna (Pematangan): sekitar 50-70 hari\n"
                "  - Menuju Siap Panen: sekitar 75-90 hari"
            ),
        )

    if prediction_id == 3:
        return (
            "SIAP PANEN",
            [
                "Sebagian besar buah telah berwarna merah.",
                f"Persentase warna merah mencapai {red:.2f}%.",
                "Buah telah mencapai tingkat kematangan optimal.",
                "Berdasarkan ciri visual, tanaman telah memasuki fase siap panen.",
            ],
            (
                "Sebagian besar buah telah mencapai tingkat kematangan optimal dan tanaman siap dipanen.\n"
                "Disarankan melakukan panen untuk memperoleh kualitas dan nilai jual yang optimal."
            ),
        )

    return "TIDAK DIKETAHUI", ["Kelas prediksi tidak dikenali."], ""


def load_optional_model(filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)


def load_first_available(*filenames: str):
    for filename in filenames:
        model = load_optional_model(filename)
        if model is not None:
            return model
    return None


def load_method_results():
    if not RESULTS_JSON.exists():
        return None
    with RESULTS_JSON.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_metadata():
    if not METADATA_JSON.exists():
        return None
    with METADATA_JSON.open("r", encoding="utf-8") as file:
        return json.load(file)


MODELS = {
    "svm": load_optional_model("svm_model.joblib"),
    "scaler": load_first_available("scaler_proposed.joblib", "scaler.joblib"),
    "pca": load_first_available("pca_proposed.joblib", "pca.joblib"),
    "vocabulary": load_optional_model("kmeans_vocab.joblib"),
    "method_results": load_method_results(),
    "metadata": load_model_metadata(),
}


def decode_image(file_storage) -> np.ndarray:
    image_bytes = file_storage.read()
    np_data = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File gambar tidak valid.")
    return image


def extract_bof_orb(image: np.ndarray, vocabulary) -> np.ndarray:
    cluster_count = int(getattr(vocabulary, "n_clusters", VOCAB_SIZE))
    histogram = np.zeros(cluster_count, dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    _, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        return histogram

    visual_words = vocabulary.predict(descriptors.astype(np.float32))
    for word_id in visual_words:
        histogram[int(word_id)] += 1.0

    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram.astype(np.float32)


def extract_hsv_histogram(
    image: np.ndarray,
    bins: tuple[int, int, int] = (16, 4, 8),
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256],
    )
    return cv2.normalize(histogram, histogram).flatten().astype(np.float32)


def extract_hsv_patch(patch: np.ndarray) -> np.ndarray:
    return extract_hsv_histogram(patch, bins=(8, 4, 4))


def extract_spatial_pyramid_hsv(image: np.ndarray) -> np.ndarray:
    features = []
    height, width = image.shape[:2]
    patch_dim = 128

    for level in range(3):
        grid_size = 2**level
        cell_height = height // grid_size
        cell_width = width // grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                patch = image[
                    row * cell_height : (row + 1) * cell_height,
                    col * cell_width : (col + 1) * cell_width,
                ]
                if patch.size == 0:
                    features.extend(np.zeros(patch_dim, dtype=np.float32))
                else:
                    features.extend(extract_hsv_patch(patch))

    return np.array(features, dtype=np.float32)


def extract_features(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, IMAGE_SIZE)
    bof = extract_bof_orb(resized, MODELS["vocabulary"])
    hsv = extract_hsv_histogram(resized)
    spm = extract_spatial_pyramid_hsv(resized)
    return np.concatenate([bof, hsv, spm]).reshape(1, -1)


def has_trained_model() -> bool:
    return all(
        MODELS[name] is not None
        for name in ("svm", "scaler", "pca", "vocabulary")
    )


def softmax_confidence(scores: np.ndarray) -> float:
    values = np.ravel(scores).astype(np.float64)
    values = values - np.max(values)
    probabilities = np.exp(values) / np.sum(np.exp(values))
    return float(np.max(probabilities) * 100.0)


def predict_with_model(image: np.ndarray) -> tuple[int, float]:
    features = extract_features(image)

    features = MODELS["scaler"].transform(features)
    features = MODELS["pca"].transform(features)

    svm = MODELS["svm"]
    prediction_id = int(svm.predict(features)[0])

    if hasattr(svm, "predict_proba"):
        confidence = float(np.max(svm.predict_proba(features)) * 100.0)
    elif hasattr(svm, "decision_function"):
        confidence = softmax_confidence(svm.decision_function(features))
    else:
        confidence = 90.0

    return prediction_id, confidence


def mask_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def predict_with_heuristic(image: np.ndarray) -> tuple[int, float]:
    resized = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    red_low = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_high = cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
    orange = cv2.inRange(hsv, (11, 70, 50), (28, 255, 255))
    green = cv2.inRange(hsv, (35, 45, 35), (90, 255, 255))

    red_ratio = mask_ratio(cv2.bitwise_or(red_low, red_high))
    orange_ratio = mask_ratio(orange)
    green_ratio = mask_ratio(green)
    warm_ratio = red_ratio + orange_ratio

    if warm_ratio < 0.005 and green_ratio < 0.08:
        prediction_id = 0
        confidence = 72.0
    elif red_ratio >= 0.08 and red_ratio >= orange_ratio:
        prediction_id = 3
        confidence = min(96.0, 78.0 + red_ratio * 120.0)
    elif warm_ratio >= 0.025 and green_ratio >= 0.08:
        prediction_id = 2
        confidence = min(94.0, 74.0 + warm_ratio * 120.0)
    elif green_ratio >= 0.08:
        prediction_id = 1
        confidence = min(93.0, 75.0 + green_ratio * 35.0)
    else:
        prediction_id = 0
        confidence = 70.0

    return prediction_id, confidence


def build_response(prediction_id: int, confidence: float, image: np.ndarray):
    prediction_id = prediction_id if prediction_id in CLASS_LABELS else 1
    green, orange, red = calculate_color_ratio(image)
    plant_status, reasons, recommendation = get_harvest_analysis(
        prediction_id,
        green,
        orange,
        red,
    )
    return {
        "prediction": CLASS_LABELS[prediction_id],
        "confidence": round(float(confidence), 1),
        "estimated_harvest": ESTIMATED_HARVEST[prediction_id],
        "description": DESCRIPTIONS[prediction_id],
        "green_percentage": green,
        "orange_percentage": orange,
        "red_percentage": red,
        "plant_status": plant_status,
        "reasons": reasons,
        "recommendation": recommendation,
    }


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": has_trained_model(),
            "method_results_loaded": RESULTS_JSON.exists(),
            "required_model_files": [
                "svm_model.joblib",
                "scaler_proposed.joblib atau scaler.joblib",
                "pca_proposed.joblib atau pca.joblib",
                "kmeans_vocab.joblib",
            ],
            "pipeline": {
                "image_size": IMAGE_SIZE,
                "orb_nfeatures": ORB_NFEATURES,
                "vocab_size": VOCAB_SIZE,
                "feature_order": "BoF + HSV + SPM",
            },
            "metadata": MODELS["metadata"],
        }
    )


@app.get("/method-comparison")
def method_comparison():
    method_results = load_method_results()
    if method_results is None:
        return jsonify(
            {
                "results": None,
                "error": "File hasil_perbandingan.json belum tersedia.",
                "required_file": str(RESULTS_JSON.name),
                "target_folder": str(MODEL_DIR),
                "export_hint": (
                    "Jalankan json.dump(all_results, file, indent=2) di Colab "
                    "setelah STEP 8, lalu salin hasil_perbandingan.json ke backend/models."
                ),
            }
        ), 404
    MODELS["method_results"] = method_results
    return jsonify(method_results)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Field multipart 'image' wajib diisi."}), 400

    try:
        image = decode_image(request.files["image"])

        if has_trained_model():
            prediction_id, confidence = predict_with_model(image)
        else:
            prediction_id, confidence = predict_with_heuristic(image)

        return jsonify(build_response(prediction_id, confidence, image))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
