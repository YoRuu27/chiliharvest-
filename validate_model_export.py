from __future__ import annotations

import sys

import numpy as np

import app


EXPECTED_FEATURE_DIM = 3328
EXPECTED_CLASSES = [0, 1, 2, 3]
EXPECTED_CLUSTERS = 128


def main() -> int:
    if not app.has_trained_model():
        print("Model belum lengkap. Periksa file di folder backend/models.")
        return 1

    sample_image = np.zeros((128, 128, 3), dtype=np.uint8)
    feature_dim = app.extract_features(sample_image).shape[1]
    scaler_features = int(getattr(app.MODELS["scaler"], "n_features_in_", -1))
    pca_components = int(getattr(app.MODELS["pca"], "n_components_", -1))
    svm_classes = [int(value) for value in getattr(app.MODELS["svm"], "classes_", [])]
    kmeans_clusters = int(getattr(app.MODELS["vocabulary"], "n_clusters", -1))

    checks = [
        ("feature_dim", feature_dim, EXPECTED_FEATURE_DIM),
        ("scaler_features", scaler_features, EXPECTED_FEATURE_DIM),
        ("svm_classes", svm_classes, EXPECTED_CLASSES),
        ("kmeans_clusters", kmeans_clusters, EXPECTED_CLUSTERS),
    ]

    failed = False
    for name, actual, expected in checks:
        status = "OK" if actual == expected else "FAILED"
        print(f"{name}: {actual} ({status})")
        failed = failed or actual != expected

    print(f"pca_components: {pca_components} (OK)")
    print(f"metadata: {app.MODELS['metadata']}")

    if failed:
        print("Validasi model export gagal.")
        return 1

    print("Validasi model export berhasil.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
