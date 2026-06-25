from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).resolve().parent / "models" / "hasil_perbandingan.json"
REQUIRED_METHODS = ["BoF", "BoF+HSV", "BoF+HSV+SPM"]
REQUIRED_CLASSES = ["belumberbuah", "belumsiap", "sebagian", "siap"]
REQUIRED_METHOD_FIELDS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "per_class",
    "confusion_matrix",
    "waktu_training_s",
    "waktu_prediksi_ms",
    "pca_komponen",
    "dimensi_fitur",
]
REQUIRED_CLASS_FIELDS = ["precision", "recall", "f1_score", "support"]


def fail(message: str) -> int:
    print(f"FAILED: {message}")
    return 1


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_metric(value: Any, field_name: str) -> str | None:
    if not is_number(value):
        return f"{field_name} harus angka."
    if field_name in {"accuracy", "precision_macro", "recall_macro", "f1_macro"}:
        if not 0 <= float(value) <= 100:
            return f"{field_name} harus berada pada rentang 0 sampai 100."
    return None


def validate_per_class(method_name: str, per_class: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(per_class, dict):
        return [f"{method_name}.per_class harus object/dictionary."]

    for class_name in REQUIRED_CLASSES:
        class_result = per_class.get(class_name)
        if not isinstance(class_result, dict):
            errors.append(f"{method_name}.per_class.{class_name} tidak ditemukan.")
            continue

        for field_name in REQUIRED_CLASS_FIELDS:
            if field_name not in class_result:
                errors.append(f"{method_name}.per_class.{class_name}.{field_name} tidak ditemukan.")
                continue
            if field_name == "support":
                if not is_number(class_result[field_name]) or class_result[field_name] < 0:
                    errors.append(f"{method_name}.per_class.{class_name}.support harus angka >= 0.")
            else:
                metric_error = validate_metric(class_result[field_name], field_name)
                if metric_error:
                    errors.append(f"{method_name}.per_class.{class_name}.{metric_error}")

    return errors


def validate_confusion_matrix(method_name: str, matrix: Any) -> list[str]:
    if not isinstance(matrix, list) or len(matrix) != 4:
        return [f"{method_name}.confusion_matrix harus list 4 baris."]

    errors: list[str] = []
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 4:
            errors.append(f"{method_name}.confusion_matrix baris {row_index + 1} harus berisi 4 angka.")
            continue
        for col_index, value in enumerate(row):
            if not is_number(value) or int(value) < 0:
                errors.append(
                    f"{method_name}.confusion_matrix[{row_index}][{col_index}] harus angka >= 0."
                )

    return errors


def validate_method(method_name: str, payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{method_name} harus object/dictionary."]

    errors: list[str] = []
    for field_name in REQUIRED_METHOD_FIELDS:
        if field_name not in payload:
            errors.append(f"{method_name}.{field_name} tidak ditemukan.")

    for field_name in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        if field_name in payload:
            metric_error = validate_metric(payload[field_name], field_name)
            if metric_error:
                errors.append(f"{method_name}.{metric_error}")

    for field_name in ["waktu_training_s", "waktu_prediksi_ms", "dimensi_fitur"]:
        if field_name in payload and (not is_number(payload[field_name]) or payload[field_name] < 0):
            errors.append(f"{method_name}.{field_name} harus angka >= 0.")

    if "pca_komponen" in payload and not isinstance(payload["pca_komponen"], str):
        errors.append(f"{method_name}.pca_komponen harus string.")

    if "per_class" in payload:
        errors.extend(validate_per_class(method_name, payload["per_class"]))

    if "confusion_matrix" in payload:
        errors.extend(validate_confusion_matrix(method_name, payload["confusion_matrix"]))

    return errors


def main() -> int:
    if not RESULTS_PATH.exists():
        return fail(f"{RESULTS_PATH} belum ada.")

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, dict):
        return fail("Root hasil_perbandingan.json harus object/dictionary.")

    errors: list[str] = []
    for method_name in REQUIRED_METHODS:
        if method_name not in results:
            errors.append(f"Metode {method_name} tidak ditemukan.")
            continue
        errors.extend(validate_method(method_name, results[method_name]))

    if errors:
        for error in errors:
            print(f"FAILED: {error}")
        return 1

    print("Validasi hasil_perbandingan.json berhasil.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
