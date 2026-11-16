import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.preprocess import (
    EDUCATION_MAP,
    FEATURE_LABELS_ID,
    FEATURES,
    SELF_EMPLOYED_MAP,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")

app = Flask(__name__)

# Model & metadata dimuat SEKALI saat aplikasi start (bukan setiap request)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model belum ada. Jalankan dulu: python train.py")
model = joblib.load(MODEL_PATH)
with open(META_PATH, encoding="utf-8") as f:
    META = json.load(f)
RANGES = META["feature_ranges"]

# Field angka beserta batas mutlak (input di luar batas ini ditolak)
NUMERIC_FIELDS = {
    "no_of_dependents": {"min": 0, "max": 20, "integer": True},
    "income_annum": {"min": 0, "max": None, "integer": False},
    "loan_amount": {"min": 1, "max": None, "integer": False},
    "loan_term": {"min": 1, "max": 40, "integer": True},
    "cibil_score": {"min": 300, "max": 900, "integer": True},
    "residential_assets_value": {"min": 0, "max": None, "integer": False},
    "commercial_assets_value": {"min": 0, "max": None, "integer": False},
    "luxury_assets_value": {"min": 0, "max": None, "integer": False},
    "bank_asset_value": {"min": 0, "max": None, "integer": False},
}


def parse_input(data):
    """
    Mengubah input form/JSON menjadi satu baris DataFrame siap prediksi.
    Mengembalikan (row_df, errors, warnings, values).
    """
    errors, warnings, values = [], [], {}

    # --- Kategori ---
    edu = str(data.get("education", "")).strip()
    if edu not in EDUCATION_MAP:
        errors.append("Pendidikan harus 'Graduate' atau 'Not Graduate'.")
    se = str(data.get("self_employed", "")).strip()
    if se not in SELF_EMPLOYED_MAP:
        errors.append("Status wiraswasta harus 'Yes' atau 'No'.")
    values["education"] = edu
    values["self_employed"] = se

    # --- Angka ---
    for field, rule in NUMERIC_FIELDS.items():
        label = FEATURE_LABELS_ID[field]
        raw = str(data.get(field, "")).replace(".", "").replace(",", "").strip()
        values[field] = raw
        if raw == "":
            errors.append(f"{label} wajib diisi.")
            continue
        try:
            num = float(raw)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        if rule["integer"] and not num.is_integer():
            errors.append(f"{label} harus bilangan bulat.")
            continue
        if num < rule["min"] or (rule["max"] is not None and num > rule["max"]):
            batas = f"{rule['min']}–{rule['max']}" if rule["max"] is not None else f"≥ {rule['min']}"
            errors.append(f"{label} harus dalam rentang {batas}.")
            continue
        values[field] = num
        # Peringatan jika di luar rentang data latih (model tidak pernah melihatnya)
        r = RANGES[field]
        if num < r["min"] or num > r["max"]:
            warnings.append(
                f"{label} ({num:,.0f}) di luar rentang data latih "
                f"({r['min']:,.0f}–{r['max']:,.0f}); prediksi bisa kurang andal."
            )

    if errors:
        return None, errors, warnings, values

    row = {f: values[f] for f in NUMERIC_FIELDS}
    row["education"] = EDUCATION_MAP[edu]
    row["self_employed"] = SELF_EMPLOYED_MAP[se]
    row_df = pd.DataFrame([row])[FEATURES]  # urutan kolom WAJIB sama dengan saat training
    return row_df, [], warnings, values


def predict(row_df):
    proba = model.predict_proba(row_df)[0]
    # model.classes_ = [0, 1] -> 0=Rejected, 1=Approved
    p_approved = float(proba[list(model.classes_).index(1)])
    status = "Approved" if p_approved >= 0.5 else "Rejected"
    return {
        "status": status,
        "status_id": "DISETUJUI" if status == "Approved" else "DITOLAK",
        "prob_approved": round(p_approved * 100, 2),
        "prob_rejected": round((1 - p_approved) * 100, 2),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, warnings, values = None, [], [], {}
    if request.method == "POST":
        row_df, errors, warnings, values = parse_input(request.form)
        if not errors:
            result = predict(row_df)
            # Rasio pinjaman terhadap pendapatan - informasi tambahan untuk pengguna
            if values["income_annum"] > 0:
                result["loan_to_income"] = round(values["loan_amount"] / values["income_annum"], 2)
    return render_template(
        "index.html",
        result=result,
        errors=errors,
        warnings=warnings,
        values=values,
        labels=FEATURE_LABELS_ID,
        meta=META,
    )


@app.route("/eksperimen")
def eksperimen():
    return render_template("eksperimen.html", meta=META, labels=FEATURE_LABELS_ID)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    row_df, errors, warnings, _ = parse_input(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400
    return jsonify({"success": True, "warnings": warnings, **predict(row_df)})


if __name__ == "__main__":
    app.run(debug=True)
