import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # supaya bisa menyimpan grafik tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from src.preprocess import (
    FEATURE_LABELS_ID,
    FEATURES,
    TARGET_NAMES,
    encode,
    load_and_clean,
    split_xy,
)

# ------------------------------------------------------------------
# Konfigurasi
# ------------------------------------------------------------------
DATA_PATH = "data/loan_approval_dataset.csv"
MODEL_DIR = "model"
IMG_DIR = "static/img"
RANDOM_STATE = 42

N_ESTIMATORS_LIST = [10, 50, 100, 200, 500]
MAX_DEPTH_LIST = [None, 3, 5, 10, 20]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def depth_label(d):
    return "None" if d is None else str(d)


# ------------------------------------------------------------------
# 1. Persiapan data
# ------------------------------------------------------------------
print("=" * 60)
print("1. PERSIAPAN DATA")
print("=" * 60)

df = load_and_clean(DATA_PATH)
print(f"Jumlah data setelah dibersihkan : {df.shape[0]} baris, {df.shape[1]} kolom")

# Analisis awal: cek ketidakseimbangan kelas
dist = df["loan_status"].value_counts()
print("\nDistribusi kelas target:")
for label, jumlah in dist.items():
    print(f"  {label:<9}: {jumlah} ({jumlah / len(df):.1%})")
rasio = dist.max() / dist.min()
print(f"Rasio mayoritas : minoritas = {rasio:.2f} : 1")
print("-> Ketidakseimbangan ringan, masih wajar. Tetap pakai stratify saat split.")

df_enc = encode(df)
X, y = split_xy(df_enc)

# stratify=y menjaga proporsi Approved/Rejected sama di train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nData latih : {len(X_train)} | Data uji : {len(X_test)}")

# Grafik distribusi kelas
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=df["loan_status"], hue=df["loan_status"], palette=["#2e7d32", "#c62828"], legend=False)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.title("Distribusi Status Pinjaman")
plt.xlabel("Status")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/distribusi_kelas.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 2. Skenario A: pengaruh jumlah pohon (n_estimators)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. SKENARIO A - PENGARUH n_estimators")
print("=" * 60)

hasil_a = []
for n in N_ESTIMATORS_LIST:
    model = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))
    # Cross-validation 5-fold di data latih -> ukuran stabilitas tambahan
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train, cv=5, scoring="accuracy",
    )
    hasil_a.append({
        "n_estimators": n,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "cv_mean": cv.mean(),
        "cv_std": cv.std(),
    })
    print(f"n_estimators={n:<4} | train={acc_train:.4f} | test={acc_test:.4f} "
          f"| CV={cv.mean():.4f} ± {cv.std():.4f}")

df_a = pd.DataFrame(hasil_a)

# Pilih n_estimators terbaik: akurasi test tertinggi,
# jika seri pilih jumlah pohon paling sedikit (lebih ringan & cepat).
best_acc_a = df_a["acc_test"].max()
best_n = int(df_a[df_a["acc_test"] == best_acc_a]["n_estimators"].min())
print(f"\n>> n_estimators terbaik: {best_n} (akurasi test {best_acc_a:.4f})")

plt.figure(figsize=(8, 5))
plt.plot(df_a["n_estimators"], df_a["acc_test"], marker="o", linewidth=2, label="Akurasi Testing")
plt.plot(df_a["n_estimators"], df_a["cv_mean"], marker="s", linestyle="--", label="Akurasi CV 5-fold (train)")
for x_, y_ in zip(df_a["n_estimators"], df_a["acc_test"]):
    plt.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
plt.xscale("log")
plt.xticks(N_ESTIMATORS_LIST, [str(n) for n in N_ESTIMATORS_LIST])
plt.xlabel("Jumlah Pohon (n_estimators)")
plt.ylabel("Akurasi")
plt.title("Skenario A: Pengaruh Jumlah Pohon terhadap Akurasi")
plt.legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/skenario_a.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 3. Skenario B: pengaruh kedalaman pohon (max_depth)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"3. SKENARIO B - PENGARUH max_depth (n_estimators={best_n})")
print("=" * 60)

hasil_b = []
for d in MAX_DEPTH_LIST:
    model = RandomForestClassifier(
        n_estimators=best_n, max_depth=d, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))
    gap = acc_train - acc_test
    hasil_b.append({
        "max_depth": depth_label(d),
        "acc_train": acc_train,
        "acc_test": acc_test,
        "gap": gap,
    })
    print(f"max_depth={depth_label(d):<5} | train={acc_train:.4f} | test={acc_test:.4f} | selisih={gap:.4f}")

df_b = pd.DataFrame(hasil_b)

# Urutkan dari dangkal ke dalam agar grafik mudah dibaca
urutan = ["3", "5", "10", "20", "None"]
df_b_plot = df_b.set_index("max_depth").loc[urutan].reset_index()

# Pilih max_depth terbaik: akurasi test tertinggi,
# jika seri pilih pohon yang lebih dangkal (lebih sederhana, lebih tahan overfitting).
best_acc_b = df_b_plot["acc_test"].max()
best_depth_label = df_b_plot[df_b_plot["acc_test"] == best_acc_b]["max_depth"].iloc[0]
best_depth = None if best_depth_label == "None" else int(best_depth_label)
print(f"\n>> max_depth terbaik: {best_depth_label} (akurasi test {best_acc_b:.4f})")

x_pos = np.arange(len(urutan))
lebar = 0.38
plt.figure(figsize=(8, 5))
plt.bar(x_pos - lebar / 2, df_b_plot["acc_train"], lebar, label="Akurasi Training", color="#1565c0")
plt.bar(x_pos + lebar / 2, df_b_plot["acc_test"], lebar, label="Akurasi Testing", color="#ef6c00")
for i, row in df_b_plot.iterrows():
    plt.text(i - lebar / 2, row["acc_train"] + 0.003, f"{row['acc_train']:.3f}", ha="center", fontsize=8)
    plt.text(i + lebar / 2, row["acc_test"] + 0.003, f"{row['acc_test']:.3f}", ha="center", fontsize=8)
plt.xticks(x_pos, urutan)
plt.ylim(max(0.0, df_b_plot[["acc_train", "acc_test"]].min().min() - 0.05), 1.02)
plt.xlabel("Kedalaman Pohon (max_depth)")
plt.ylabel("Akurasi")
plt.title(f"Skenario B: Training vs Testing (n_estimators={best_n})")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/skenario_b.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 4. Model terbaik + Feature Importance
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. MODEL TERBAIK & FEATURE IMPORTANCE")
print("=" * 60)

best_model = RandomForestClassifier(
    n_estimators=best_n, max_depth=best_depth, random_state=RANDOM_STATE, n_jobs=-1
)
best_model.fit(X_train, y_train)

importances = pd.DataFrame({
    "feature": FEATURES,
    "label": [FEATURE_LABELS_ID[f] for f in FEATURES],
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
print(importances[["feature", "importance"]].to_string(index=False))

top10 = importances.head(10)
plt.figure(figsize=(9, 6))
sns.barplot(data=top10, x="importance", y="label", hue="label", palette="viridis", legend=False)
for i, v in enumerate(top10["importance"]):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.xlabel("Tingkat Kepentingan (Mean Decrease in Impurity)")
plt.ylabel("Fitur")
plt.title("10 Fitur Paling Berpengaruh - Random Forest")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/feature_importance.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 5. Evaluasi akhir
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. EVALUASI AKHIR MODEL TERBAIK")
print("=" * 60)

y_pred = best_model.predict(X_test)
acc_final = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
report_text = classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=4)
report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print(f"Konfigurasi : n_estimators={best_n}, max_depth={best_depth_label}")
print(f"Akurasi     : {acc_final:.4f}")
print(f"F1-macro    : {f1_macro:.4f}\n")
print(report_text)
print("Confusion Matrix (baris=aktual, kolom=prediksi) [Rejected, Approved]:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Model Terbaik")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/confusion_matrix.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 6. Simpan model & metadata
# ------------------------------------------------------------------
# Latih ulang dengan SELURUH data (train + test) untuk model produksi?
# Tidak. Kita simpan model yang sudah dievaluasi agar angka di laporan
# sama persis dengan model yang dipakai di web.
joblib.dump(best_model, f"{MODEL_DIR}/model.joblib", compress=3)

# Rentang nilai tiap fitur -> dipakai untuk validasi input di web
ranges = {
    f: {"min": float(df_enc[f].min()), "max": float(df_enc[f].max()),
        "median": float(df_enc[f].median())}
    for f in FEATURES
}

metadata = {
    "best_params": {"n_estimators": best_n, "max_depth": best_depth_label},
    "metrics": {
        "accuracy": acc_final,
        "f1_macro": f1_macro,
        "report": report_dict,
        "confusion_matrix": cm.tolist(),
    },
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "class_distribution": {k: int(v) for k, v in dist.items()},
    "scenario_a": df_a.to_dict(orient="records"),
    "scenario_b": df_b_plot.to_dict(orient="records"),
    "feature_importance": importances.to_dict(orient="records"),
    "feature_ranges": ranges,
}
with open(f"{MODEL_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\nSelesai! File yang dihasilkan:")
print(f"  - {MODEL_DIR}/model.joblib")
print(f"  - {MODEL_DIR}/metadata.json")
print(f"  - {IMG_DIR}/*.png (5 grafik)")
