import pandas as pd

# Urutan fitur yang dipakai model (urutan ini WAJIB sama saat training & prediksi)
FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

TARGET = "loan_status"

# Label Encoding manual (lebih jelas daripada LabelEncoder untuk pemula)
EDUCATION_MAP = {"Graduate": 1, "Not Graduate": 0}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}
TARGET_MAP = {"Approved": 1, "Rejected": 0}
TARGET_NAMES = ["Rejected", "Approved"]  # indeks 0 dan 1

# Nama fitur dalam Bahasa Indonesia (untuk grafik & tampilan web)
FEATURE_LABELS_ID = {
    "no_of_dependents": "Jumlah Tanggungan",
    "education": "Pendidikan",
    "self_employed": "Wiraswasta",
    "income_annum": "Pendapatan Tahunan",
    "loan_amount": "Jumlah Pinjaman",
    "loan_term": "Tenor Pinjaman",
    "cibil_score": "Skor Kredit (CIBIL)",
    "residential_assets_value": "Aset Rumah Tinggal",
    "commercial_assets_value": "Aset Komersial",
    "luxury_assets_value": "Aset Mewah",
    "bank_asset_value": "Aset di Bank",
}


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Membaca CSV lalu membersihkan masalah yang ada di dataset asli."""
    df = pd.read_csv(csv_path)

    # 1) Nama kolom di file asli diawali spasi, contoh: ' loan_status'
    df.columns = df.columns.str.strip()

    # 2) Nilai teks juga diawali spasi, contoh: ' Approved'
    for col in ["education", "self_employed", TARGET]:
        df[col] = df[col].astype(str).str.strip()

    # 3) loan_id hanya nomor urut, tidak punya makna prediktif -> dibuang
    df = df.drop(columns=["loan_id"])

    # 4) Ada nilai aset rumah negatif (-100000). Nilai aset tidak mungkin negatif,
    #    jadi dianggap salah input dan diganti 0.
    df["residential_assets_value"] = df["residential_assets_value"].clip(lower=0)

    # 5) Hapus baris duplikat (jika ada)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah kolom kategori menjadi angka (Label Encoding)."""
    df = df.copy()
    df["education"] = df["education"].map(EDUCATION_MAP)
    df["self_employed"] = df["self_employed"].map(SELF_EMPLOYED_MAP)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map(TARGET_MAP)
    return df


def split_xy(df: pd.DataFrame):
    """Memisahkan fitur (X) dan target (y)."""
    X = df[FEATURES]
    y = df[TARGET]
    return X, y
