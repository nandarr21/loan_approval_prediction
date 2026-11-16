"""Pintu masuk untuk Vercel. Vercel menjalankan file di folder api/,
jadi di sini kita cukup mengimpor objek `app` dari app.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402,F401
