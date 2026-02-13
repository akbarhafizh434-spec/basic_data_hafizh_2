import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Memuat model dan scaler
@st.cache_resource
def load_resources():
    try:
        with open('bayesian_ridge_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        with open('standard_scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        st.error("Error: File model atau scaler tidak ditemukan.")
        st.stop()

loaded_model, loaded_scaler = load_resources()

# 1. TAMBAHKAN kolom Jurusan di feature_cols (Sesuaikan nama kolom dengan saat training)
feature_cols = [
    'Usia', 'Durasi_Jam', 'Nilai_Ujian', 
    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja',
    'Jurusan_Teknik', 'Jurusan_IT', 'Jurusan_Bisnis', 'Jurusan_Kesehatan', 'Jurusan_Seni' # Contoh nama kolom jurusan
]

st.set_page_config(page_title="Prediksi Gaji Awal Lulusan Vokasi")
st.title("💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi")

st.markdown("--- ")
st.subheader("Masukkan Informasi Peserta:")

# Input dari pengguna
usia = st.slider("Usia (tahun)", 18, 60, 25)
durasi_jam = st.slider("Durasi Pelatihan (jam)", 20, 100, 60)
nilai_ujian = st.slider("Nilai Ujian", 50.0, 100.0, 75.0)
jenis_kelamin_raw = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Wanita'])
status_bekerja_raw = st.selectbox("Status Bekerja", ['Sudah Bekerja', 'Belum Bekerja'])

# 2. TAMBAHKAN Input Jurusan
jurusan_raw = st.selectbox("Jurusan", ['Teknik', 'IT', 'Bisnis', 'Kesehatan', 'Seni'])

if st.button("Prediksi Gaji"):
    # Membuat DataFrame awal
    input_data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Jenis_Kelamin': jenis_kelamin_raw,
        'Status_Bekerja': status_bekerja_raw,
        'Jurusan': jurusan_raw
    }
    new_df = pd.DataFrame([input_data])

    # 3. PROSES ONE-HOT ENCODING MANUAL
    # Buat DataFrame kosong dengan SEMUA kolom fitur bernilai 0
    df_encoded = pd.DataFrame(0, index=new_df.index, columns=feature_cols)

    # Isi nilai numerik dasar
    df_encoded['Usia'] = usia
    df_encoded['Durasi_Jam'] = durasi_jam
    df_encoded['Nilai_Ujian'] = nilai_ujian

    # Set nilai 1 untuk Jenis Kelamin
    if jenis_kelamin_raw == 'Laki-laki':
        df_encoded['Jenis_Kelamin_Laki-laki'] = 1
    else:
        df_encoded['Jenis_Kelamin_Wanita'] = 1

    # Set nilai 1 untuk Status Bekerja
    if status_bekerja_raw == 'Sudah Bekerja':
        df_encoded['Status_Bekerja_Sudah Bekerja'] = 1
    else:
        df_encoded['Status_Bekerja_Belum Bekerja'] = 1

    # Set nilai 1 untuk Jurusan (Sesuai pilihan user)
    # Pastikan string 'Jurusan_' + jurusan_raw cocok dengan nama kolom di feature_cols
    kolom_jurusan_pilihan = f"Jurusan_{jurusan_raw}"
    if kolom_jurusan_pilihan in df_encoded.columns:
        df_encoded[kolom_jurusan_pilihan] = 1

    # 4. SCALING DAN PREDIKSI
    # Pastikan urutan kolom di df_encoded sama persis dengan feature_cols
    df_final = df_encoded[feature_cols]
    
    scaled_new_data = loaded_scaler.transform(df_final)
    predicted_salary = loaded_model.predict(scaled_new_data)

    st.subheader("Hasil Prediksi:")
    st.success(f"Prediksi Gaji Awal Anda: **{predicted_salary[0]:.2f} Juta Rupiah**")

st.markdown("--- ")
st.info("Catatan: Pastikan nama kolom jurusan di kode ini sama dengan nama kolom saat Anda melakukan training model.")
