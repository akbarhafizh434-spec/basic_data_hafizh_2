import streamlit as st
import pandas as pd
import pickle

# Memuat model dan scaler yang telah disimpan
@st.cache_resource
def load_resources():
    try:
        with open('bayesian_ridge_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        with open('standard_scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Make sure 'bayesian_ridge_model.pkl' and 'standard_scaler.pkl' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

loaded_model, loaded_scaler = load_resources()

# Pemetaan jenis kelamin (harus sesuai dengan saat training)
mapping_gender = {'Pria': 'Laki-laki', 'L': 'Laki-laki', 'Perempuan': 'Wanita', 'P': 'Wanita', 'wanita': 'Wanita', 'Laki-Laki': 'Laki-laki'}

# Daftar kolom fitur yang digunakan saat training (sesuai dengan X_train/X_test)
# Ini penting untuk memastikan urutan dan keberadaan kolom saat prediksi
feature_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

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

# Input yang tidak digunakan oleh model akhir, tetapi ada di data asli
# Pendidikan dan Jurusan tidak digunakan sebagai fitur dalam model akhir seperti yang terlihat dari feature_cols
# pendidikan = st.selectbox("Pendidikan Terakhir", ['SMA', 'SMK', 'D3', 'S1'])
# jurusan = st.selectbox("Jurusan Pelatihan", ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])

if st.button("Prediksi Gaji"):
    # Membuat DataFrame dari input pengguna
    input_data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Jenis_Kelamin': jenis_kelamin_raw,
        'Status_Bekerja': status_bekerja_raw
    }
    new_df = pd.DataFrame([input_data])

    # Pra-pemrosesan data input baru (mereplikasi langkah-langkah training)

    # 1. Terapkan pemetaan jenis kelamin
    new_df['Jenis_Kelamin'] = new_df['Jenis_Kelamin'].replace(mapping_gender)

    # 2. One-Hot Encode fitur kategorikal
    # Buat dummy columns untuk semua kemungkinan kategori agar konsisten dengan training
    df_encoded_temp = pd.DataFrame(0, index=new_df.index, columns=['Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'])

    # Mengatur nilai 1 untuk kolom one-hot yang relevan
    if 'Laki-laki' in new_df['Jenis_Kelamin'].values:
        df_encoded_temp['Jenis_Kelamin_Laki-laki'] = 1
    elif 'Wanita' in new_df['Jenis_Kelamin'].values:
        df_encoded_temp['Jenis_Kelamin_Wanita'] = 1

    if 'Sudah Bekerja' in new_df['Status_Bekerja'].values:
        df_encoded_temp['Status_Bekerja_Sudah Bekerja'] = 1
    elif 'Belum Bekerja' in new_df['Status_Bekerja'].values:
        df_encoded_temp['Status_Bekerja_Belum Bekerja'] = 1

    # Gabungkan kolom numerik dengan one-hot encoded
    new_df_processed = pd.concat([
        new_df[['Usia', 'Durasi_Jam', 'Nilai_Ujian']],
        df_encoded_temp
    ], axis=1)

    # Pastikan urutan kolom sesuai dengan feature_cols yang digunakan saat training
    new_df_processed = new_df_processed[feature_cols]

    # 3. Scaling fitur numerik menggunakan scaler yang dimuat
    # Fitur numerik adalah 'Usia', 'Durasi_Jam', 'Nilai_Ujian'
    # Pastikan hanya kolom numerik yang discale
    scaled_numerical_features = loaded_scaler.transform(new_df_processed[['Usia', 'Durasi_Jam', 'Nilai_Ujian']])
    scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=['Usia', 'Durasi_Jam', 'Nilai_Ujian'], index=new_df_processed.index)

    # Gabungkan kembali dengan kolom one-hot yang tidak discale
    scaled_new_df = pd.concat([
        scaled_numerical_df,
        new_df_processed[['Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']]
    ], axis=1)

    # Lakukan prediksi
    predicted_salary = loaded_model.predict(scaled_new_df)

    st.subheader("Hasil Prediksi:")
    st.success(f"Prediksi Gaji Awal Anda: **{predicted_salary[0]:.2f} Juta Rupiah**")

st.markdown("--- ")
st.info("Aplikasi ini menggunakan model Bayesian Ridge Regression untuk memprediksi gaji awal berdasarkan input yang diberikan.")
