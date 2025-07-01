import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Risiko Kesehatan Kehamilan", layout="centered")

# Judul utama
st.title("Aplikasi Prediksi Risiko Kesehatan Kehamilan")
st.markdown("---")

# Load model, scaler, encoder
model = joblib.load('model_risk.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('ordinal_encoder.pkl')  # Harus objek sklearn encoder!

# Form input
st.header("Silakan Masukkan Data Kesehatan Ibu")

usia = st.number_input("Usia (tahun)", min_value=0, max_value=60, value=0)
tekanan_sistolik = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=0, max_value=200, value=0)
tekanan_diastolik = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=0, max_value=140, value=0)
gula_darah = st.number_input("Kadar Gula Darah (mg/dL)", min_value=0.0, max_value=500.0, value=0.0)
# suhu_tubuh = st.number_input("Suhu Tubuh (°C)", min_value=0.0, max_value=42.0, value=0.0, step=0.1)
suhu_tubuh = st.number_input("Suhu Tubuh (°F)", min_value=80.0, max_value=110.0, value=98.6, step=0.1)

detak_jantung = st.number_input("Detak Jantung (bpm)", min_value=0, max_value=180, value=0)

# Tombol prediksi
if st.button("Prediksi Risiko"):
    # Cek apakah semua input sudah diisi (tidak ada yang 0)
    if 0 in [usia, tekanan_sistolik, tekanan_diastolik, gula_darah, suhu_tubuh, detak_jantung]:
        st.warning("⚠️ Silakan isi semua data dengan nilai yang valid terlebih dahulu.")
    else:
        # Buat dataframe input
        data_input = pd.DataFrame([[usia, tekanan_sistolik, tekanan_diastolik, gula_darah, suhu_tubuh, detak_jantung]],
                                  columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

        # Scaling
        data_scaled = scaler.transform(data_input)

        # Prediksi
        pred_encoded = model.predict(data_scaled)

        # Inverse transform jika memungkinkan
        try:
            hasil_label = encoder.inverse_transform(pred_encoded)[0]
        except AttributeError:
            hasil_label = pred_encoded[0]

        # Tampilkan hasil
        st.success(f"Hasil Prediksi: **{hasil_label.upper()}**")

        # Simpan hasil ke session_state
        if 'riwayat_prediksi' not in st.session_state:
            st.session_state.riwayat_prediksi = []

        st.session_state.riwayat_prediksi.append({
            'Usia': usia,
            'Sistolik': tekanan_sistolik,
            'Diastolik': tekanan_diastolik,
            'Gula Darah': gula_darah,
            'Suhu Tubuh': suhu_tubuh,
            'Detak Jantung': detak_jantung,
            'Hasil Prediksi': hasil_label.upper()
        })

# Tampilkan riwayat prediksi
if 'riwayat_prediksi' in st.session_state and st.session_state.riwayat_prediksi:
    st.subheader("Riwayat Prediksi Sebelumnya")
    riwayat_df = pd.DataFrame(st.session_state.riwayat_prediksi)
    st.dataframe(riwayat_df, use_container_width=True)

    # Tombol download CSV (opsional)
    csv = riwayat_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Riwayat Prediksi sebagai CSV",
        data=csv,
        file_name='riwayat_prediksi.csv',
        mime='text/csv',
    )
