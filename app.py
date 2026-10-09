import joblib
import pandas as pd
import streamlit as st

# 1. Konfigurasi Halaman Utama
st.set_page_config(
    page_title='Sistem Prediksi Risiko Diabetes',
    page_icon='🩺',
    layout='centered',
    initial_sidebar_state='collapsed',
)

# 2. Custom CSS untuk Tampilan Modern & Elegan
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f7f6;
    }
    
    .stForm {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid #eef2f5;
    }
    
    .section-title {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 15px;
        margin-top: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    .result-card {
        background-color: #ffffff;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 3. Load Model ML & Kolom
@st.cache_resource
def load_assets():
  model = joblib.load('diabetes_model.pkl')
  columns = joblib.load('feature_columns.pkl')
  return model, columns


try:
  model, feature_columns = load_assets()
except Exception as e:
  st.error(
      '⚠️ File model belum ditemukan! Jalankan perintah "python train.py" di'
      ' terminal terlebih dahulu.'
  )
  st.stop()

# Header Aplikasi
st.markdown(
    "<h1 style='text-align: center; color: #0f172a;'>🩺 Prediksi Risiko"
    ' Diabetes</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #64748b; margin-bottom: 25px;'>Sistem"
    ' analisis awal potensi risiko diabetes berdasarkan indikator kesehatan'
    ' pribadi.</p>',
    unsafe_allow_html=True,
)

# Inisialisasi Session State
if 'calculated' not in st.session_state:
  st.session_state.calculated = False

# ==========================================
# TAMPILAN 1: FORMULIR INPUT PENGGUNA
# ==========================================
if not st.session_state.calculated:
  with st.form('diabetes_form'):

    # Seksi 1: Informasi Diri
    st.markdown(
        "<div class='section-title'>👤 Informasi Profil Diri</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)

    with col1:
      gender_input = st.radio('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
      gender = 'Female' if gender_input == 'Perempuan' else 'Male'

      age = st.slider('Usia (Tahun)', min_value=1, max_value=100, value=25)

    with col2:
      st.markdown('**Pengukuran Fisik**')
      weight = st.slider(
          'Berat Badan (kg)',
          min_value=20.0,
          max_value=180.0,
          value=60.0,
          step=0.5,
      )
      height = st.slider(
          'Tinggi Badan (cm)',
          min_value=100.0,
          max_value=220.0,
          value=165.0,
          step=0.5,
      )

      # BMI dihitung di latar belakang tanpa ditampilkan di UI
      bmi = weight / ((height / 100) ** 2)

    # Seksi 2: Gaya Hidup & Riwayat Medis
    st.markdown(
        "<div class='section-title'>❤️ Gaya Hidup & Riwayat Medis</div>",
        unsafe_allow_html=True,
    )
    col3, col4 = st.columns(2)

    with col3:
      hypertension_input = st.radio(
          'Riwayat Hipertensi (Tekanan Darah Tinggi)', ['Tidak Ada', 'Ada']
      )

      heart_disease_input = st.radio(
          'Riwayat Penyakit Jantung', ['Tidak Ada', 'Ada']
      )

    with col4:
      smoking_options = {
          'Tidak Pernah Merokok': 'never',
          'Masih Merokok Aktif': 'current',
          'Pernah Merokok (Sudah Berhenti)': 'former',
          'Tidak Tahu / Info Tidak Ada': 'No Info',
      }
      smoking_input = st.radio(
          'Riwayat Merokok', list(smoking_options.keys())
      )
      smoking_history = smoking_options[smoking_input]

    # Seksi 3: Pemeriksaan Laboratorium
    st.markdown(
        "<div class='section-title'>🧪 Hasil Tes Laboratorium</div>",
        unsafe_allow_html=True,
    )
    col5, col6 = st.columns(2)

    with col5:
      unknown_hba1c = st.checkbox('Belum pernah cek HbA1c')
      if unknown_hba1c:
        hba1c = 5.5
        st.caption('ℹ️ *Menggunakan acuan nilai normal (5.5%)*')
      else:
        hba1c = st.slider(
            'Kadar HbA1c (%)',
            min_value=3.5,
            max_value=10.0,
            value=5.5,
            step=0.1,
        )

    with col6:
      unknown_glucose = st.checkbox('Belum pernah cek Gula Darah')
      if unknown_glucose:
        blood_glucose = 100
        st.caption('ℹ️ *Menggunakan acuan nilai normal (100 mg/dL)*')
      else:
        blood_glucose = st.slider(
            'Kadar Gula Darah (mg/dL)',
            min_value=50,
            max_value=300,
            value=100,
            step=1,
        )

    st.markdown('<br>', unsafe_allow_html=True)
    submitted = st.form_submit_button('🔍 Analisis Risiko Sekarang')

    if submitted:
      input_dict = {
          'age': age,
          'hypertension': 1 if hypertension_input == 'Ada' else 0,
          'heart_disease': 1 if heart_disease_input == 'Ada' else 0,
          'bmi': bmi,
          'HbA1c_level': hba1c,
          'blood_glucose_level': blood_glucose,
      }

      for col in feature_columns:
        if col not in input_dict:
          input_dict[col] = 0

      if f'gender_{gender}' in input_dict:
        input_dict[f'gender_{gender}'] = 1
      if f'smoking_history_{smoking_history}' in input_dict:
        input_dict[f'smoking_history_{smoking_history}'] = 1

      input_df = pd.DataFrame([input_dict])[feature_columns]

      prob = model.predict_proba(input_df)[0][1] * 100

      st.session_state.prob = prob
      st.session_state.unknown_lab = unknown_hba1c or unknown_glucose
      st.session_state.calculated = True
      st.rerun()

# ==========================================
# TAMPILAN 2: HASIL PREDIKSI KARTU VISUAL
# ==========================================
else:
  prob = st.session_state.prob

  st.markdown("<div class='result-card'>", unsafe_allow_html=True)

  st.markdown(
      "<h3 style='color: #334155;'>Hasil Analisis Estimasi Risiko</h3>",
      unsafe_allow_html=True,
  )

  st.markdown(
      f"<h1 style='font-size: 3rem; color: #0f172a;"
      f" margin-bottom:0px;'>{prob:.1f}%</h1>",
      unsafe_allow_html=True,
  )
  st.progress(int(prob))

  st.markdown('<br>', unsafe_allow_html=True)

  if prob < 25:
    st.success(
        '🟢 **Risiko Rendah**: Indikasi profil kesehatan Anda saat ini'
        ' berada dalam batas normal.'
    )
  elif prob < 55:
    st.warning(
        '🟡 **Risiko Sedang**: Terdapat beberapa potensi indikator risiko.'
        ' Disarankan mulai mengatur pola makan dan gaya hidup.'
    )
  else:
    st.error(
        '🔴 **Risiko Tinggi**: Estimasi menunjukkan tingkat risiko yang cukup'
        ' tinggi. Disarankan untuk berkonsultasi langsung dengan dokter.'
    )

  if st.session_state.unknown_lab:
    st.info(
        '💡 *Catatan: Analisis ini menggunakan estimasi acuan nilai normal'
        ' untuk parameter lab yang tidak Anda isi.*'
    )

  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown('<br>', unsafe_allow_html=True)

  if st.button('🔄 Hitung Ulang / Isikan Data Baru'):
    st.session_state.calculated = False
    st.rerun()