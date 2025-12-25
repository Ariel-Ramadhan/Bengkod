import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import os

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        # Cek apakah file model ada
        model_path = 'best_churn_model_tuned.pkl'
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan: {model_path}")
            st.stop()
        
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        st.stop()

model = load_model()

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Judul aplikasi
st.title("\Customer Churn Prediction System 📊")
st.markdown("Aplikasi ini memprediksi kemungkinan pelanggan akan churn (berhenti berlangganan) berdasarkan karakteristik dan pola penggunaan layanan.")

# Sidebar untuk informasi tambahan
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.markdown("""
    **Tentang Aplikasi Ini:**
    - Model: Random Forest Classifier
    - Akurasi: 81.2%
    - F1-Score: 0.78
    
    **Cara Penggunaan:**
    1. Isi data pelanggan pada tab "Prediksi Tunggal"
    2. Atau upload file CSV pada tab "Prediksi Massal"
    3. Lihat insight bisnis pada tab "Insight Bisnis"
    
    **Kontak Pengembang:**
    - Email: data.science@example.com
    - GitHub: @data-science-team
    """)
    
    st.markdown("---")
    st.markdown("Deployed with ❤️ using Streamlit")

# Tabs untuk navigasi
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Tunggal", "📊 Prediksi Massal", "📈 Insight Bisnis"])

with tab1:
    st.header("Prediksi Tunggal Pelanggan")
    st.markdown("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], key="gender")
        senior_citizen = st.selectbox("Status Lansia", ["No", "Yes"], help="Apakah pelanggan berusia di atas 65 tahun?", key="senior")
        partner = st.selectbox("Memiliki Pasangan", ["No", "Yes"], key="partner")
        dependents = st.selectbox("Memiliki Tanggungan", ["No", "Yes"], help="Anak atau anggota keluarga yang menjadi tanggungan", key="dependents")
        
    with col2:
        tenure = st.slider("Masa Berlangganan (bulan)", 0, 72, 10, help="Lama pelanggan berlangganan dalam bulan", key="tenure")
        phone_service = st.selectbox("Layanan Telepon", ["No", "Yes"], key="phone")
        
        multiple_lines_options = ["No", "Yes"] if phone_service == "Yes" else ["No phone service"]
        multiple_lines = st.selectbox("Jalur Telepon Multipel", multiple_lines_options, key="multiple")
        
        internet_service = st.selectbox("Jenis Internet", ["No", "DSL", "Fiber optic"], key="internet")
        
    with col3:
        if internet_service != "No":
            online_security = st.selectbox("Keamanan Online", ["No", "Yes"], key="security")
            online_backup = st.selectbox("Backup Online", ["No", "Yes"], key="backup")
            device_protection = st.selectbox("Proteksi Perangkat", ["No", "Yes"], key="device")
            tech_support = st.selectbox("Dukungan Teknis", ["No", "Yes"], key="tech")
            streaming_tv = st.selectbox("TV Streaming", ["No", "Yes"], key="tv")
            streaming_movies = st.selectbox("Film Streaming", ["No", "Yes"], key="movies")
        else:
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
    
    contract = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"], key="contract")
    paperless_billing = st.selectbox("Tagihan Digital", ["No", "Yes"], key="billing")
    payment_method = st.selectbox("Metode Pembayaran", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ], key="payment")
    monthly_charges = st.number_input("Biaya Bulanan ($)", min_value=0.0, value=70.0, step=0.1, key="monthly")
    total_charges = st.number_input("Total Biaya ($)", min_value=0.0, value=float(monthly_charges * tenure), step=0.1, 
                                    help="Total biaya sepanjang masa berlangganan", key="total")
    
    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Churn", type="primary", use_container_width=True):
        with st.spinner("Memproses prediksi..."):
            # Buat DataFrame dari input pengguna
            input_data = pd.DataFrame([{
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }])
            
            try:
                # Lakukan prediksi
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0][1]
                
                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi")
                
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 1:
                        st.error(f"🚨 **CHURN: Ya**")
                        st.markdown(f"**Probabilitas Churn: {prediction_proba:.2%}**")
                        st.markdown("Pelanggan ini memiliki risiko tinggi untuk churn. Rekomendasi tindakan segera diperlukan.")
                    else:
                        st.success(f"✅ **CHURN: Tidak**")
                        st.markdown(f"**Probabilitas Churn: {prediction_proba:.2%}**")
                        st.markdown("Pelanggan ini memiliki risiko rendah untuk churn. Pertahankan kualitas layanan.")
                
                with col2:
                    # Visualisasi probability
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#ff4b4b', '#2196f3'] if prediction == 1 else ['#2196f3', '#ff4b4b']
                    sizes = [prediction_proba, 1-prediction_proba]
                    labels = ['Churn', 'Tidak Churn']
                    explode = (0.05, 0)
                    
                    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                          startangle=90, textprops={'fontsize': 12})
                    ax.set_title('Distribusi Probabilitas Churn', fontsize=14)
                    st.pyplot(fig)
                    plt.close()
                
                # Rekomendasi berdasarkan prediksi
                st.subheader("Rekomendasi Strategis")
                if prediction == 1:
                    st.warning("peringatan Tinggi - Diperlukan intervensi segera")
                    recommendations = [
                        "Hubungi pelanggan untuk menawarkan diskon khusus atau paket bundling",
                        "Tawarkan upgrade ke kontrak jangka panjang dengan insentif",
                        "Identifikasi keluhan spesifik dan berikan solusi personal",
                        "Tetapkan pelanggan ini dalam program retensi prioritas",
                        "Lakukan analisis lebih dalam tentang pola penggunaan layanan"
                    ]
                else:
                    st.info("peringatan Rendah - Fokus pada kepuasan pelanggan")
                    recommendations = [
                        "Tawarkan layanan tambahan yang sesuai dengan pola penggunaan",
                        "Kirim survei kepuasan untuk memahami kebutuhan lebih dalam",
                        "Beri penghargaan loyalitas untuk meningkatkan engagement",
                        "Monitor perubahan penggunaan secara berkala",
                        "Berikan edukasi tentang fitur-fitur baru yang dapat meningkatkan pengalaman"
                    ]
                
                for i, rec in enumerate(recommendations):
                    st.markdown(f"{i+1}. {rec}")
                
            except Exception as e:
                st.error(f"Terjadi error saat melakukan prediksi: {str(e)}")
                st.exception(e)

with tab2:
    st.header("Prediksi Massal (Batch Prediction)")
    st.markdown("Upload file CSV untuk memprediksi churn untuk banyak pelanggan sekaligus.")
    
    # Template download
    st.subheader("📥 Format File yang Diperlukan")
    st.markdown("File CSV harus memiliki kolom-kolom berikut dengan nama yang tepat:")
    
    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
        'MonthlyCharges', 'TotalCharges'
    ]
    
    template_df = pd.DataFrame(columns=required_columns)
    st.dataframe(template_df, use_container_width=True)
    
    # Contoh data untuk download
    st.markdown("Download contoh file CSV:")
    sample_data = {
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No'],
        'tenure': [1, 34, 67],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'No', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [29.85, 56.95, 70.7],
        'TotalCharges': [29.85, 1889.5, 4747.9]
    }
    sample_df = pd.DataFrame(sample_data)
    
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="Download Contoh CSV",
        data=csv,
        file_name="contoh_data_churn.csv",
        mime="text/csv"
    )
    
    # Upload file
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diupload! Jumlah data: {len(batch_data)}")
            
            # Validasi kolom
            missing_cols = [col for col in required_columns if col not in batch_data.columns]
            if missing_cols:
                st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
            else:
                # Tampilkan preview data
                st.subheader("Preview Data")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                # Tombol untuk memproses prediksi batch
                if st.button("Jalankan Prediksi Massal", type="primary", use_container_width=True):
                    with st.spinner("Memproses prediksi massal..."):
                        try:
                            # Lakukan prediksi
                            predictions = model.predict(batch_data)
                            prediction_proba = model.predict_proba(batch_data)[:, 1]
                            
                            # Tambahkan hasil ke DataFrame
                            result_df = batch_data.copy()
                            result_df['Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
                            result_df['Churn Probability'] = prediction_proba
                            
                            # Tampilkan hasil
                            st.subheader("Hasil Prediksi")
                            st.dataframe(result_df[['Prediction', 'Churn Probability']].head(10), use_container_width=True)
                            
                            # Statistik hasil prediksi
                            churn_count = len(result_df[result_df['Prediction'] == 'Yes'])
                            total_count = len(result_df)
                            churn_percentage = (churn_count / total_count) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Pelanggan", total_count)
                            with col2:
                                st.metric("Prediksi Churn", churn_count)
                            with col3:
                                st.metric("Persentase Churn", f"{churn_percentage:.1f}%")
                            
                            # Visualisasi hasil
                            fig, ax = plt.subplots(figsize=(10, 6))
                            churn_counts = result_df['Prediction'].value_counts()
                            colors = ['#ff4b4b', '#2196f3']
                            ax.pie(churn_counts.values, labels=churn_counts.index, 
                                  autopct='%1.1f%%', colors=colors, startangle=90)
                            ax.set_title('Distribusi Prediksi Churn', fontsize=14)
                            st.pyplot(fig)
                            plt.close()
                            
                            # Download hasil
                            st.subheader("📥 Download Hasil Prediksi")
                            csv_result = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Hasil CSV",
                                data=csv_result,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Terjadi error saat memproses prediksi massal: {str(e)}")
                            st.exception(e)
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {str(e)}")

with tab3:
    st.header("Insight Bisnis dan Analisis Churn")
    st.markdown("Analisis mendalam tentang pola churn dan rekomendasi bisnis untuk mengurangi churn rate.")
    
    # Simulasi data untuk visualisasi insight
    st.subheader("📊 Pola Churn Berdasarkan Karakteristik Pelanggan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualisasi churn vs tenure
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        tenure_ranges = [0, 12, 24, 36, 48, 60, 72]
        tenure_labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
        
        # Data simulasi berdasarkan analisis sebelumnya
        churn_by_tenure = [45, 28, 15, 8, 3, 1]
        
        bars = ax1.bar(tenure_labels, churn_by_tenure, color=['#ff4b4b', '#ff7b57', '#ffa557', '#ffd557', '#d5ff57', '#a5ff57'])
        ax1.set_title('Tingkat Churn Berdasarkan Masa Berlangganan', fontsize=14)
        ax1.set_xlabel('Masa Berlangganan (bulan)', fontsize=12)
        ax1.set_ylabel('Persentase Churn (%)', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Tambahkan label nilai di atas bar
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        # Visualisasi churn vs layanan internet
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        internet_services = ['Fiber optic', 'DSL', 'No Internet']
        churn_rates = [40, 20, 5]
        
        bars = ax2.barh(internet_services, churn_rates, color=['#ff4b4b', '#ff7b57', '#ffa557'])
        ax2.set_title('Tingkat Churn Berdasarkan Jenis Internet', fontsize=14)
        ax2.set_xlabel('Persentase Churn (%)', fontsize=12)
        ax2.set_ylabel('Jenis Internet', fontsize=12)
        ax2.grid(axis='x', alpha=0.3)
        
        # Tambahkan label nilai di samping bar
        for i, v in enumerate(churn_rates):
            ax2.text(v + 1, i, f'{v}%', va='center')
        
        st.pyplot(fig2)
        plt.close()
    
    st.subheader("💡 Rekomendasi Strategis untuk Mengurangi Churn")
    
    # Rekomendasi berdasarkan analisis
    recommendations = [
        {
            "judul": "Konversi Kontrak Jangka Pendek ke Jangka Panjang",
            "deskripsi": "Pelanggan dengan kontrak Month-to-month memiliki tingkat churn 3x lebih tinggi. Tawarkan insentif untuk mengonversi ke kontrak 1-2 tahun.",
            "dampak": "Potensi pengurangan churn: 25%",
            "biaya": "Sedang"
        },
        {
            "judul": "Program Retensi Pelanggan Baru",
            "deskripsi": "54% churn terjadi pada pelanggan dengan masa berlangganan <12 bulan. Berikan program onboarding eksklusif dan follow-up rutin.",
            "dampak": "Potensi pengurangan churn: 30%",
            "biaya": "Rendah"
        },
        {
            "judul": "Optimasi Layanan Fiber Optic",
            "deskripsi": "Pelanggan Fiber optic dengan layanan tambahan memiliki tingkat churn tertinggi. Evaluasi harga paket bundling dan kualitas layanan.",
            "dampak": "Potensi pengurangan churn: 20%",
            "biaya": "Tinggi"
        },
        {
            "judul": "Perbaikan Metode Pembayaran",
            "deskripsi": "Pelanggan dengan pembayaran Electronic check memiliki tingkat churn 15% lebih tinggi. Dorong konversi ke metode pembayaran otomatis.",
            "dampak": "Potensi pengurangan churn: 10%",
            "biaya": "Rendah"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"{i+1}. {rec['judul']}"):
            st.markdown(f"**Deskripsi:** {rec['deskripsi']}")
            st.markdown(f"**Dampak Potensial:** {rec['dampak']}")
            st.markdown(f"**Estimasi Biaya Implementasi:** {rec['biaya']}")
    
    st.subheader("📈 Proyeksi ROI dari Program Retensi")
    st.markdown("Berdasarkan analisis biaya-manfaat, berikut proyeksi ROI dari implementasi program retensi:")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    months = ['Bulan 1', 'Bulan 3', 'Bulan 6', 'Bulan 12']
    roi_values = [-50, 20, 100, 250]  # Simulasi ROI dalam persentase
    
    line = ax3.plot(months, roi_values, marker='o', linewidth=3, markersize=10, color='#2196f3', label='ROI')
    ax3.fill_between(months, roi_values, alpha=0.2, color='#2196f3')
    ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax3.set_title('Proyeksi ROI Program Retensi Pelanggan', fontsize=14)
    ax3.set_ylabel('ROI (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Tambahkan nilai pada titik data
    for i, v in enumerate(roi_values):
        ax3.text(i, v+10, f'{v}%', ha='center', fontweight='bold')
    
    st.pyplot(fig3)
    plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Customer Churn Prediction System &copy; 2024 | Powered by Machine Learning</p>
    <p><small>Model Accuracy: 81.2% | F1-Score: 0.78 | Menggunakan Random Forest Classifier</small></p>
</div>
""", unsafe_allow_html=True)