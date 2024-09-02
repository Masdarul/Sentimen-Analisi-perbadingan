import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Judul
st.title(":blue[Analisis Sentimen Aplikasi Belajar Online]")

# Menyiapkan data
df = pd.read_excel("data/hasil.xlsx")
def split_data(df):
    # Menghapus atau mengganti NaN di kolom 'text_akhir'
    df['text_akhir'] = df['text_akhir'].fillna("")  # Mengganti NaN dengan string kosong
    
    X = df['text_akhir']
    y = df['polarity']
    
    # Vectorisasi teks dengan TF-IDF
    tfidf = TfidfVectorizer(max_features=200, min_df=17, max_df=0.8)
    X_tfidf = tfidf.fit_transform(X)
    
    # Mengubah sparse matrix menjadi dense array
    X_dense = X_tfidf.toarray()
    
    return train_test_split(X_dense, y, test_size=0.3, random_state=42)


# Fungsi untuk mengevaluasi model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1-Score": f1_score(y_test, y_pred, average='macro'),
    }


# Define tabs
tab1, tab2, tab3 = st.tabs(["Beranda","Visualisasi Aplikasi", "Visualisasi Evaluasi"])


with tab1:
    st.caption("Setiap platform pembelajaran daring memiliki keunggulan dan kekurangan masing-masing yang membuatnya cocok untuk tujuan dan audiens yang berbeda. Coursera dan edX menonjol dengan program-program yang diakui secara akademis dari universitas ternama, sementara Udemy dan SoloLearn menawarkan fleksibilitas dan akses seumur hidup ke berbagai kursus yang mencakup topik umum hingga teknis. Ruangguru menyediakan solusi berbasis kurikulum nasional untuk pendidikan di Indonesia, dan Simplilearn berfokus pada peningkatan keterampilan profesional dengan sertifikasi yang diakui industri. Dengan memahami kekuatan dan kelemahan masing-masing, pengguna dapat memilih platform yang paling sesuai dengan kebutuhan belajar mereka, baik untuk pengembangan pribadi, profesional, atau akademik.")

    st.html("<a href='https://github.com/Masdarul/Sentimen-Analisi-perbadingan/blob/main/pdf/Analisis%20Sentimen%20Aplikasi%20Belajar%20Online%20Perbandingan%20Platform%20dan%20Algoritma.pdf' target='_blank'>Detail</a> || <a href='https://github.com/Masdarul/Sentimen-Analisi-perbadingan/blob/main/pdf/glosarium.pdf'>Glosarium</a>")

    st.subheader("Algoritma", divider="gray")
    st.caption("Secara keseluruhan, Support Vector Machine (SVM) dan XGBoost muncul sebagai algoritma yang paling kuat dan efektif dalam analisis sentimen ini, dengan hasil akurasi dan metrik evaluasi yang unggul. Logistic Regression juga memberikan hasil yang seimbang dan bisa diandalkan, sementara Naive Bayes dan K-Nearest Neighbors (KNN) mungkin lebih cocok untuk dataset yang lebih sederhana atau ketika kecepatan menjadi prioritas. Decision Tree menunjukkan tanda-tanda overfitting, yang memerlukan penyesuaian lebih lanjut atau penggunaan teknik pruning.")

    header = ["Number","Model","Accuracy test","Accuracy train"]
    rows = [
        ["1","Naive Bayes","79,81%","80,18%"],
        ["2","Support Vector Machine","86,61%","94,75%"],
        ["3","Decision Tree","78,61%","98,88%"],
        ["4","Logistic Regression","87,21%","88,61%"],
        ["5","K-Nearest Neighbors", "80,51%","86,01%"]
    ]
    dt = pd.DataFrame(rows, columns=header)
    my_table = st.table(dt)
    st.subheader("Visualisasi", divider="gray")
    # visualisasi
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [count for count in df['polarity'].value_counts()]
    labels = list(df['polarity'].value_counts().index)
    explode = (0.1, 0)  # Jika hanya ada dua kategori, pastikan explode memiliki panjang yang sama dengan jumlah kategori

    ax.pie(x=sizes, labels=labels, autopct='%1.1f%%', explode=explode, textprops={'fontsize': 14})
    ax.set_title('Polaritas Sentimen pada Semua aplikasi', fontsize=16, pad=20)

    # Menampilkan pie chart di Streamlit
    st.pyplot(fig)  # Pass the figure object here
    st.caption("Pada Diagram lingkaran ini, dapat melihat bahwa aplikasi pembelajaran online dari Coursera, Ruangguru, edX, Udemy, Sololearn, dan Simplilearn menunjukkan bahwa 73,2% ulasan pengguna bersifat positif, sedangkan 26,8% ulasan bersifat negatif. Hal ini menunjukkan mayoritas pengguna memiliki pengalaman yang baik dengan aplikasi-aplikasi tersebut, meskipun ada sejumlah kecil ulasan yang mengindikasikan ketidakpuasan")

    # Membuat plot countplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Aplikasi', hue='polarity', data=df, palette={'positif': 'blue', 'negatif': 'orange'}, ax=ax)

    # Menambahkan angka di atas setiap batang
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Memastikan label hanya ditambahkan jika tinggi batang lebih dari 0
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), 
                        textcoords='offset points')

    # Menambahkan judul
    ax.set_title('Statistik Sentimen Pengguna Platform Pembelajaran Online Di Indonesia')

    # Menampilkan plot di Streamlit
    st.pyplot(fig)  # Pass the figure object here
    st.caption("Pada diagram batang ini, terlihat statistik ulasan aplikasi pembelajaran online di Indonesia. Aplikasi Coursera menerima 220 ulasan positif dan 72 ulasan negatif. Ruangguru memiliki 6.815 ulasan positif dan 2.878 ulasan negatif. edX memperoleh 309 ulasan positif dan 53 ulasan negatif. Udemy mendapatkan 711 ulasan positif dan 310 ulasan negatif. SoloLearn menerima 3.379 ulasan positif dan 873 ulasan negatif. Simplilearn memperoleh 58 ulasan positif dan 11 ulasan negatif. Data ini menunjukkan bahwa pengguna di Indonesia lebih banyak menggunakan aplikasi Ruangguru dibandingkan dengan aplikasi lainnya.")
with tab2:
    # Option to select application
    option = st.selectbox(
        "Pilih Aplikasi",
        ("Coursera", "Ruangguru", "Edx", "Udemy", "SoloLearn", "Simplilearn"),
    )

    # Filter data berdasarkan aplikasi yang dipilih
    df_selected = df[df['Aplikasi'].str.lower() == option.lower()]

    # Filter data untuk ulasan positif dan negatif
    df_positif = df_selected[df_selected['polarity'].str.lower() == 'positif']
    df_negatif = df_selected[df_selected['polarity'].str.lower() == 'negatif']

    # Tampilkan Ulasan Positif dengan tinggi terbatas
    st.subheader("Ulasan Positif", divider="gray")
    if not df_positif.empty:
        st.dataframe(df_positif[['Nama Pengguna', 'Ulasan', 'Rating']], height=200)  # Mengatur tinggi agar tabel bisa digulir
    else:
        st.write("Tidak ada ulasan positif untuk aplikasi ini.")

    # Tampilkan Ulasan Negatif dengan tinggi terbatas
    st.subheader("Ulasan Negatif", divider="gray")
    if not df_negatif.empty:
        st.dataframe(df_negatif[['Nama Pengguna', 'Ulasan', 'Rating']], height=200)  # Mengatur tinggi agar tabel bisa digulir
    else:
        st.write("Tidak ada ulasan negatif untuk aplikasi ini.")

    
    if not df_selected.empty:
        # Menghitung jumlah ulasan positif dan negatif
        counts = df_selected['polarity'].value_counts()
        positive_count = counts.get('positif', 0)
        negative_count = counts.get('negatif', 0)
        total_count = positive_count + negative_count

        # Menghitung persentase
        positive_percent = (positive_count / total_count) * 100 if total_count > 0 else 0
        negative_percent = (negative_count / total_count) * 100 if total_count > 0 else 0

        # Display pie chart of sentiment polarity
        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [count for count in df_selected['polarity'].value_counts()]
        labels = list(df_selected['polarity'].value_counts().index)
        explode = [0.1] + [0] * (len(sizes) - 1)
        ax.pie(x=sizes, labels=labels, autopct='%1.1f%%', explode=explode, textprops={'fontsize': 14})
        ax.set_title(f'Polaritas Sentimen pada Ulasan {option}', fontsize=16, pad=20)
        st.pyplot(fig)
        st.caption(f"Pada diagram lingkaran ini, dapat melihat bahwa {positive_count} ulasan pengguna {option} bersifat positif ({positive_percent:.2f}%), sedangkan {negative_count} ulasan bersifat negatif ({negative_percent:.2f}%).")

        # Display count plot of ratings
                # Menghitung jumlah ulasan untuk setiap rating
        rating_counts = df_selected['Rating'].value_counts().sort_index()

        rating_1_count = rating_counts.get(1, 0)
        rating_2_count = rating_counts.get(2, 0)
        rating_3_count = rating_counts.get(3, 0)
        rating_4_count = rating_counts.get(4, 0)
        rating_5_count = rating_counts.get(5, 0)

        # Tentukan rating terbanyak
        max_rating = rating_counts.idxmax()
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Rating', data=df_selected)
        for p in plt.gca().patches:
            plt.gca().annotate(f'{int(p.get_height())}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), 
                               textcoords='offset points')
        plt.title(f'Rating dari untuk {option}')
        st.pyplot(plt)
        st.caption(f"Pada diagram batang ini, terlihat bahwa pada aplikasi {option}: "
                   f"rating 1 memiliki {rating_1_count} ulasan, "
                   f"rating 2 memiliki {rating_2_count} ulasan, "
                   f"rating 3 memiliki {rating_3_count} ulasan, "
                   f"rating 4 memiliki {rating_4_count} ulasan, dan "
                   f"rating 5 memiliki {rating_5_count} ulasan. "
                   f"Rating terbanyak ada pada rating {max_rating}.")

        # Display word clouds for positive and negative comments
        positive_texts = " ".join(df_selected[df_selected['polarity'] == 'positif']['text_akhir'].dropna().astype(str))
        negative_texts = " ".join(df_selected[df_selected['polarity'] == 'negatif']['text_akhir'].dropna().astype(str))



        if positive_texts:
            wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.title(f'Kata-kata Komentar Positif untuk {option}')
            plt.axis('off')
            st.pyplot(plt)
        
        if negative_texts:
            wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_texts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_negative, interpolation='bilinear')
            plt.title(f'Kata-kata Komentar Negatif untuk {option}')
            plt.axis('off')
            st.pyplot(plt)
            

with tab3:
    st.header("Evaluasi Model")

    option = st.selectbox(
        "Pilih Evaluasi",
        ("Naive Bayes", "Support Vector Machine", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors", "XGBoost")
    )
    

    df = pd.read_excel("data/hasil.xlsx")
    if option == "XGBoost":
        le = LabelEncoder()
        df['polarity'] = le.fit_transform(df['polarity'])
        X_train, X_test, y_train, y_test = split_data(df)

    else:
        X_train, X_test, y_train, y_test = split_data(df)

    # Select and evaluate model based on user input
    if option == "Naive Bayes":
        model = GaussianNB()
    elif option == "Support Vector Machine":
        model = SVC()
    elif option == "Decision Tree":
        model = DecisionTreeClassifier()
    elif option == "Logistic Regression":
        model = LogisticRegression()
    elif option == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    elif option == "XGBoost":
        model = XGBClassifier()

    # Evaluate the selected model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Display the evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(pd.DataFrame([metrics]))

    # Visualize the metrics
    st.subheader("Visualization of Metrics")
    fig, ax = plt.subplots()
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax)
    ax.set_ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    st.pyplot(fig)



footer = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: blue;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            z-index: 100; 
        }
        .footer a {
            color: white; 
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline; 
        }
    </style>
    <div class="footer">
        <p>© 2024 Masdarul Rizqi. Semua Hak Dilindungi. | Kontak: <a href="https://github.com/Masdarul">Github</a> || <a href="https://www.linkedin.com/in/masdarul-rizqi-46ba17249/">LinkedIn</a></p>
    </div>
    """

# Render footer di halaman Streamlit
st.markdown(footer, unsafe_allow_html=True)