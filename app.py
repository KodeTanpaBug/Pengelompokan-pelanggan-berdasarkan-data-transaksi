from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import base64
import io

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    elbow_image = silhouette_image = cluster_image1 = cluster_image2 = cluster_image3 = table_html = None
    cluster_summary = None
    error = None
    clusterCount = None

    if request.method == 'POST':
        try:
            file = request.files['file']
            clusterCount = int(request.form['clusterCount'])

            if not file.filename.endswith('.csv'):
                raise ValueError("File harus berformat CSV")

            if file.content_type != 'text/csv':
                raise ValueError("Tipe file harus CSV (text/csv)")

            if clusterCount < 1:
                raise ValueError("Jumlah cluster harus minimal 1")

            df = pd.read_csv(file)

            expected_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
            if not all(col in df.columns for col in expected_cols):
                raise ValueError("CSV harus memiliki kolom: Age, Annual Income (k$), Spending Score (1-100)")

            if df[expected_cols].isnull().any().any():
                raise ValueError("Data tidak boleh memiliki nilai kosong (missing values)")

            if clusterCount >= len(df):
                raise ValueError(f"Jumlah cluster (K) tidak boleh >= jumlah data ({len(df)})")

            # Hapus kolom Cluster jika ada
            if 'Cluster' in df.columns:
                df.drop(columns='Cluster', inplace=True)

            X = df[expected_cols]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # === ELBOW METHOD ===
            inertia = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, random_state=42, max_iter=500)
                km.fit(X_scaled)
                inertia.append(km.inertia_)

            fig, ax = plt.subplots()
            ax.plot(range(1, 11), inertia, marker='o', linestyle='--', color='blue')
            ax.set_title("Metode Elbow untuk Menentukan K Optimal")
            ax.set_xlabel("Jumlah Cluster (K)")
            ax.set_ylabel("Inertia")
            ax.grid(True)
            elbow_image = plot_to_base64(fig)

            # === SILHOUETTE SCORE ===
            silhouette_scores = []
            k_range = range(2, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, max_iter=500).fit(X_scaled)
                score = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(score)

            fig, ax = plt.subplots()
            ax.plot(k_range, silhouette_scores, marker='o', linestyle='--', color='green')
            ax.set_title("Silhouette Score untuk Menentukan K Optimal")
            ax.set_xlabel("Jumlah Cluster (K)")
            ax.set_ylabel("Silhouette Score")
            ax.set_ylim(0, 1)
            ax.grid(True)
            silhouette_image = plot_to_base64(fig)

            # === FINAL CLUSTERING ===
            kmeans = KMeans(n_clusters=clusterCount, random_state=42, max_iter=500)
            df['Cluster'] = kmeans.fit_predict(X_scaled)

            palette = sns.color_palette("husl", clusterCount)

            # === PLOT 1: Usia vs Pendapatan ===
            fig = plt.figure()
            sns.scatterplot(data=df, x="Age", y="Annual Income (k$)", hue="Cluster", palette=palette)
            plt.title("Clustering Usia vs Pendapatan")
            cluster_image1 = plot_to_base64(fig)

            # === PLOT 2: Usia vs Skor Pengeluaran ===
            fig = plt.figure()
            sns.scatterplot(data=df, x="Age", y="Spending Score (1-100)", hue="Cluster", palette=palette)
            plt.title("Clustering Usia vs Skor Pengeluaran")
            cluster_image2 = plot_to_base64(fig)

            # === PLOT 3: Pendapatan vs Skor Pengeluaran ===
            fig = plt.figure()
            sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette=palette)
            plt.title("Clustering Pendapatan vs Skor Pengeluaran")
            cluster_image3 = plot_to_base64(fig)

            # === TABEL HASIL ===
            table_html = df.to_html(classes='table table-bordered table-striped', index=False)

            # === RINGKASAN CLUSTER ===
            cluster_summary_df = df.groupby('Cluster')[expected_cols].mean().reset_index()
            cluster_summary = cluster_summary_df.to_dict(orient='records')

        except Exception as e:
            error = str(e)

    return render_template('index.html',
                           elbow_image=elbow_image,
                           silhouette_image=silhouette_image,
                           cluster_image1=cluster_image1,
                           cluster_image2=cluster_image2,
                           cluster_image3=cluster_image3,
                           table=table_html,
                           cluster_summary=cluster_summary,
                           error=error,
                           clusterCount=clusterCount)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    base64_str = base64.b64encode(buf.read()).decode('utf-8')
    fig.clf()
    plt.close(fig)
    return f"data:image/png;base64,{base64_str}"

if __name__ == '__main__':
    app.run(debug=True)