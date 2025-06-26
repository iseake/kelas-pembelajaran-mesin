import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from io import StringIO
import pydotplus # Pastikan Anda sudah menginstal: pip install pydotplus graphviz
from IPython.display import Image

# 1. Dataset
data = {
    'Kartu': ['Prabayar', 'Pascabayar', 'Prabayar', 'Prabayar', 'Pascabayar', 'Pascabayar', 'Prabayar', 'Prabayar', 'Pascabayar', 'Pascabayar', 'Pascabayar'],
    'Panggilan': ['Sedikit', 'Banyak', 'Banyak', 'Banyak', 'Cukup', 'Cukup', 'Cukup', 'Cukup', 'Sedikit', 'Banyak', 'Sedikit'],
    'Block': ['Sedang', 'Sedang', 'Sedang', 'Rendah', 'Tinggi', 'Sedang', 'Sedang', 'Rendah', 'Tinggi', 'Tinggi', 'Rendah'],
    'Bonus': ['Tidak', 'Ya', 'Ya', 'Tidak', 'Ya', 'Ya', 'Ya', 'Tidak', 'Ya', 'Ya', 'Ya'],
}
df = pd.DataFrame(data)

print("Dataset Asli:")
print(df)
print("\n" + "="*50 + "\n")

# 2. Preprocessing Data (Mengubah Data Kategorikal menjadi Numerik)
# Kita akan menggunakan LabelEncoder untuk setiap kolom fitur dan kolom target
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

print("Dataset Setelah Preprocessing (Label Encoding):")
print(df)
print("\n" + "="*50 + "\n")

# Pisahkan fitur (X) dan target (y)
X = df.drop('Bonus', axis=1)
y = df['Bonus']

# Nama fitur dan kelas untuk visualisasi
feature_names = X.columns
target_names = label_encoders['Bonus'].classes_ # Mengambil nama kelas asli dari LabelEncoder

# 3. Membangun Decision Tree (Menggunakan Kriteria Entropy untuk mendekati ID3)
# Mengatur criterion='entropy' akan menggunakan information gain untuk pemisahan
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X, y)

print("Model Decision Tree Berhasil Dilatih!")
print("\n" + "="*50 + "\n")

# 4. Visualisasi Decision Tree
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,
                feature_names=feature_names,
                class_names=target_names,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Simpan sebagai file PNG (opsional)
graph.write_png('decision_tree_id3.png')

# Tampilkan gambar di Jupyter Notebook atau lingkungan yang mendukung
Image(graph.create_png())

# 5. Contoh Prediksi
print("Contoh Prediksi:")

# Kondisi baru: Kartu=Prabayar, Panggilan=Cukup, Block=Sedang
# Kita perlu mengkonversi nilai-nilai ini ke format numerik menggunakan encoder yang sama
new_data = pd.DataFrame({
    'Kartu': ['Prabayar'],
    'Panggilan': ['Cukup'],
    'Block': ['Sedang']
})

# Mengkonversi data input menggunakan label_encoders yang sudah ada
for col in new_data.columns:
    new_data[col] = label_encoders[col].transform(new_data[col])

prediction = model.predict(new_data)
predicted_class = label_encoders['Bonus'].inverse_transform(prediction)

print(f"Kondisi: Kartu=Prabayar, Panggilan=Cukup, Block=Sedang")
print(f"Prediksi 'Bonus': {predicted_class[0]}")
