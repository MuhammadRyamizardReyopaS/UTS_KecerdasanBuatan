# Hate Speech Detection - Indonesian Language

Klasifikasi Ujaran Kebencian (Hate Speech) dalam Bahasa Indonesia menggunakan **Naive Bayes** dan **TF-IDF**. Proyek ini merupakan implementasi dari tugas UTS Kecerdasan Buatan untuk mendeteksi konten berbahaya di media sosial.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)](https://scikit-learn.org/)

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk **mendeteksi ujaran kebencian** dalam tweet berbahasa Indonesia menggunakan pendekatan Machine Learning. Dataset yang digunakan adalah **ID Hate Speech Dataset** yang terdiri dari 713 tweet dengan label:
- **Non_HS** (Non Hate Speech): Tweet yang tidak mengandung ujaran kebencian
- **HS** (Hate Speech): Tweet yang mengandung ujaran kebencian

Model yang dibangun mampu mengklasifikasikan teks dengan akurasi **~88%** menggunakan algoritma **Multinomial Naive Bayes** dan representasi fitur **TF-IDF**.

---

## 🎯 Fitur Utama

- ✅ **Preprocessing Teks Lengkap** (Case folding, Cleaning, Tokenisasi, Stopword Removal, Stemming)
- ✅ **Handling Imbalanced Data** dengan Under-Sampling
- ✅ **TF-IDF Feature Extraction** (Unigram + Bigram)
- ✅ **Model Training** dengan Naive Bayes
- ✅ **Evaluasi Komprehensif** (Confusion Matrix, Accuracy, Precision, Recall, F1-Score)
- ✅ **Visualisasi Data** (Word Cloud, Charts, Heatmap)

---

## 📊 Dataset

**Sumber:** https://github.com/ialfina/id-hatespeech-detection.git

**Catatan:** Dataset tidak seimbang (imbalanced), sehingga dilakukan **Random Under-Sampling** untuk menciptakan distribusi 50%-50%.

---

## 🚀 Instalasi & Penggunaan
## ✅ Langkah 1 – Import Library
# Install library jika belum ada
!pip install Sastrawi wordcloud --quiet

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
## Load Dataset & batasi 50 kolom pertama
df = pd.read_csv('ujaran_dataset.csv')

# Ambil hanya 100 baris pertama
df = df.head(50)

# Preprocessing
df['clean_text'] = df['tweet'].astype(str).apply(preprocess)

df.head()
## Membuat dua label untuk Dataset
df['class'].value_counts().plot(kind='bar')
plt.title('Distribusi Label')
plt.xlabel('Label (0=Non Hate, 1=Hate)')
plt.ylabel('Jumlah')
plt.show()
## ✅ Langkah 2 – Preprocessing Teks
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords_id = set(stopwords.words('english'))

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    # lowercase
    text = text.lower()

    # hapus angka & tanda baca
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # tokenisasi
    tokens = nltk.word_tokenize(text)

    # hapus stopword + kata 1 huruf
    tokens = [t for t in tokens if t not in stopwords_id and len(t) > 1]

    # gabung kembali
    text = ' '.join(tokens)

    # stemming
    text = stemmer.stem(text)

    return text

# ✅ pastikan kolom yang dipakai adalah 'tweet'
df['clean_text'] = df['tweet'].astype(str).apply(preprocess)

df.head()

## ✅ Langkah 3 – Representasi Fitur (Bag of Words)
# Langkah 3: Representasi BoW
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['class']
##  ✅ Langkah 4 – Train Model Naive Bayes
# Langkah 4: Training Model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

y_pred = model_nb.predict(X_test)

## ✅ Langkah 5 – Evaluasi Model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))

## ✅ Langkah 6 – Visualisasi Distribusi Label & Word Cloud
# Visualisasi Distribusi Label
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df['class'])
plt.title("Distribusi Label Hate Speech vs Non-Hate Speech")
plt.show()
# Word Cloud (khusus teks hate speech)
from wordcloud import WordCloud

hate_text = ' '.join(df[df['class']==1]['clean_text'])

wordcloud = WordCloud(width=800, height=400).generate(hate_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud Ujaran Kebencian")
plt.show()


---

## 📌 Catatan

**Disclaimer:** Model ini dibuat untuk tujuan edukatif. Penggunaan untuk deteksi ujaran kebencian di dunia nyata memerlukan pengembangan lebih lanjut dengan dataset yang lebih besar dan diverse.

---
