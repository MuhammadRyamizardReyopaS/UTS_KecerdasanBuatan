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
---
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
---


---

## 📌 Catatan

**Disclaimer:** Model ini dibuat untuk tujuan edukatif. Penggunaan untuk deteksi ujaran kebencian di dunia nyata memerlukan pengembangan lebih lanjut dengan dataset yang lebih besar dan diverse.

---
