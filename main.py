## main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from data_loader import compute_histograms, get_first_20_image_paths
from kmeans import KMeansManual
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from random import choice

def extract_label_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 0:
        return ''.join(filter(str.isalpha, parts[0]))
    return ""

class_names = ['red', 'blue', 'green', 'gray', 'white']
train_image_paths = get_first_20_image_paths('C:/Users/EfeBasol/Documents/Kmeans/train', class_names)
train_data = compute_histograms(train_image_paths)
kmeans = KMeansManual(n_clusters=5, max_iters=100)
kmeans.fit(train_data)

# Her bir küme için etiketleri say
clusters = kmeans.assign_clusters(train_data)
cluster_labels = {i: [] for i in range(kmeans.n_clusters)}
for idx, cluster in enumerate(clusters):
    for data_idx in cluster:
        filename = os.path.basename(train_image_paths[data_idx])
        label = extract_label_from_filename(filename)
        cluster_labels[idx].append(label)

# Her kümenin hangi renk sınıfını temsil ettiğini anlamak için etiketleri say
for cluster_idx, labels in cluster_labels.items():
    print(f"Küme {cluster_idx}:")
    for class_name in class_names:
        count = labels.count(class_name)
        print(f"  {class_name} sayısı: {count}")

## Deneysel sonuçların elde edilmesi
true_labels = [extract_label_from_filename(os.path.basename(path)) for path in train_image_paths]
accuracy_list = kmeans.repeat_kmeans(train_data, true_labels, num_repeats=10)

print("Her tekrarlama için doğruluk oranları:")
for i, accuracy in enumerate(accuracy_list):
    print(f"Tekrarlama {i+1}: Doğruluk = {accuracy:.2f}")

## Gerçek ve Tahmin Edilen Etiketleri Hazırlama
true_labels = [extract_label_from_filename(os.path.basename(path)) for path in train_image_paths]

# Kümeler için en yaygın etiketleri hesaplama
predicted_labels = []
for cluster in clusters:
    label_count = {label: 0 for label in class_names}
    label_count = {label: 0 for label in class_names}
    for idx in cluster:
        label = extract_label_from_filename(os.path.basename(train_image_paths[idx]))
        label_count[label] += 1
    most_common_label = max(label_count, key=label_count.get)
    predicted_labels.extend([most_common_label] * len(cluster))

# Karışıklık Matrisini Hesaplama ve Görselleştirme
cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Karışıklık Matrisi')
plt.ylabel('Gerçek Etiketler')
plt.xlabel('Tahmin Edilen Etiketler')
plt.show()


