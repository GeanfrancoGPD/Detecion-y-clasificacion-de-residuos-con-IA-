# src/evaluate.py
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse
import os

# Crear carpeta 'reports' si no existe
os.makedirs('reports', exist_ok=True)

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='../models/best_model.h5')
parser.add_argument('--test_dir', type=str, default='../data/test')
parser.add_argument('--img_size', type=int, default=224)
args = parser.parse_args()

# Cargar test
test_gen = ImageDataGenerator(rescale=1./255)
test_flow = test_gen.flow_from_directory(
    args.test_dir,
    target_size=(args.img_size, args.img_size),
    batch_size=1,
    shuffle=False,
    class_mode='categorical'
)

# Cargar modelo
model = load_model(args.model)

# Predicciones
preds = model.predict(test_flow, steps=test_flow.samples, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_flow.classes
labels = list(test_flow.class_indices.keys())

# Clases reales presentes en test
unique_classes = np.unique(y_true)
filtered_labels = [labels[i] for i in unique_classes]

# Reporte de clasificación
print(classification_report(
    y_true,
    y_pred,
    labels=unique_classes,
    target_names=filtered_labels,
    zero_division=0
))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')

# Guardar figura
plt.savefig('reports/confusion_matrix.png', bbox_inches='tight')
plt.close()
print("Matriz de confusión guardada en reports/")
