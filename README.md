# Clasificador de Residuos
Este proyecto es un clasificador de residuos basado en una red neuronal convolucional (CNN) utilizando TensorFlow y Keras. El modelo clasifica imágenes de residuos en seis categorías: Cartón, Vidrio, Metal, Papel, Plástico y Otros. Además, cuenta con una interfaz gráfica simple construida con streamlit para facilitar la carga y clasificación de imágenes.

## Requisitos
Antes de ejecutar el proyecto, asegúrate de tener instalados los siguientes paquetes:

Python 3.x
TensorFlow
NumPy
Matplotlib
Seaborn
Pillow
scikit-learn
Tkinter (generalmente incluido en instalaciones de Python)
Puedes instalar las dependencias necesarias utilizando pip:

## Instalación
pip install -r requirements.txt

## Estructura
(data/train, data/val, data/test)

## Entrenar
python src/train.py --train_dir data/train --val_dir data/val

## Evaluar
python src/evaluate.py --model models/final_trash_classifier.h5 --test_dir data/test

## Demo (Streamlit)
streamlit run src/app_streamlit.py

