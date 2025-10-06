# src/train.py
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='../data/train')
parser.add_argument('--val_dir', type=str, default='../data/val')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--fine_tune_lr', type=float, default=1e-5)
parser.add_argument('--model_out', type=str, default='../models/best_model.h5')
args = parser.parse_args()

IMG_SIZE = (args.img_size, args.img_size)
BATCH = args.batch

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(args.train_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
val_flow = val_gen.flow_from_directory(args.val_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

NUM_CLASSES = len(train_flow.class_indices)
print("Clases detectadas:", train_flow.class_indices)
print("Número de clases:", NUM_CLASSES)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

# Freeze base
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(args.base_lr), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(args.model_out, save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss', verbose=1)
]

history = model.fit(train_flow, validation_data=val_flow, epochs=args.epochs, callbacks=callbacks)

# Fine-tune: descongelar últimas N capas
for layer in base.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(args.fine_tune_lr), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft = model.fit(train_flow, validation_data=val_flow, epochs=10, callbacks=callbacks)

# Guardar clase -> índice para su uso en inferencia
import os
os.makedirs('models', exist_ok=True)  # crea carpeta si no existe

# Guardar modelo entrenado
model.save('models/final_trash_classifier.h5')

# Guardar índices de clases
import json
with open('models/class_indices.json', 'w') as f:
    json.dump(train_flow.class_indices, f)

print("✅ Modelo y class_indices guardados correctamente en 'models/'")