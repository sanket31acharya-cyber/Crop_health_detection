# <FULL TRAIN SCRIPT HERE>

"""
train_plant_disease.py
Train a plant/crop disease classifier using transfer learning (ResNet50 or MobileNetV3).
Produces: saved model (.h5), training logs, confusion matrix, and classification report.

Usage:
    python train_plant_disease.py
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# -------------------------
# Config / hyperparameters
# -------------------------
SEED = 42
DATA_DIR = "path/to/DATA_DIR"  # <-- EDIT THIS: folder that contains train/val/test
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BACKBONE = "ResNet50"  # options: "ResNet50", "MobileNetV3"
INITIAL_HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
FINE_TUNE_AT = 140  # for ResNet50; for MobileNetV3 pick a small number near end
LR_HEAD = 1e-3
LR_FINE = 1e-5
MODEL_OUTPUT = "plant_disease_model.h5"
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------
# Reproducibility
# -------------------------
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Data loader
# -------------------------
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"train dir not found: {train_dir}. Create train/val/test folders as explained.")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Prefetch for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# -------------------------
# Data augmentation
# -------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.05, 0.05),
])

# -------------------------
# Preprocessing for chosen backbone
# -------------------------
if BACKBONE == "ResNet50":
    preprocess_input = tf.keras.applications.resnet.preprocess_input
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
elif BACKBONE == "MobileNetV3":
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    base_model = tf.keras.applications.MobileNetV3Small(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
else:
    raise ValueError("Unsupported BACKBONE")

base_model.trainable = False  # freeze for head training

# -------------------------
# Build model
# -------------------------
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Callbacks
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_path = f"best_model_{timestamp}.h5"
callbacks = [
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
]

# -------------------------
# Train head
# -------------------------
history_head = model.fit(
    train_ds,
    epochs=INITIAL_HEAD_EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

# -------------------------
# Fine-tuning
# -------------------------
base_model.trainable = True

# Freeze until 'FINE_TUNE_AT'
if BACKBONE == "ResNet50":
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
elif BACKBONE == "MobileNetV3":
    # MobileNetV3 has fewer layers; unfreeze last 30-40 layers usually
    for layer in base_model.layers[:-40]:
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    epochs=INITIAL_HEAD_EPOCHS + FINE_TUNE_EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

# -------------------------
# Save final model
# -------------------------
model.save(MODEL_OUTPUT)
print(f"Saved model to {MODEL_OUTPUT}")

# -------------------------
# Evaluate on test set
# -------------------------
print("Evaluating on test set...")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds_idx = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(preds_idx.tolist())

print("Classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.show()

plot_confusion_matrix(cm, class_names, normalize=False, title="Confusion Matrix")

# Done
print("Training and evaluation completed.")
