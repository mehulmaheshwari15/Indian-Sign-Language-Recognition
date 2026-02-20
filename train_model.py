import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
MODEL_SAVE  = "isl_model.h5"

# ── Data Loading & Augmentation ──────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,          # 80 / 20 split
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)

# ── Print detected class labels ──────────────────────────────────────────────
class_indices = train_gen.class_indices          # {"A": 0, "B": 1, ...}
num_classes   = len(class_indices)

print("=" * 50)
print(f"Detected {num_classes} classes:")
print(sorted(class_indices.keys()))
print(f"Training samples   : {train_gen.samples}")
print(f"Validation samples : {val_gen.samples}")
print("=" * 50)

# ── Model: MobileNetV2 Transfer Learning ────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False          # freeze pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
]

# ── Training ─────────────────────────────────────────────────────────────────
print("\nStarting training …\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print(f"\nBest model saved to: {MODEL_SAVE}")
