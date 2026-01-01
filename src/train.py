import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model # type: ignore

# ---------------------------
# BASIC CONFIG
# ---------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16        # smaller batch helps CPU & generalization
EPOCHS = 40            # early stopping will stop earlier
DATA_DIR = "data/final"

# ---------------------------
# CHECK GPU (OPTIONAL)
# ---------------------------
print("GPUs:", tf.config.list_physical_devices('GPU'))

# ---------------------------
# LOAD DATA
# ---------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# ---------------------------
# NORMALIZATION + AUGMENTATION
# ---------------------------
normalization = tf.keras.layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization(x)), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))

# Performance boost (important on CPU)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ---------------------------
# MODEL (TRANSFER LEARNING)
# ---------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Fine-tuning: freeze most layers, unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.summary()

# ---------------------------
# FOCAL LOSS (FOR HARD CLASSES)
# ---------------------------
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=1)
    return loss

# ---------------------------
# COMPILE
# ---------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# ---------------------------
# CLASS WEIGHTS (BOOST MENINGIOMA)
# Adjust index if class order differs
# ---------------------------
# Example order: ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']
class_weights = {
    0: 1.0,   # glioma
    1: 2.5,   # meningioma (hard class)
    2: 1.0,   # normal
    3: 1.0    # pituitary
}

# ---------------------------
# CALLBACKS
# ---------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=3,
        min_lr=1e-6
    )
]

# ---------------------------
# TRAIN
# ---------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ---------------------------
# SAVE MODEL
# ---------------------------
os.makedirs("models", exist_ok=True)
model.save("models/brain_tumor_model.h5")
print("✅ Final model saved to models/brain_tumor_model.h5")
