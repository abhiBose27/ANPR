import tensorflow as tf
from keras import Sequential, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import image_dataset_from_directory

# Config
IMG_SIZE = (28, 28)
BATCH_SIZE = 32
REAL_DATASET_DIR = "datasets/preprocessed_cnn_dataset1/"
SYNTHETIC_DATASET_DIR = "datasets/preprocessed_cnn_dataset/"
EPOCHS_PRETRAIN = 5
EPOCHS_FINETUNE = 5

# Load datasets
synthetic_train_ds = image_dataset_from_directory(
    SYNTHETIC_DATASET_DIR,
    validation_split=0.2,
    subset='training',
    seed=42,
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

synthetic_val_ds = image_dataset_from_directory(
    SYNTHETIC_DATASET_DIR,
    validation_split=0.2,
    subset='validation',
    seed=42,
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

real_train_ds = image_dataset_from_directory(
    REAL_DATASET_DIR,
    validation_split=0.2,
    subset='training',
    seed=42,
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

real_val_ds = image_dataset_from_directory(
    REAL_DATASET_DIR,
    validation_split=0.2,
    subset='validation',
    seed=42,
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

""" data_augmentation = Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
]) """

# Normalize pixels
normalization_layer = layers.Rescaling(1./255)
synthetic_train_ds = synthetic_train_ds.map(lambda x, y: (normalization_layer(x), y))
synthetic_val_ds = synthetic_val_ds.map(lambda x, y: (normalization_layer(x), y))
real_train_ds = real_train_ds.map(lambda x, y: (normalization_layer(x), y))
real_val_ds = real_val_ds.map(lambda x, y: (normalization_layer(x), y))
#train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x), training=True), y))
#val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch
AUTOTUNE = tf.data.AUTOTUNE
synthetic_train_ds = synthetic_train_ds.prefetch(AUTOTUNE)
synthetic_val_ds = synthetic_val_ds.prefetch(AUTOTUNE)
real_train_ds = real_train_ds.prefetch(AUTOTUNE)
real_val_ds = real_val_ds.prefetch(AUTOTUNE)

# Model definition
model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(36, activation='softmax')  # 0-9 + A-Z = 36 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Save best model
checkpoint = ModelCheckpoint(
    'cnn_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
earlystop_cb = EarlyStopping(
    patience=5, 
    restore_best_weights=True
)

print("🧪 Pretraining on synthetic dataset...")
model.fit(
    synthetic_train_ds,
    validation_data=synthetic_val_ds,
    epochs=EPOCHS_PRETRAIN,
    callbacks=[checkpoint, earlystop_cb],
)

# Step 2: Fine-tune on real dataset
print("🔧 Fine-tuning on real dataset...")
model.fit(
    real_train_ds,
    validation_data=real_val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=[checkpoint, earlystop_cb]
)

print("✅ Training complete. Model saved as cnn_model.keras")
