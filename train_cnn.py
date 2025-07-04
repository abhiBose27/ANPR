import tensorflow as tf
from keras import Sequential, layers, models
from keras.callbacks import ModelCheckpoint
from keras.utils import image_dataset_from_directory
from keras.models import load_model

class CNNModelTrainer:
    def __init__(self, best_keras, last_keras, dataset_paths=None, model_path=None, batch_size=32, img_size=(28, 28), epochs=15, validation_split=0.2, seed=42, nb_classes=36):
        self.best_keras = best_keras
        self.last_keras = last_keras
        self.model_path = model_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.dataset_paths = dataset_paths
        self.epochs = epochs
        self.validation_split = validation_split
        self.seed = seed
        self.nb_classes = nb_classes
        self.model = None
        self.train_ds = None
        self.val_ds = None

    def get_model(self):
        if self.model_path:
            self.model = load_model(self.model_path)
            return
        self.model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*self.img_size, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(self.nb_classes, activation='softmax')  # 0-9 + A-Z = 36 classes
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_training_ds(self):
        print("Merging datasets from: ", self.dataset_paths)
        train_datasets = []
        val_datasets = []
        for dataset_path in self.dataset_paths:
            train_ds_raw = image_dataset_from_directory(
                directory=dataset_path,
                validation_split=self.validation_split,
                subset="training",
                color_mode="grayscale",
                image_size=self.img_size,
                batch_size=self.batch_size,
                seed=self.seed
            )
            val_ds = image_dataset_from_directory(
                directory=dataset_path,
                validation_split=self.validation_split,
                subset="validation",
                color_mode="grayscale",
                image_size=self.img_size,
                batch_size=self.batch_size,
                seed=self.seed
            )
            train_datasets.append(train_ds_raw)
            val_datasets.append(val_ds)
        
        data_augmentation = models.Sequential([
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ])

        raw_train_ds = train_datasets[0]
        for ds in train_datasets[1:]:
            raw_train_ds = raw_train_ds.concatenate(ds)

        self.val_ds = val_datasets[0]
        for ds in val_datasets[1:]:
            self.val_ds = self.val_ds.concatenate(ds)

        normalization_layer = layers.Rescaling(1./255)
        autotune = tf.data.AUTOTUNE
        self.train_ds = raw_train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
        self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        self.val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=autotune)

    def train(self):
        checkpoint = ModelCheckpoint(
            filepath=self.best_keras,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=[checkpoint],
        )

    def evaluate_and_save(self):
        loss, accuracy = self.model.evaluate(self.val_ds)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        self.model.save(self.last_keras)

if __name__ == "__main__":
    trainer = CNNModelTrainer(
        best_keras="cnn_model/best1.keras",
        last_keras="cnn_model/last1.keras",
        dataset_paths=["datasets/cnn_dataset", "datasets/cnn_dataset2"]
    )
    trainer.get_model()
    trainer.get_training_ds()
    trainer.train()
    trainer.evaluate_and_save()