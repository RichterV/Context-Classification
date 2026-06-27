import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D,GlobalAveragePooling2D, MaxPooling2D, Dropout,
                                     concatenate, UpSampling2D, BatchNormalization,
                                     Flatten, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import yaml

# Carrega o arquivo de configuração
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Configurações gerais
imagens = config["imagens"]
contextos = config["contextos"]
patience = config["training"]["patience"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
test_size_ = config["training"]["test_size"]
seed_value = config["random_seed"]
learning_rate = config["training"]["learning_rate"]
reduce_lr_patience = config["training"]["reduce_lr"]["patience"]
reduce_lr_factor   = config["training"]["reduce_lr"]["factor"]
reduce_lr_min_lr   = config["training"]["reduce_lr"]["min_lr"]

# Augmentação
augment_config = config["training"]["augment"]

# Definir seeds
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# Configurar GPU uma única vez antes do loop
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'{len(gpus)} GPU(s) detectadas e configuradas para uso.')
    except RuntimeError as e:
        print(e)
else:
    print('Nenhuma GPU detectada. Usando CPU.')

class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=2, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = [self.add_weight(name=f'tp_{i}', initializer='zeros') for i in range(num_classes)]
        self.fp = [self.add_weight(name=f'fp_{i}', initializer='zeros') for i in range(num_classes)]
        self.fn = [self.add_weight(name=f'fn_{i}', initializer='zeros') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_cls = tf.argmax(y_pred, axis=-1)
        y_true_cls = tf.argmax(y_true, axis=-1)
        for i in range(self.num_classes):
            pred_i = tf.cast(tf.equal(y_pred_cls, i), tf.float32)
            true_i = tf.cast(tf.equal(y_true_cls, i), tf.float32)
            self.tp[i].assign_add(tf.reduce_sum(pred_i * true_i))
            self.fp[i].assign_add(tf.reduce_sum(pred_i * (1 - true_i)))
            self.fn[i].assign_add(tf.reduce_sum((1 - pred_i) * true_i))

    def result(self):
        f1_list = []
        for i in range(self.num_classes):
            precision = self.tp[i] / (self.tp[i] + self.fp[i] + 1e-7)
            recall    = self.tp[i] / (self.tp[i] + self.fn[i] + 1e-7)
            f1_list.append(2 * precision * recall / (precision + recall + 1e-7))
        return tf.reduce_mean(tf.stack(f1_list))

    def reset_state(self):
        for i in range(self.num_classes):
            self.tp[i].assign(0.)
            self.fp[i].assign(0.)
            self.fn[i].assign(0.)


# Bloco de convolução com normalização e dropout
def conv_block(x, filters, dropout_rate=0.2):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x

# Modelo U-Net++ aprimorado

def unetpp_classifier_model(input_shape=(40, 40, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    x00 = conv_block(inputs, 32)
    p0 = MaxPooling2D((2, 2))(x00)

    x10 = conv_block(p0, 64)
    p1 = MaxPooling2D((2, 2))(x10)

    x20 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(x20)

    x30 = conv_block(p2, 256)

    x01 = conv_block(concatenate([UpSampling2D((2, 2))(x10), x00]), 32)
    x11 = conv_block(concatenate([UpSampling2D((2, 2))(x20), x10]), 64)
    x21 = conv_block(concatenate([UpSampling2D((2, 2))(x30), x20]), 128)

    x02 = conv_block(concatenate([UpSampling2D((2, 2))(x11), x00, x01]), 32)
    x12 = conv_block(concatenate([UpSampling2D((2, 2))(x21), x10, x11]), 64)

    x03 = conv_block(concatenate([UpSampling2D((2, 2))(x12), x00, x01, x02]), 32)
    
    gap = GlobalAveragePooling2D()(x03)
    dense = Dense(64, activation='relu')(gap)

    output = Dense(num_classes, activation='softmax')(dense)

    return Model(inputs=inputs, outputs=output)

    
# Loop por contexto e tipo de imagem
for context_size in contextos:
    for img_type in imagens:
        K.clear_session()

        print("\n" + "=" * 100)
        print(f"Iniciando treino para imagem '{img_type}' com contexto {context_size}x{context_size}")

        img_name = os.path.splitext(os.path.basename(img_type))[0]

        try:
            X = np.load(f'context_data/X_context_rgb_{context_size}_{img_name}.npy')
            Y = np.load(f'context_data/Y_labels_{context_size}_{img_name}.npy')
        except FileNotFoundError:
            print(f"⚠️ Arquivos não encontrados para {img_name}. Pulei.")
            continue

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size_, random_state=seed_value, stratify=Y.argmax(axis=1)
        )

        datagen = ImageDataGenerator(
            rotation_range=augment_config.get("rotation_range", 0),
            zoom_range=augment_config.get("zoom_range", 0),
            horizontal_flip=augment_config.get("horizontal_flip", False),
            vertical_flip=augment_config.get("vertical_flip", False),
        )

        y_integers = Y.argmax(axis=1)
        weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
        class_weight = dict(enumerate(weights))
        print(f"Class weights → não-floresta: {class_weight[0]:.3f} | floresta: {class_weight[1]:.3f}")

        input_channels = X.shape[-1]
        model = unetpp_classifier_model(input_shape=(context_size, context_size, input_channels), num_classes=2)

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(class_id=1, name='precision'),
                tf.keras.metrics.Recall(class_id=1, name='recall'),
                MacroF1Score(num_classes=2, name='f1_score'),
            ]
        )

        log_dir = os.path.join("logs", img_name, str(context_size))
        os.makedirs(f"context_models/{img_name}", exist_ok=True)

        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping_callback = EarlyStopping(monitor='val_f1_score', patience=patience, restore_best_weights=True, mode='max')
        checkpoint_callback = ModelCheckpoint(
            filepath=f'context_models/{img_name}/best_model_{context_size}_{img_name}.keras',
            monitor='val_f1_score',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            verbose=1,
            mode='max',
            min_lr=reduce_lr_min_lr
        )

        steps_per_epoch = len(X_train) // batch_size

        history = model.fit(
            datagen.flow(X_train, Y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_test, Y_test),
            epochs=epochs,
            verbose=2,
            class_weight=class_weight,
            callbacks=[tensorboard_callback, early_stopping_callback, checkpoint_callback, reduce_lr]
        )

        results = model.evaluate(X_test, Y_test, verbose=0)
        metrics_names = model.metrics_names
        print(f'\n✅ Modelo para {img_name} finalizado:')
        for name, val in zip(metrics_names, results):
            print(f'   → {name}: {val:.4f}')
