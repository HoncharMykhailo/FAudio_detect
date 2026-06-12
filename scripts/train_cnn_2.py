import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, \
    BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Помилка налаштування GPU: {e}")

DATA_DIRS = [
  #  r"data\in_the_wild",
   r"data\f2s",
  #  r"data\HomeBrew"
]
MODELS_DIR = r"training\HomeBrew"

MAX_SAMPLES_PER_CLASS = 32*700 #batch size * k

os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILES = {
    'wave': 'deepfake_wave_1d.keras',
    'spec': 'deepfake_spec_2d.keras',
    'fft': 'deepfake_fft_1d.keras',
    'mel': 'deepfake_cnn.keras',
    'lfcc': 'deepfake_lfcc_cnn.keras'
}


def build_cnn_2d(input_shape):
    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.3),
        Flatten(), Dense(64, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_1d(input_shape):
    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        Conv1D(16, 5, activation='relu'), MaxPooling1D(4),
        Conv1D(32, 5, activation='relu'), MaxPooling1D(4), Dropout(0.3),
        Flatten(), Dense(32, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def load_and_combine(feat_name, split_name):
    """
    Завантажує дані з кількох датасетів, бере не більше MAX_SAMPLES_PER_CLASS
    файлів кожного класу з одного датасету, безпечно для RAM.
    """
    total_samples = 0
    feature_shape = None

    for data_dir in DATA_DIRS:
        fake_path = os.path.join(data_dir, split_name, "fake", f"{feat_name}.npz")
        real_path = os.path.join(data_dir, split_name, "real", f"{feat_name}.npz")

        if os.path.exists(fake_path):
            with np.load(fake_path) as fake_data:
                n_fake = min(fake_data['X'].shape[0], MAX_SAMPLES_PER_CLASS)
                total_samples += n_fake
                if feature_shape is None: feature_shape = fake_data['X'].shape[1:]

        if os.path.exists(real_path):
            with np.load(real_path) as real_data:
                n_real = min(real_data['X'].shape[0], MAX_SAMPLES_PER_CLASS)
                total_samples += n_real
                if feature_shape is None: feature_shape = real_data['X'].shape[1:]

    if total_samples == 0:
        raise ValueError(f"Дані для ознаки {feat_name} не знайдені в жодному з датасетів!")

    new_shape = (total_samples,) + feature_shape
    X = np.empty(new_shape, dtype=np.float32)
    y = np.empty((total_samples,), dtype=np.int8)

    current_idx = 0

    print(f"  [Виділено пам'ять під {total_samples} зразків (max: {MAX_SAMPLES_PER_CLASS} на клас з датасету)]")

    for data_dir in DATA_DIRS:
        fake_path = os.path.join(data_dir, split_name, "fake", f"{feat_name}.npz")
        if os.path.exists(fake_path):
            print(f"  Читання: {fake_path} ...")
            with np.load(fake_path) as fake_data:
                n = min(fake_data['X'].shape[0], MAX_SAMPLES_PER_CLASS)
                X[current_idx:current_idx + n] = fake_data['X'][:n]
                y[current_idx:current_idx + n] = fake_data['y'][:n]
                current_idx += n

    for data_dir in DATA_DIRS:
        real_path = os.path.join(data_dir, split_name, "real", f"{feat_name}.npz")
        if os.path.exists(real_path):
            print(f"  Читання: {real_path} ...")
            with np.load(real_path) as real_data:
                n = min(real_data['X'].shape[0], MAX_SAMPLES_PER_CLASS)
                X[current_idx:current_idx + n] = real_data['X'][:n]
                y[current_idx:current_idx + n] = real_data['y'][:n]
                current_idx += n

    gc.collect()
    return X, y


def prepare_judge_data():
    print("\n" + "=" * 50)
    print("⚖️ ФАЗА 2: ПІДГОТОВКА ДАНИХ ДЛЯ СУДДІ (META-LEARNER)")
    print("=" * 50)

    preds_list = []
    y_judge = None

    feature_order = ['mel', 'lfcc', 'wave', 'spec', 'fft']

    for feat_name in feature_order:
        print(f"\n🧠 Обробка {feat_name.upper()}...")

        model_path = os.path.join(MODELS_DIR, MODEL_FILES[feat_name])
        model = load_model(model_path)

        X_val, y_val = load_and_combine(feat_name, 'validate')

        if y_judge is None:
            y_judge = y_val

        pred = model.predict(X_val, batch_size=64, verbose=1)
        preds_list.append(pred)

        del X_val, y_val, model
        gc.collect()
        tf.keras.backend.clear_session()

    X_judge = np.column_stack(preds_list)
    return X_judge, y_judge


def train_meta_learner():
    X_judge, y_judge = prepare_judge_data()

    print(f"\n✅ Дані для Judge готові! Розмір матриці: {X_judge.shape}")

    judge_model = Sequential([
        Dense(16, activation='relu', input_shape=(5,)),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    judge_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("🚀 Тренування Judge...")
    judge_model.fit(X_judge, y_judge, epochs=50, batch_size=16, validation_split=0.2)

    judge_path = os.path.join(MODELS_DIR, "deepfake_judge.keras")
    judge_model.save(judge_path)
    print(f"✅ ФІНАЛ: Модель Judge успішно збережена як '{judge_path}'!")


if __name__ == "__main__":
    print("=== ПОЧАТОК ГЛОБАЛЬНОГО ТРЕНУВАННЯ (MIXED DATASETS) ===")

    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    models_config = [
        ('wave', MODEL_FILES['wave'], build_cnn_1d),
        ('spec', MODEL_FILES['spec'], build_cnn_2d),
        ('fft', MODEL_FILES['fft'], build_cnn_1d),
        ('mel', MODEL_FILES['mel'], build_cnn_2d),
        ('lfcc', MODEL_FILES['lfcc'], build_cnn_2d)
    ]

    for feat_name, filename, builder in models_config:
        print("\n" + "=" * 50)
        print(f"🚀 ФАЗА 1: ТРЕНУВАННЯ БАЗОВОЇ МОДЕЛІ: {feat_name.upper()}")
        print("=" * 50)

        X_train, y_train = load_and_combine(feat_name, 'training')
        X_test, y_test = load_and_combine(feat_name, 'test')

        model = builder(X_train.shape[1:])

        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=32,
            shuffle=True,  
            validation_data=(X_test, y_test),
            callbacks=[early_stop]
        )

        save_path = os.path.join(MODELS_DIR, filename)
        model.save(save_path)
        print(f"✅ Базова модель збережена як: {save_path}")

        del X_train, y_train, X_test, y_test, model
        gc.collect()
        tf.keras.backend.clear_session()

    train_meta_learner()
    print("\n🎉 Увесь конвеєр машинного навчання завершено успішно!")
