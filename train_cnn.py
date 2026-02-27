import os
import numpy as np
import librosa
import scipy.fftpack
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# ==========================================
# 1. ШЛЯХИ ДО ПАПОК
# ==========================================
TRAIN_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t2\for-2seconds\training\fake"
TRAIN_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t2\for-2seconds\training\real"

TEST_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t2\for-2seconds\testing\fake"
TEST_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t2\for-2seconds\testing\real"

SAMPLE_RATE = 22050
DURATION = 1.0  # Нарізка
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * DURATION)
MIN_RMS = 0.005  # Поріг гучності


# ==========================================
# 2. ФУНКЦІЇ ЕКСТРАКЦІЇ ОЗНАК
# ==========================================
def extract_lfcc(y, sr, n_lfcc=40, n_filters=128):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) ** 2
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    filter_freqs = np.linspace(0, sr / 2, n_filters + 2)

    lin_filters = np.zeros((n_filters, len(fft_freqs)))
    for i in range(n_filters):
        lower, center, upper = filter_freqs[i], filter_freqs[i + 1], filter_freqs[i + 2]
        for j, f in enumerate(fft_freqs):
            if lower <= f <= center:
                lin_filters[i, j] = (f - lower) / (center - lower)
            elif center < f <= upper:
                lin_filters[i, j] = (upper - f) / (upper - center)

    linear_spec = np.dot(lin_filters, S)
    log_linear_spec = librosa.power_to_db(linear_spec)
    return scipy.fftpack.dct(log_linear_spec, axis=0, type=2, norm='ortho')[:n_lfcc]


def process_file_dual_features(file_path):
    mels, lfccs = [], []
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        step = SAMPLES_PER_CHUNK

        for i in range(0, len(y), step):
            chunk = y[i:i + step]

            if len(chunk) < step // 2: continue
            if float(np.sqrt(np.mean(chunk ** 2))) < MIN_RMS: continue

            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)), "constant")

            mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
            mels.append(librosa.power_to_db(mel, ref=np.max))

            lfccs.append(extract_lfcc(chunk, sr))

        return mels, lfccs
    except Exception as e:
        print(f"Помилка {file_path}: {e}")
        return [], []


# ==========================================
# 3. ЗАВАНТАЖЕННЯ ДАНИХ (ОДНОЧАСНО)
# ==========================================
def load_dual_dataset(real_dir, fake_dir, dataset_name=""):
    X_mel, X_lfcc, y = [], [], []

    print(f"⏳ Завантаження {dataset_name} REAL...")
    for file in os.listdir(real_dir):
        if file.endswith((".wav", ".mp3", ".flac", ".ogg")):
            mels, lfccs = process_file_dual_features(os.path.join(real_dir, file))
            X_mel.extend(mels)
            X_lfcc.extend(lfccs)
            y.extend([0] * len(mels))

    print(f"⏳ Завантаження {dataset_name} FAKE...")
    for file in os.listdir(fake_dir):
        if file.endswith((".wav", ".mp3", ".flac", ".ogg")):
            mels, lfccs = process_file_dual_features(os.path.join(fake_dir, file))
            X_mel.extend(mels)
            X_lfcc.extend(lfccs)
            y.extend([1] * len(mels))

    # Перетворюємо в numpy масиви і додаємо канал (1) для Conv2D
    return np.array(X_mel)[..., np.newaxis], np.array(X_lfcc)[..., np.newaxis], np.array(y)



# ==========================================
# 4. CNN
# ==========================================
def build_cnn(input_shape):
    model = Sequential([
        BatchNormalization(input_shape=input_shape),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# ==========================================
# 5. ГОЛОВНИЙ БЛОК
# ==========================================
if __name__ == "__main__":
    print("\n--- ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ (Mel + LFCC) ---")
    X_train_mel, X_train_lfcc, y_train = load_dual_dataset(TRAIN_REAL_DIR, TRAIN_FAKE_DIR, "TRAIN")
    X_test_mel, X_test_lfcc, y_test = load_dual_dataset(TEST_REAL_DIR, TEST_FAKE_DIR, "TEST")

    print(f"\n✅ Дані готові! Тренувальних: {len(y_train)}, Тестових: {len(y_test)}")


    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

    # --- ТРЕНУВАННЯ MEL ---
    print("\n" + "=" * 40)
    print("🚀 ПОЧАТОК ТРЕНУВАННЯ: MEL Спектрограма")
    print("=" * 40)
    mel_model = build_cnn(X_train_mel.shape[1:])
    mel_model.fit(
        X_train_mel, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test_mel, y_test),
        callbacks=[early_stop]
    )
    mel_model.save("deepfake_cnn.keras")
    print("🎉 Найкраща Mel-модель збережена!")

    # --- ТРЕНУВАННЯ LFCC ---
    print("\n" + "=" * 40)
    print("🚀 ПОЧАТОК ТРЕНУВАННЯ: LFCC Кепстр")
    print("=" * 40)
    lfcc_model = build_cnn(X_train_lfcc.shape[1:])
    lfcc_model.fit(
        X_train_lfcc, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test_lfcc, y_test),
        callbacks=[early_stop]
    )
    lfcc_model.save("deepfake_lfcc_cnn.keras")
    print("🎉 Найкраща LFCC-модель збережена!")

    print("\n✅ УСІ ПРОЦЕСИ ЗАВЕРШЕНО! Обидві моделі готові для app.py.")