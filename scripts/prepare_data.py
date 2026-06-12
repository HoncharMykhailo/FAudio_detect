import os
import gc
import numpy as np
import librosa
import scipy.fftpack
import concurrent.futures

# ==========================================
# 1. НАЛАШТУВАННЯ ШЛЯХІВ
# ==========================================

#TEST_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\for-2seconds\testing\fake"
#TEST_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\for-2seconds\testing\real"
#TRAIN_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\for-2seconds\training\fake"
#TRAIN_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\for-2seconds\training\real"
#VAL_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\for-2seconds\validation\fake"
#VAL_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\for-2seconds\validation\real"
#OUTPUT_BASE_DIR = r"data\f2s"

#TEST_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\in the wild\release_in_the_wild\test\fake"
#TEST_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\in the wild\release_in_the_wild\test\real"
#TRAIN_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\in the wild\release_in_the_wild\train\fake"
#TRAIN_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\in the wild\release_in_the_wild\train\real"
#VAL_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\in the wild\release_in_the_wild\val\fake"
#VAL_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\in the wild\release_in_the_wild\val\real"
#OUTPUT_BASE_DIR = r"data\in_the_wild"


TEST_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\HomeBrew\test\fake"
TEST_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\HomeBrew\test\real"
TRAIN_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\HomeBrew\train\fake"
TRAIN_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\HomeBrew\train\real"
VAL_FAKE_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\HomeBrew\val\fake"
VAL_REAL_DIR = r"C:\Users\my380\Documents\education\4c2\diploma\t3\audio\HomeBrew\val\real"
OUTPUT_BASE_DIR = r"data\HomeBrew"



SAMPLE_RATE = 44100
DUR = 1.0
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * DUR)
MIN_RMS = 0.005

# Скільки логічних процесорів використовувати (рекомендую 10-12 з 16, щоб не перевантажити RAM)
MAX_WORKERS = 12


# ==========================================
# 2. ФУНКЦІЇ ОБРОБКИ (ВОРКЕРИ)
# ==========================================
def extract_lfcc(y, sr, n_mfcc=40, n_filters=128):
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
    log_linear_spec = librosa.power_to_db(np.dot(lin_filters, S))
    return scipy.fftpack.dct(log_linear_spec, axis=0, type=2, norm='ortho')[:n_mfcc]


def process_chunk(chunk, sr):
    wave = librosa.resample(chunk, orig_sr=sr, target_sr=8000)
    spec = librosa.power_to_db(np.abs(librosa.stft(chunk, n_fft=1024, hop_length=512)), ref=np.max)
    fft = np.abs(np.fft.rfft(chunk))
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000), ref=np.max)
    lfcc = extract_lfcc(chunk, sr)
    return wave, spec, fft, mel, lfcc


# Ця функція виконується в окремому процесі для ОДНОГО файлу
def process_single_file(args):
    file_path, label_val = args
    file_data = {'wave': [], 'spec': [], 'fft': [], 'mel': [], 'lfcc': [], 'y': []}

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        for i in range(0, len(y), SAMPLES_PER_CHUNK):
            chunk = y[i:i + SAMPLES_PER_CHUNK]
            if len(chunk) < SAMPLES_PER_CHUNK // 2: continue
            if float(np.sqrt(np.mean(chunk ** 2))) < MIN_RMS: continue
            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)), "constant")

            wave, spec, fft, mel, lfcc = process_chunk(chunk, sr)
            file_data['wave'].append(wave)
            file_data['spec'].append(spec)
            file_data['fft'].append(fft)
            file_data['mel'].append(mel)
            file_data['lfcc'].append(lfcc)
            file_data['y'].append(label_val)
    except Exception as e:
        pass  # Ігноруємо биті файли

    return file_data


# ==========================================
# 3. ОСНОВНА ФУНКЦІЯ ОРКЕСТРАЦІЇ
# ==========================================
def process_and_save_folder(folder_path, split_name, label_name, label_val):
    print(f"\n📂 Відкриваємо папку: {folder_path}")
    print(f"   Призначення: {split_name} | Клас: {label_name} | Потоків: {MAX_WORKERS}")

    output_dir = os.path.join(OUTPUT_BASE_DIR, split_name, label_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(folder_path):
        print(f"⚠️ Папка не знайдена: {folder_path}")
        return

    valid_files = [f for f in os.listdir(folder_path) if f.endswith((".wav", ".mp3", ".flac"))]
    total_files = len(valid_files)

    if total_files == 0:
        print("⚠️ Папка порожня!")
        return

    # Підготовлюємо аргументи для багатопроцесорності
    file_paths = [(os.path.join(folder_path, f), label_val) for f in valid_files]
    main_data = {'wave': [], 'spec': [], 'fft': [], 'mel': [], 'lfcc': [], 'y': []}

    print(f"🚀 Запуск паралельної обробки {total_files} файлів...")

    # Використовуємо ProcessPoolExecutor для паралельних обчислень
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # submit відправляє завдання у пул
        futures = {executor.submit(process_single_file, args): args for args in file_paths}

        last_percent = -1
        completed = 0

        # Обробляємо результати по мірі їх готовності
        for future in concurrent.futures.as_completed(futures):
            result = future.result()

            # Додаємо результати окремого файлу до загального масиву
            if len(result['y']) > 0:
                for key in main_data.keys():
                    main_data[key].extend(result[key])

            # Прогрес-бар
            completed += 1
            current_percent = int((completed / total_files) * 100)
            if current_percent > last_percent:
                bar = '█' * (current_percent // 2) + '-' * (50 - (current_percent // 2))
                print(f"\rПрогрес: [{bar}] {current_percent}% ({completed}/{total_files})", end="")
                last_percent = current_percent

    print("\n💾 Форматування та збереження архівів...")

    if len(main_data['y']) == 0:
        print("⚠️ Жодного придатного аудіо-фрагмента не знайдено.")
        return

    y_array = np.array(main_data['y'], dtype=np.float32)
    for feat in ['wave', 'spec', 'fft', 'mel', 'lfcc']:
        feat_array = np.array(main_data[feat], dtype=np.float32)[..., np.newaxis]

        filename = os.path.join(output_dir, f"{feat}.npz")
        np.savez_compressed(filename, X=feat_array, y=y_array)
        print(f"  ✅ Збережено: {filename} (Розмір: {feat_array.shape})")

        del feat_array

    del main_data, y_array
    gc.collect()


if __name__ == "__main__":
    # Цей блок if __name__ == "__main__": є КРИТИЧНО ВАЖЛИВИМ для ProcessPoolExecutor у Windows!
    print("=== ПОЧАТОК ПАРАЛЕЛЬНОЇ ОБРОБКИ ДАНИХ ===")

    process_and_save_folder(TRAIN_FAKE_DIR, "training", "fake", label_val=1)
    process_and_save_folder(TRAIN_REAL_DIR, "training", "real", label_val=0)

    process_and_save_folder(TEST_FAKE_DIR, "test", "fake", label_val=1)
    process_and_save_folder(TEST_REAL_DIR, "test", "real", label_val=0)

    process_and_save_folder(VAL_FAKE_DIR, "validate", "fake", label_val=1)
    process_and_save_folder(VAL_REAL_DIR, "validate", "real", label_val=0)

    print("\n🎉 Усі дані успішно підготовлені значно швидше!")
