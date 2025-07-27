import os
import json
import cv2
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm
import config


def detect_tiff_type(image_path):
    """Определяет тип TIFF файла для выбора стратегии обработки"""
    from PIL import Image

    with Image.open(image_path) as img:
        # Получаем метаданные
        compression = img.info.get("compression", "unknown")
        width, height = img.size

        # Определяем тип на основе характеристик
        if "group" in str(compression).lower() or compression == "group4":
            return "compressed_document"  # Сжатые документы
        elif compression == "raw" or "none" in str(compression).lower():
            return "uncompressed_scan"  # Несжатые сканы
        else:
            return "standard_tiff"  # Стандартные TIFF


def preprocess_image(image_path):
    """Предобработка изображения для улучшения качества OCR"""

    # Определяем тип TIFF для выбора стратегии
    tiff_type = detect_tiff_type(image_path)
    print(f"Тип TIFF: {tiff_type}")

    # Загружаем как grayscale сразу для TIFF файлов
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    print(f"Исходный размер: {image.shape}, среднее значение: {np.mean(image):.1f}")

    # Проверяем нужна ли инверсия (для WhiteIsZero TIFF)
    if np.mean(image) < 128:
        image = cv2.bitwise_not(image)
        print("Применена инверсия")

    # Адаптивное масштабирование в зависимости от размера файла
    height, width = image.shape
    total_pixels = height * width

    print(f"Всего пикселей: {total_pixels:,}")

    # Целевое разрешение для OCR - как обычный скриншот
    target_pixels = 2_000_000  # ~1414x1414 или 1920x1040

    if total_pixels > target_pixels * 2:  # Очень большие - агрессивно уменьшаем
        scale_factor = 0.4  # Сильно уменьшаем
        interpolation = cv2.INTER_AREA
        print("Очень большое изображение - сильно уменьшаем")
    elif total_pixels > target_pixels:  # Большие - уменьшаем
        scale_factor = 0.6  # Умеренно уменьшаем
        interpolation = cv2.INTER_AREA
        print("Большое изображение - уменьшаем")
    elif total_pixels < target_pixels // 4:  # Очень маленькие - увеличиваем
        scale_factor = 2.0  # Увеличиваем маленькие
        interpolation = cv2.INTER_CUBIC
        print("Маленькое изображение - увеличиваем")
    else:  # Оптимальный размер - не трогаем
        scale_factor = 1.0
        interpolation = cv2.INTER_LANCZOS4
        print("Оптимальный размер - не масштабируем")

    if scale_factor != 1.0:
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=interpolation)

    print(f"Финальный размер: {image.shape}")

    # Адаптивная предобработка
    # Упрощенная предобработка - имитируем качество скриншота
    # Минимальная обработка для сохранения естественности

    # Очень легкий денойзинг только если изображение очень зашумленное
    if total_pixels > 4_000_000:  # Только для больших файлов
        denoised = cv2.fastNlMeansDenoising(image, h=5)  # Слабый денойзинг
    else:
        denoised = image  # Маленькие файлы не трогаем

    # Простая OTSU бинаризация - как делают современные сканеры
    _, final_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Использована простая OTSU бинаризация (скриншот-стиль)")

    return final_thresh


def extract_text_from_image(image_path):
    """Извлечение текста из изображения с помощью Tesseract OCR"""

    try:
        processed_image = preprocess_image(image_path)

        pil_image = Image.fromarray(processed_image)

        # Специальные настройки для старых русских документов
        # PSM 4 - одна колонка текста переменного размера
        # Добавляем whitelist для кириллических символов
        # Адаптивная конфигурация tesseract в зависимости от типа файла
        tiff_type = detect_tiff_type(image_path)

        if tiff_type == "uncompressed_scan":
            # Для несжатых сканов - консервативные настройки
            custom_config = f"--oem 3 --psm 4 -l {config.TESSERACT_LANGUAGES} -c preserve_interword_spaces=1"
        else:
            # Для сжатых документов - стандартные настройки
            custom_config = f"--oem 3 --psm 4 -l {config.TESSERACT_LANGUAGES} -c preserve_interword_spaces=1"

        text = pytesseract.image_to_string(pil_image, config=custom_config)

        return text.strip()

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")
        return ""


def process_tiff_file(file_path, output_dir, file_metadata=None):
    """Обработка одного TIFF файла и сохранение результата в JSON"""

    filename = os.path.basename(file_path)
    file_id = os.path.splitext(filename)[0]

    print(f"Обрабатываю файл: {filename}")

    extracted_text = extract_text_from_image(file_path)

    if not extracted_text:
        print(f"Предупреждение: Текст не извлечен из {filename}")

    result = {"file_id": file_id, "filename": filename, "text": extracted_text, "text_length": len(extracted_text)}

    # Добавляем метаданные об архиве если переданы
    if file_metadata:
        result.update(
            {
                "unique_id": file_metadata.get("unique_id", file_id),
                "original_name": file_metadata.get("original_name", filename),
                "original_path": file_metadata.get("original_path", ""),
                "archive_source": file_metadata.get("archive_source", ""),
                "archive_id": file_metadata.get("archive_id", ""),
            }
        )

    output_path = os.path.join(output_dir, f"{file_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Результат сохранен: {output_path}")
    return result


def process_all_files():
    """Обработка всех TIFF файлов из папки tiff_reports"""

    source_dir = config.DATA_PATHS["tiff_reports"]
    output_dir = config.DATA_PATHS["extracted_text"]

    os.makedirs(output_dir, exist_ok=True)

    tiff_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".tif", ".tiff"))]

    if not tiff_files:
        print("TIFF файлы не найдены в папке tiff_reports!")
        return

    print(f"Найдено {len(tiff_files)} TIFF файлов для обработки")

    results = []
    failed_files = []

    for filename in tqdm(tiff_files, desc="Обработка TIFF файлов"):
        file_path = os.path.join(source_dir, filename)
        try:
            result = process_tiff_file(file_path, output_dir)
            results.append(result)
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {str(e)}")
            failed_files.append(filename)

    summary = {
        "total_files": len(tiff_files),
        "processed_files": len(results),
        "failed_files": len(failed_files),
        "successful_extractions": len([r for r in results if r["text_length"] > 0]),
        "average_text_length": sum(r["text_length"] for r in results) / len(results) if results else 0,
    }

    print(f"\n🏁 ФИНАЛЬНАЯ СВОДКА ОБРАБОТКИ:")
    print(f"📁 Всего файлов: {summary['total_files']}")
    print(f"✅ Обработано: {summary['processed_files']}")
    print(f"❌ Ошибок: {summary['failed_files']}")
    print(f"📝 Успешных извлечений: {summary['successful_extractions']}")
    print(f"📊 Средняя длина текста: {summary['average_text_length']:.0f} символов")

    if failed_files:
        print(f"\n❌ Файлы с ошибками: {', '.join(failed_files[:10])}{'...' if len(failed_files) > 10 else ''}")

    return results


def process_test_files():
    """Обработка тестовых файлов"""

    test_dir = config.DATA_PATHS["test_files"]
    output_dir = config.DATA_PATHS["extracted_text"]

    os.makedirs(output_dir, exist_ok=True)

    tiff_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".tif", ".tiff"))]

    if not tiff_files:
        print("Тестовые TIFF файлы не найдены!")
        return

    print(f"Найдено {len(tiff_files)} тестовых файлов")

    results = []
    for filename in tqdm(tiff_files, desc="Обработка тестовых файлов"):
        file_path = os.path.join(test_dir, filename)
        result = process_tiff_file(file_path, output_dir)
        results.append(result)

    summary = {
        "total_files": len(results),
        "successful_extractions": len([r for r in results if r["text_length"] > 0]),
        "average_text_length": sum(r["text_length"] for r in results) / len(results) if results else 0,
    }

    print(f"\nСводка обработки:")
    print(f"Всего файлов: {summary['total_files']}")
    print(f"Успешных извлечений: {summary['successful_extractions']}")
    print(f"Средняя длина текста: {summary['average_text_length']:.0f} символов")

    return results


def process_extracted_files(batch_size=10, resume_from=None):
    """Обработка файлов извлеченных из архивов с поддержкой checkpoint'ов"""

    from archive_processor import ArchiveProcessor

    processor = ArchiveProcessor()
    metadata = processor.load_metadata()

    if not metadata or not metadata.get("archives"):
        print("Метаданные архивов не найдены. Сначала запустите обработку архивов.")
        return []

    all_files = processor.get_all_extracted_files()
    if not all_files:
        print("Извлеченные файлы не найдены.")
        return []

    output_dir = config.DATA_PATHS["extracted_text"]
    os.makedirs(output_dir, exist_ok=True)

    # Поддержка resume - пропускаем уже обработанные файлы
    processed_files = set()
    if resume_from:
        existing_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
        processed_files = {os.path.splitext(f)[0] for f in existing_files}
        print(f"Найдено {len(processed_files)} уже обработанных файлов")

    total_files = len(all_files)
    results = []
    failed_files = []

    print(f"Начинаю обработку {total_files} файлов из архивов...")
    print(f"Размер batch: {batch_size}")

    # Обработка батчами для управления памятью
    for i in range(0, total_files, batch_size):
        batch = all_files[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size

        print(f"\n=== Batch {batch_num}/{total_batches} ({len(batch)} файлов) ===")

        for file_info in tqdm(batch, desc=f"Batch {batch_num}"):
            unique_id = file_info["unique_id"]

            # Пропускаем уже обработанные
            if unique_id in processed_files:
                print(f"Пропускаю {unique_id} (уже обработан)")
                continue

            file_path = file_info["extracted_path"]

            if not os.path.exists(file_path):
                print(f"Файл не найден: {file_path}")
                failed_files.append(unique_id)
                continue

            try:
                # Передаем метаданные архива
                archive_metadata = {
                    "unique_id": unique_id,
                    "original_name": file_info["original_name"],
                    "original_path": file_info["original_path"],
                    "archive_source": file_info["archive_source"],
                    "archive_id": file_info.get("archive_id", ""),
                }

                result = process_tiff_file(file_path, output_dir, archive_metadata)
                results.append(result)
                processed_files.add(unique_id)

            except Exception as e:
                print(f"Ошибка при обработке {unique_id}: {str(e)}")
                failed_files.append(unique_id)

        # Checkpoint после каждого batch
        print(f"Batch {batch_num} завершен. Обработано: {len(results)}, ошибок: {len(failed_files)}")

    # Финальная сводка
    summary = {
        "total_files": total_files,
        "processed_files": len(results),
        "failed_files": len(failed_files),
        "successful_extractions": len([r for r in results if r["text_length"] > 0]),
        "average_text_length": sum(r["text_length"] for r in results) / len(results) if results else 0,
    }

    print(f"\n🏁 ФИНАЛЬНАЯ СВОДКА ОБРАБОТКИ АРХИВОВ:")
    print(f"📁 Всего файлов: {summary['total_files']}")
    print(f"✅ Обработано: {summary['processed_files']}")
    print(f"❌ Ошибок: {summary['failed_files']}")
    print(f"📝 Успешных извлечений: {summary['successful_extractions']}")
    print(f"📊 Средняя длина текста: {summary['average_text_length']:.0f} символов")

    if failed_files:
        print(f"\n❌ Файлы с ошибками: {', '.join(failed_files[:10])}{'...' if len(failed_files) > 10 else ''}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("🚀 ЗАПУСК ОБРАБОТКИ ВСЕХ ФАЙЛОВ!")
        results = process_all_files()
    elif len(sys.argv) > 1 and sys.argv[1] == "--archives":
        print("🗂️ ЗАПУСК ОБРАБОТКИ ФАЙЛОВ ИЗ АРХИВОВ!")
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        resume = True if len(sys.argv) > 3 and sys.argv[3] == "--resume" else False
        results = process_extracted_files(batch_size=batch_size, resume_from=resume)
    else:
        print("🧪 Обработка тестовых файлов")
        print("Для всех файлов: --all")
        print("Для архивов: --archives [batch_size] [--resume]")
        results = process_test_files()
