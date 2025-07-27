#!/usr/bin/env python3
"""
Главный файл для полной обработки архивов с TIFF файлами
Поддерживает весь pipeline: извлечение → OCR → чанкинг → embeddings → индексация
"""

import os
import sys
import time
import logging
from pathlib import Path

# Добавляем src в путь для импорта модулей
sys.path.append("src")

from archive_processor import ArchiveProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("archive_processing.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ArchivePipeline:
    def __init__(self):
        self.archive_processor = ArchiveProcessor()
        self.total_start_time = time.time()

    def step1_extract_archives(self):
        """Шаг 1: Извлечение файлов из архивов"""
        logger.info("🗂️ === ШАГ 1: ИЗВЛЕЧЕНИЕ АРХИВОВ ===")

        start_time = time.time()
        metadata = self.archive_processor.process_all_archives()

        if metadata["total_files"] == 0:
            logger.error("Не найдено файлов для извлечения!")
            return False

        extraction_time = time.time() - start_time

        logger.info(f"✅ Извлечение завершено за {extraction_time:.2f} сек")
        logger.info(f"📁 Архивов: {metadata['total_archives']}")
        logger.info(f"📄 Файлов: {metadata['total_files']}")

        # Оценка времени для больших объемов
        if metadata["total_files"] > 0:
            avg_time_per_file = extraction_time / metadata["total_files"]
            estimated_1000_files = (avg_time_per_file * 1000) / 60
            logger.info(f"⏱️ Оценка времени на 1000 файлов: {estimated_1000_files:.1f} минут")

        return True

    def step2_ocr_processing(self, batch_size=5):
        """Шаг 2: OCR обработка извлеченных файлов"""
        logger.info("📖 === ШАГ 2: OCR ОБРАБОТКА ===")

        start_time = time.time()

        # Ленивый импорт
        from ocr_processor import process_extracted_files

        # Проверяем есть ли уже обработанные файлы для resume
        extracted_text_dir = Path("data/extracted_text")
        existing_files = []
        if extracted_text_dir.exists():
            existing_files = list(extracted_text_dir.glob("*.json"))

        logger.info(f"📋 Найдено {len(existing_files)} уже обработанных файлов")

        results = process_extracted_files(batch_size=batch_size, resume_from=True)

        ocr_time = time.time() - start_time

        # Проверяем общее количество файлов для обработки
        total_text_files = len(list(extracted_text_dir.glob("*.json"))) if extracted_text_dir.exists() else 0

        if total_text_files == 0:
            logger.error("OCR обработка не дала результатов - нет извлеченных текстов!")
            return False

        successful = len([r for r in results if r["text_length"] > 0])
        avg_text_length = sum(r["text_length"] for r in results) / len(results) if results else 0

        logger.info(f"✅ OCR завершен за {ocr_time:.2f} сек")
        if results:
            logger.info(f"📝 Успешных извлечений: {successful}/{len(results)}")
            logger.info(f"📊 Средняя длина текста: {avg_text_length:.0f} символов")
        else:
            logger.info(f"📋 Все файлы уже обработаны (resume). Всего текстов: {total_text_files}")
            logger.info(f"📊 Продолжаем с существующими данными")

        return True

    def step3_text_processing(self):
        """Шаг 3: Обработка текста и создание чанков"""
        logger.info("✂️ === ШАГ 3: ОБРАБОТКА ТЕКСТА И ЧАНКИНГ ===")

        start_time = time.time()

        try:
            # Ленивый импорт
            from text_processor import process_all_extracted_texts

            results = process_all_extracted_texts()

            if not results:
                logger.error("Обработка текста не дала результатов!")
                return False

            processing_time = time.time() - start_time

            total_chunks = sum(len(r.get("chunks", [])) for r in results)
            avg_chunks_per_file = total_chunks / len(results) if results else 0

            logger.info(f"✅ Обработка текста завершена за {processing_time:.2f} сек")
            logger.info(f"📄 Обработано файлов: {len(results)}")
            logger.info(f"🧩 Всего чанков: {total_chunks}")
            logger.info(f"📊 Среднее количество чанков на файл: {avg_chunks_per_file:.1f}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при обработке текста: {e}")
            return False

    def step4_create_embeddings(self):
        """Шаг 4: Создание векторных представлений"""
        logger.info("🧠 === ШАГ 4: СОЗДАНИЕ EMBEDDINGS ===")

        start_time = time.time()

        try:
            # Ленивый импорт
            from embeddings_creator import create_embeddings_for_test_data

            total_chunks = create_embeddings_for_test_data()

            if total_chunks == 0:
                logger.error("Создание embeddings не дало результатов!")
                return False

            embeddings_time = time.time() - start_time

            logger.info(f"✅ Создание embeddings завершено за {embeddings_time:.2f} сек")
            logger.info(f"🧠 Всего обработано чанков: {total_chunks}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при создании embeddings: {e}")
            return False

    def step5_build_search_index(self):
        """Шаг 5: Построение поискового индекса"""
        logger.info("🔍 === ШАГ 5: ПОСТРОЕНИЕ ПОИСКОВОГО ИНДЕКСА ===")

        start_time = time.time()

        try:
            # Ленивый импорт
            from search_engine import SearchEngine

            search_engine = SearchEngine()
            search_engine.build_index()

            index_time = time.time() - start_time

            logger.info(f"✅ Индекс построен за {index_time:.2f} сек")

            # Тестовый поиск для проверки
            test_results = search_engine.search("геология", top_k=3)
            if test_results:
                logger.info(f"🔍 Тестовый поиск успешен: найдено {len(test_results)} результатов")
            else:
                logger.warning("⚠️ Тестовый поиск не дал результатов")

            return True

        except Exception as e:
            logger.error(f"Ошибка при построении индекса: {e}")
            return False

    def run_full_pipeline(self, batch_size=5, skip_steps=None):
        """Запуск полного pipeline обработки"""
        skip_steps = skip_steps or []

        logger.info("🚀 === ЗАПУСК ПОЛНОГО PIPELINE ОБРАБОТКИ АРХИВОВ ===")
        logger.info(f"⚙️ Размер batch для OCR: {batch_size}")

        steps = [
            (1, "extract_archives", self.step1_extract_archives),
            (2, "ocr_processing", lambda: self.step2_ocr_processing(batch_size)),
            (3, "text_processing", self.step3_text_processing),
            (4, "create_embeddings", self.step4_create_embeddings),
            (5, "build_search_index", self.step5_build_search_index),
        ]

        for step_num, step_name, step_func in steps:
            if step_name in skip_steps:
                logger.info(f"⏭️ Пропускаю шаг {step_num}: {step_name}")
                continue

            logger.info(f"\n{'='*60}")

            if not step_func():
                logger.error(f"❌ Шаг {step_num} завершился с ошибкой!")
                return False

            # Показываем прогресс
            elapsed = time.time() - self.total_start_time
            logger.info(f"⏱️ Общее время выполнения: {elapsed:.2f} сек")

        total_time = time.time() - self.total_start_time

        logger.info(f"\n{'='*60}")
        logger.info("🎉 === PIPELINE ЗАВЕРШЕН УСПЕШНО! ===")
        logger.info(f"⏱️ Общее время выполнения: {total_time:.2f} сек ({total_time/60:.1f} мин)")
        logger.info("✅ Система готова к использованию!")
        logger.info("📞 Запустите ask_geo.py для работы с системой")

        return True


def main():
    """Главная функция с обработкой аргументов командной строки"""

    pipeline = ArchivePipeline()

    if len(sys.argv) == 1:
        # Без аргументов - запуск полного pipeline
        logger.info("Запуск полного pipeline (используйте --help для справки)")
        success = pipeline.run_full_pipeline()

    elif "--help" in sys.argv or "-h" in sys.argv:
        print("🗂️ ОБРАБОТКА АРХИВОВ - Справка по использованию")
        print()
        print("Команды:")
        print("  python process_archives.py                    - Полный pipeline")
        print("  python process_archives.py --batch 10         - С указанием размера batch")
        print("  python process_archives.py --skip ocr         - Пропустить OCR")
        print("  python process_archives.py --extract-only     - Только извлечение")
        print("  python process_archives.py --ocr-only         - Только OCR")
        print("  python process_archives.py --index-only       - Только индексация")
        print()
        print("Параметры:")
        print("  --batch N      - Размер batch для OCR (по умолчанию 5)")
        print("  --skip STEP    - Пропустить шаг (extract_archives, ocr_processing, etc.)")
        print()
        return

    elif "--extract-only" in sys.argv:
        logger.info("Запуск только извлечения архивов")
        success = pipeline.step1_extract_archives()

    elif "--ocr-only" in sys.argv:
        batch_size = 5
        if "--batch" in sys.argv:
            try:
                batch_idx = sys.argv.index("--batch")
                batch_size = int(sys.argv[batch_idx + 1])
            except (IndexError, ValueError):
                logger.warning("Неверный параметр --batch, используется значение по умолчанию: 5")

        logger.info("Запуск только OCR обработки")
        success = pipeline.step2_ocr_processing(batch_size)

    elif "--index-only" in sys.argv:
        logger.info("Запуск только построения индекса")
        success = pipeline.step5_build_search_index()

    else:
        # Полный pipeline с параметрами
        batch_size = 5
        skip_steps = []

        if "--batch" in sys.argv:
            try:
                batch_idx = sys.argv.index("--batch")
                batch_size = int(sys.argv[batch_idx + 1])
            except (IndexError, ValueError):
                logger.warning("Неверный параметр --batch, используется значение по умолчанию: 5")

        if "--skip" in sys.argv:
            try:
                skip_idx = sys.argv.index("--skip")
                skip_steps = [sys.argv[skip_idx + 1]]
            except IndexError:
                logger.warning("Неверный параметр --skip")

        success = pipeline.run_full_pipeline(batch_size=batch_size, skip_steps=skip_steps)

    if success:
        logger.info("🎉 Обработка завершена успешно!")
        sys.exit(0)
    else:
        logger.error("❌ Обработка завершилась с ошибками!")
        sys.exit(1)


if __name__ == "__main__":
    main()
