import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArchiveProcessor:
    def __init__(self, archives_dir: str = "tiff_reports", extracted_dir: str = "data/extracted_files"):
        self.archives_dir = Path(archives_dir)
        self.extracted_dir = Path(extracted_dir)
        self.metadata_file = self.extracted_dir / "metadata.json"
        
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        
    def scan_archives(self) -> List[Path]:
        """Сканирование архивов в папке tiff_reports/"""
        if not self.archives_dir.exists():
            logger.info(f"Создаю папку {self.archives_dir}")
            self.archives_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Папка {self.archives_dir} была пуста. Поместите ZIP архивы в эту папку и запустите снова.")
            return []
            
        zip_files = list(self.archives_dir.glob("*.zip"))
        logger.info(f"Найдено {len(zip_files)} ZIP архивов")
        return zip_files
    
    def _get_archive_id(self, archive_path: Path) -> str:
        """Генерация уникального ID архива"""
        return archive_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    
    def _generate_unique_file_id(self, archive_id: str, original_name: str, counter: int) -> str:
        """Генерация уникального ID для файла"""
        base_name = Path(original_name).stem
        return f"{archive_id}_{counter:04d}_{base_name}"
    
    def extract_tiff_files(self, archive_path: Path) -> Dict:
        """Рекурсивное извлечение TIFF файлов из архива"""
        archive_id = self._get_archive_id(archive_path)
        archive_dir = self.extracted_dir / archive_id
        archive_dir.mkdir(exist_ok=True)
        
        extracted_files = []
        total_files = 0
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                tiff_files = [f for f in file_list if f.lower().endswith(('.tiff', '.tif'))]
                
                logger.info(f"Архив {archive_path.name}: найдено {len(tiff_files)} TIFF файлов из {len(file_list)} общих")
                
                for counter, file_path in enumerate(tiff_files, 1):
                    try:
                        original_name = Path(file_path).name
                        unique_id = self._generate_unique_file_id(archive_id, original_name, counter)
                        
                        file_info = zip_ref.getinfo(file_path)
                        file_data = zip_ref.read(file_path)
                        
                        extracted_path = archive_dir / f"{unique_id}.tiff"
                        
                        with open(extracted_path, 'wb') as out_file:
                            out_file.write(file_data)
                        
                        extracted_files.append({
                            "unique_id": unique_id,
                            "original_path": file_path,
                            "original_name": original_name,
                            "extracted_path": str(extracted_path),
                            "archive_source": str(archive_path),
                            "file_size": file_info.file_size,
                            "date_time": list(file_info.date_time)
                        })
                        
                        total_files += 1
                        
                    except Exception as e:
                        logger.error(f"Ошибка при извлечении {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Ошибка при открытии архива {archive_path}: {e}")
            return {"archive_id": archive_id, "files": [], "error": str(e)}
        
        logger.info(f"Успешно извлечено {total_files} файлов из архива {archive_path.name}")
        
        return {
            "archive_id": archive_id,
            "archive_path": str(archive_path),
            "files": extracted_files,
            "total_files": total_files,
            "extraction_time": time.time()
        }
    
    def process_all_archives(self) -> Dict:
        """Batch обработка всех архивов"""
        archives = self.scan_archives()
        if not archives:
            logger.warning("Архивы не найдены")
            return {"archives": [], "total_files": 0}
        
        all_metadata = {
            "archives": [],
            "total_archives": len(archives),
            "total_files": 0,
            "processing_time": time.time()
        }
        
        logger.info(f"Начинаю обработку {len(archives)} архивов...")
        
        for archive in tqdm(archives, desc="Обработка архивов"):
            start_time = time.time()
            logger.info(f"Обрабатываю архив: {archive.name}")
            
            archive_metadata = self.extract_tiff_files(archive)
            archive_metadata["processing_time"] = time.time() - start_time
            
            all_metadata["archives"].append(archive_metadata)
            all_metadata["total_files"] += archive_metadata.get("total_files", 0)
            
            logger.info(f"Архив {archive.name} обработан за {archive_metadata['processing_time']:.2f} сек")
        
        all_metadata["total_processing_time"] = time.time() - all_metadata["processing_time"]
        
        self._save_metadata(all_metadata)
        
        logger.info(f"Обработка завершена. Всего файлов: {all_metadata['total_files']}")
        logger.info(f"Общее время: {all_metadata['total_processing_time']:.2f} секунд")
        
        return all_metadata
    
    def _save_metadata(self, metadata: Dict):
        """Сохранение метаданных в JSON"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Метаданные сохранены в {self.metadata_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении метаданных: {e}")
    
    def load_metadata(self) -> Dict:
        """Загрузка метаданных"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка при загрузке метаданных: {e}")
        return {}
    
    def get_all_extracted_files(self) -> List[Dict]:
        """Получение списка всех извлеченных файлов"""
        metadata = self.load_metadata()
        all_files = []
        
        for archive in metadata.get("archives", []):
            for file_info in archive.get("files", []):
                all_files.append(file_info)
        
        return all_files
    
    def cleanup_extracted_files(self):
        """Очистка извлеченных файлов"""
        if self.extracted_dir.exists():
            shutil.rmtree(self.extracted_dir)
            logger.info(f"Папка {self.extracted_dir} очищена")
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    processor = ArchiveProcessor()
    
    logger.info("=== ТЕСТ ОБРАБОТКИ АРХИВОВ ===")
    
    archives = processor.scan_archives()
    if archives:
        logger.info(f"Найдены архивы: {[a.name for a in archives]}")
        
        start_time = time.time()
        metadata = processor.process_all_archives()
        
        logger.info("=== РЕЗУЛЬТАТЫ ===")
        logger.info(f"Обработано архивов: {metadata['total_archives']}")
        logger.info(f"Всего извлечено файлов: {metadata['total_files']}")
        logger.info(f"Общее время: {metadata['total_processing_time']:.2f} сек")
        
        if metadata['total_files'] > 0:
            avg_time = metadata['total_processing_time'] / metadata['total_files']
            logger.info(f"Среднее время на файл: {avg_time:.3f} сек")
            
            estimated_time_1000 = (avg_time * 1000) / 60
            logger.info(f"Оценка времени на 1000 файлов: {estimated_time_1000:.1f} минут")
    else:
        logger.warning("Архивы не найдены в папке tiff_reports/")