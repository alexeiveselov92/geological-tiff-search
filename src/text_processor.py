import os
import json
import re
from typing import List, Dict
from tqdm import tqdm
import config

def clean_ocr_text(text: str) -> str:
    """Очистка текста от OCR артефактов"""
    
    text = re.sub(r'\|+', ' ', text)
    
    text = re.sub(r'[_\-]{3,}', ' ', text)
    
    text = re.sub(r'[^\w\sА-я\.,;:!?\(\)\[\]№%\-/]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'^\s*[\.\-\|]+\s*', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'\n\s*\n', '\n', text)
    
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 2:
            clean_lines.append(line)
    
    return '\n'.join(clean_lines)

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Разбивка текста на чанки с перекрытием"""
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            last_newline = text.rfind('\n', start, end)
            last_period = text.rfind('.', start, end)
            
            best_break = max(last_space, last_newline, last_period)
            if best_break > start:
                end = best_break + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = max(start + chunk_size - overlap, end - overlap)
    
    return chunks

def process_extracted_text_file(file_path: str, output_dir: str) -> Dict:
    """Обработка одного JSON файла с извлеченным текстом"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    file_id = data['file_id']
    raw_text = data['text']
    
    print(f"Обрабатываю файл: {file_id}")
    
    cleaned_text = clean_ocr_text(raw_text)
    
    chunks = split_text_into_chunks(cleaned_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": f"{file_id}_chunk_{i:03d}",
            "file_id": file_id,
            "filename": data['filename'],
            "chunk_index": i,
            "text": chunk,
            "text_length": len(chunk),
            "metadata": {
                "total_chunks": len(chunks),
                "original_text_length": len(raw_text),
                "cleaned_text_length": len(cleaned_text)
            }
        }
        processed_chunks.append(chunk_data)
    
    output_path = os.path.join(output_dir, f"{file_id}_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Создано {len(chunks)} чанков, сохранено: {output_path}")
    
    return {
        "file_id": file_id,
        "chunks_count": len(chunks),
        "total_length": sum(len(chunk) for chunk in chunks),
        "average_chunk_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
    }

def process_all_extracted_texts():
    """Обработка всех извлеченных текстов"""
    
    input_dir = config.DATA_PATHS["extracted_text"]
    output_dir = config.DATA_PATHS["processed_chunks"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print("JSON файлы с извлеченным текстом не найдены!")
        return
    
    print(f"Найдено {len(json_files)} файлов для обработки")
    
    results = []
    total_chunks = 0
    
    for filename in tqdm(json_files, desc="Обработка текстов"):
        file_path = os.path.join(input_dir, filename)
        result = process_extracted_text_file(file_path, output_dir)
        results.append(result)
        total_chunks += result["chunks_count"]
    
    summary = {
        "total_files": len(results),
        "total_chunks": total_chunks,
        "average_chunks_per_file": total_chunks / len(results) if results else 0,
        "average_chunk_length": sum(r["average_chunk_length"] for r in results) / len(results) if results else 0
    }
    
    print(f"\nСводка обработки:")
    print(f"Всего файлов: {summary['total_files']}")
    print(f"Всего чанков: {summary['total_chunks']}")
    print(f"Среднее количество чанков на файл: {summary['average_chunks_per_file']:.1f}")
    print(f"Средняя длина чанка: {summary['average_chunk_length']:.0f} символов")
    
    return results

if __name__ == "__main__":
    results = process_all_extracted_texts()