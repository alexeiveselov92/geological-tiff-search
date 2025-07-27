import os
import json
import pickle
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import config

class SimpleEmbeddingsCreator:
    """Простой создатель эмбеддингов на основе TF-IDF для тестирования"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Для русского языка
            ngram_range=(1, 2),
            min_df=1
        )
        self.is_fitted = False
    
    def create_embeddings_for_chunks(self, chunks_data: list) -> list:
        """Создание TF-IDF эмбеддингов для списка чанков"""
        texts = [chunk["text"] for chunk in chunks_data]
        
        print(f"Создаю TF-IDF эмбеддинги для {len(texts)} чанков...")
        
        if not self.is_fitted:
            embeddings_matrix = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            embeddings_matrix = self.vectorizer.transform(texts)
        
        embeddings = embeddings_matrix.toarray()
        
        result = []
        for chunk, embedding in zip(chunks_data, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding.tolist()
            chunk_with_embedding["embedding_dim"] = len(embedding)
            result.append(chunk_with_embedding)
        
        return result
    
    def process_chunks_file(self, file_path: str, output_dir: str):
        """Обработка одного файла с чанками"""
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        file_id = os.path.splitext(os.path.basename(file_path))[0].replace('_chunks', '')
        
        print(f"Обрабатываю файл: {file_id} ({len(chunks_data)} чанков)")
        
        chunks_with_embeddings = self.create_embeddings_for_chunks(chunks_data)
        
        output_path = os.path.join(output_dir, f"{file_id}_embeddings.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
        
        print(f"Эмбеддинги сохранены: {output_path}")
        
        return len(chunks_data)
    
    def process_all_chunks(self):
        """Обработка всех файлов с чанками"""
        input_dir = config.DATA_PATHS["processed_chunks"]
        output_dir = config.DATA_PATHS["embeddings"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        chunks_files = [f for f in os.listdir(input_dir) if f.endswith('_chunks.json')]
        
        if not chunks_files:
            print("Файлы с чанками не найдены!")
            return
        
        print(f"Найдено {len(chunks_files)} файлов с чанками")
        
        # Сначала собираем все тексты для обучения векторайзера
        all_texts = []
        all_chunks_data = []
        
        for filename in chunks_files:
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            all_chunks_data.extend(chunks_data)
            all_texts.extend([chunk["text"] for chunk in chunks_data])
        
        # Обучаем векторайзер на всех текстах
        print("Обучаю TF-IDF векторайзер на всех текстах...")
        embeddings_matrix = self.vectorizer.fit_transform(all_texts)
        self.is_fitted = True
        
        # Создаем эмбеддинги для каждого файла
        total_chunks = 0
        start_idx = 0
        
        for filename in tqdm(chunks_files, desc="Создание эмбеддингов"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            file_id = os.path.splitext(filename)[0].replace('_chunks', '')
            end_idx = start_idx + len(chunks_data)
            
            # Извлекаем соответствующие эмбеддинги
            file_embeddings = embeddings_matrix[start_idx:end_idx].toarray()
            
            # Добавляем эмбеддинги к чанкам
            chunks_with_embeddings = []
            for chunk, embedding in zip(chunks_data, file_embeddings):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = embedding.tolist()
                chunk_with_embedding["embedding_dim"] = len(embedding)
                chunks_with_embeddings.append(chunk_with_embedding)
            
            # Сохраняем файл
            output_path = os.path.join(output_dir, f"{file_id}_embeddings.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
            
            total_chunks += len(chunks_data)
            start_idx = end_idx
        
        print(f"\nВсего обработано {total_chunks} чанков")
        
        # Создаем поисковый индекс
        self.create_search_index(output_dir, all_chunks_data, embeddings_matrix.toarray())
        
        return total_chunks
    
    def create_search_index(self, embeddings_dir: str, all_chunks: list, all_embeddings: np.ndarray):
        """Создание поискового индекса из всех эмбеддингов"""
        print("Создаю поисковый индекс...")
        
        # Подготавливаем данные для индекса
        index_chunks = []
        for chunk in all_chunks:
            index_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "file_id": chunk["file_id"],
                "filename": chunk["filename"],
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"]
            })
        
        index_data = {
            "chunks": index_chunks,
            "embeddings": all_embeddings,
            "vectorizer": self.vectorizer,
            "model_name": "TF-IDF",
            "total_chunks": len(index_chunks),
            "embedding_dim": all_embeddings.shape[1] if len(all_embeddings) > 0 else 0
        }
        
        index_path = os.path.join(embeddings_dir, "search_index.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Поисковый индекс создан: {index_path}")
        print(f"Всего чанков в индексе: {len(index_chunks)}")
        print(f"Размерность эмбеддингов: {index_data['embedding_dim']}")
        
        return index_path

def create_simple_embeddings():
    """Создание простых TF-IDF эмбеддингов для тестовых данных"""
    creator = SimpleEmbeddingsCreator()
    return creator.process_all_chunks()

if __name__ == "__main__":
    create_simple_embeddings()