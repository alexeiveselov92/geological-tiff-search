import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

class EmbeddingsCreator:
    def __init__(self, model_name: str = None):
        """Инициализация создателя эмбеддингов"""
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Загружаю модель: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Модель загружена успешно!")
    
    def create_embeddings_for_chunks(self, chunks_data: list) -> list:
        """Создание эмбеддингов для списка чанков"""
        texts = [chunk["text"] for chunk in chunks_data]
        
        print(f"Создаю эмбеддинги для {len(texts)} чанков...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
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
        
        total_chunks = 0
        for filename in tqdm(chunks_files, desc="Создание эмбеддингов"):
            file_path = os.path.join(input_dir, filename)
            chunks_count = self.process_chunks_file(file_path, output_dir)
            total_chunks += chunks_count
        
        print(f"\nВсего обработано {total_chunks} чанков")
        
        self.create_search_index(output_dir)
        
        return total_chunks
    
    def create_search_index(self, embeddings_dir: str):
        """Создание поискового индекса из всех эмбеддингов"""
        print("Создаю поисковый индекс...")
        
        all_chunks = []
        all_embeddings = []
        
        embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embeddings.json')]
        
        for filename in embedding_files:
            file_path = os.path.join(embeddings_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            for chunk in chunks:
                all_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "file_id": chunk["file_id"],
                    "filename": chunk["filename"],
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"]
                })
                all_embeddings.append(chunk["embedding"])
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        index_data = {
            "chunks": all_chunks,
            "embeddings": embeddings_array,
            "model_name": self.model_name,
            "total_chunks": len(all_chunks),
            "embedding_dim": embeddings_array.shape[1] if len(all_embeddings) > 0 else 0
        }
        
        index_path = os.path.join(embeddings_dir, "search_index.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Поисковый индекс создан: {index_path}")
        print(f"Всего чанков в индексе: {len(all_chunks)}")
        print(f"Размерность эмбеддингов: {index_data['embedding_dim']}")
        
        return index_path

def create_embeddings_for_test_data():
    """Создание эмбеддингов для тестовых данных"""
    creator = EmbeddingsCreator()
    return creator.process_all_chunks()

if __name__ == "__main__":
    create_embeddings_for_test_data()