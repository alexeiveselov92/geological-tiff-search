import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import config

class GeologicalSearchEngine:
    """Поисковая система для геологических документов"""
    
    def __init__(self, index_path: str = None):
        if index_path is None:
            index_path = os.path.join(config.DATA_PATHS["embeddings"], "search_index.pkl")
        
        self.index_path = index_path
        self.index_data = None
        self.model = None
        
        # Загружаем индекс только если он уже существует
        if os.path.exists(self.index_path):
            self.load_index()
    
    def build_index(self):
        """Построение поискового индекса из embeddings"""
        from embeddings_creator import EmbeddingsCreator
        
        print("🔍 Строю поисковый индекс...")
        
        # Проверяем есть ли уже созданные embeddings
        embeddings_dir = config.DATA_PATHS["embeddings"]
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(f"Директория с embeddings не найдена: {embeddings_dir}")
        
        embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embeddings.json')]
        if not embedding_files:
            raise FileNotFoundError("Не найдены файлы с embeddings. Сначала запустите создание embeddings.")
        
        print(f"Найдено {len(embedding_files)} файлов с embeddings")
        
        # Создаем индекс
        creator = EmbeddingsCreator()
        index_path = creator.create_search_index(embeddings_dir)
        
        # Загружаем созданный индекс
        self.load_index()
        
        return index_path
    
    def load_index(self):
        """Загрузка поискового индекса"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Поисковый индекс не найден: {self.index_path}")
        
        print(f"Загружаю поисковый индекс: {self.index_path}")
        
        with open(self.index_path, 'rb') as f:
            self.index_data = pickle.load(f)
        
        # Загружаем модель для векторизации запросов
        model_name = self.index_data["model_name"]
        print(f"Загружаю модель: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print(f"Индекс загружен:")
        print(f"- Модель: {self.index_data['model_name']}")
        print(f"- Всего чанков: {self.index_data['total_chunks']}")
        print(f"- Размерность эмбеддингов: {self.index_data['embedding_dim']}")
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """Векторизация поискового запроса"""
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        query_vector = self.model.encode([query])[0]
        return query_vector
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """
        Поиск релевантных чанков по запросу
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов для возврата
            min_similarity: Минимальный порог сходства
        
        Returns:
            Список релевантных чанков с оценками сходства
        """
        if self.index_data is None:
            raise ValueError("Индекс не загружен")
        
        # Векторизация запроса
        query_vector = self.vectorize_query(query)
        
        # Вычисление косинусного сходства
        similarities = cosine_similarity(
            query_vector.reshape(1, -1), 
            self.index_data["embeddings"]
        )[0]
        
        # Сортировка по убыванию сходства
        ranked_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i, idx in enumerate(ranked_indices):
            if i >= top_k:
                break
            
            similarity = similarities[idx]
            if similarity < min_similarity:
                break
            
            chunk = self.index_data["chunks"][idx].copy()
            chunk["similarity"] = float(similarity)
            chunk["rank"] = i + 1
            
            results.append(chunk)
        
        return results
    
    def search_with_details(self, query: str, top_k: int = 5) -> Dict:
        """
        Расширенный поиск с дополнительной информацией
        
        Returns:
            Словарь с результатами поиска и метаинформацией
        """
        results = self.search(query, top_k)
        
        # Группировка результатов по файлам
        files_found = {}
        for result in results:
            file_id = result["file_id"]
            if file_id not in files_found:
                files_found[file_id] = []
            files_found[file_id].append(result)
        
        # Статистика
        avg_similarity = np.mean([r["similarity"] for r in results]) if results else 0
        max_similarity = max([r["similarity"] for r in results]) if results else 0
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "files_found": files_found,
            "stats": {
                "average_similarity": float(avg_similarity),
                "max_similarity": float(max_similarity),
                "files_count": len(files_found)
            }
        }
    
    def get_chunk_context(self, chunk_id: str, context_size: int = 1) -> List[Dict]:
        """
        Получение контекста вокруг найденного чанка
        
        Args:
            chunk_id: ID чанка
            context_size: Количество соседних чанков с каждой стороны
        
        Returns:
            Список чанков включая контекст
        """
        # Находим целевой чанк
        target_chunk = None
        target_idx = None
        
        for i, chunk in enumerate(self.index_data["chunks"]):
            if chunk["chunk_id"] == chunk_id:
                target_chunk = chunk
                target_idx = i
                break
        
        if target_chunk is None:
            return []
        
        file_id = target_chunk["file_id"]
        chunk_index = target_chunk["chunk_index"]
        
        # Находим все чанки из того же файла
        file_chunks = []
        for chunk in self.index_data["chunks"]:
            if chunk["file_id"] == file_id:
                file_chunks.append(chunk)
        
        # Сортируем по индексу чанка
        file_chunks.sort(key=lambda x: x["chunk_index"])
        
        # Находим позицию целевого чанка в файле
        target_pos = None
        for i, chunk in enumerate(file_chunks):
            if chunk["chunk_id"] == chunk_id:
                target_pos = i
                break
        
        if target_pos is None:
            return [target_chunk]
        
        # Определяем границы контекста
        start = max(0, target_pos - context_size)
        end = min(len(file_chunks), target_pos + context_size + 1)
        
        context_chunks = file_chunks[start:end]
        
        # Отмечаем целевой чанк
        for chunk in context_chunks:
            chunk["is_target"] = (chunk["chunk_id"] == chunk_id)
        
        return context_chunks

def test_search_engine():
    """Тестирование поисковой системы"""
    try:
        search_engine = GeologicalSearchEngine()
        
        # Тестовые запросы на русском языке
        test_queries = [
            "месторождение",
            "песок гравий",
            "Борисово",
            "геологическая разведка",
            "Протва",
            "1959"
        ]
        
        print("\n=== ТЕСТИРОВАНИЕ ПОИСКОВОЙ СИСТЕМЫ ===\n")
        
        for query in test_queries:
            print(f"\nЗапрос: '{query}'")
            print("-" * 50)
            
            search_results = search_engine.search_with_details(query, top_k=3)
            
            print(f"Найдено результатов: {search_results['total_results']}")
            print(f"Файлов затронуто: {search_results['stats']['files_count']}")
            print(f"Максимальное сходство: {search_results['stats']['max_similarity']:.3f}")
            
            for i, result in enumerate(search_results['results'], 1):
                print(f"\n{i}. Файл: {result['filename']}")
                print(f"   Чанк: {result['chunk_id']}")
                print(f"   Сходство: {result['similarity']:.3f}")
                print(f"   Текст: {result['text'][:200]}...")
        
        return search_engine
    
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        return None

# Алиас для обратной совместимости
SearchEngine = GeologicalSearchEngine

if __name__ == "__main__":
    test_search_engine()