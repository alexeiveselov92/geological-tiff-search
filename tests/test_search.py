#!/usr/bin/env python3
"""
Тестовый скрипт для проверки поисковой системы
"""

import sys
import os

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search_engine import GeologicalSearchEngine

def test_search_simple():
    """Простой тест поиска"""
    print("=== ПРОСТОЙ ТЕСТ ПОИСКА ===\n")
    
    try:
        # Инициализация поисковой системы
        search_engine = GeologicalSearchEngine()
        
        # Тестовые запросы
        queries = [
            "месторождение",
            "песок и гравий", 
            "Борисово",
            "Протва",
            "геологическая разведка"
        ]
        
        for query in queries:
            print(f"Запрос: '{query}'")
            print("-" * 40)
            
            results = search_engine.search(query, top_k=3)
            
            if not results:
                print("Результатов не найдено")
            else:
                print(f"Найдено {len(results)} результатов:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Файл: {result['filename']}")
                    print(f"   Сходство: {result['similarity']:.3f}")
                    print(f"   Текст: {result['text'][:150]}...")
            
            print("\n" + "="*60 + "\n")
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False
    
    return True

def test_search_detailed():
    """Детальный тест поиска"""
    print("=== ДЕТАЛЬНЫЙ ТЕСТ ПОИСКА ===\n")
    
    try:
        search_engine = GeologicalSearchEngine()
        
        query = "месторождение песок гравий Борисово"
        print(f"Детальный поиск по запросу: '{query}'")
        print("="*60)
        
        results = search_engine.search_with_details(query, top_k=5)
        
        print(f"Всего найдено: {results['total_results']}")
        print(f"Файлов затронуто: {results['stats']['files_count']}")
        print(f"Средняя релевантность: {results['stats']['average_similarity']:.3f}")
        print(f"Максимальная релевантность: {results['stats']['max_similarity']:.3f}")
        
        print("\nТоп результаты:")
        print("-" * 40)
        
        for result in results['results']:
            print(f"\nРанг: {result['rank']}")
            print(f"Файл: {result['filename']}")
            print(f"Чанк: {result['chunk_id']}")
            print(f"Релевантность: {result['similarity']:.3f}")
            print(f"Длина текста: {len(result['text'])} символов")
            print(f"Текст: {result['text'][:200]}...")
        
        print("\nГруппировка по файлам:")
        print("-" * 40)
        for file_id, file_results in results['files_found'].items():
            print(f"Файл {file_id}: {len(file_results)} чанков")
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False
    
    return True

def interactive_search():
    """Интерактивный поиск"""
    print("=== ИНТЕРАКТИВНЫЙ ПОИСК ===")
    print("Введите 'exit' для выхода\n")
    
    try:
        search_engine = GeologicalSearchEngine()
        
        while True:
            query = input("Введите поисковый запрос: ").strip()
            
            if query.lower() in ['exit', 'quit', 'выход']:
                print("До свидания!")
                break
            
            if not query:
                continue
                
            print(f"\nПоиск по запросу: '{query}'")
            print("-" * 50)
            
            results = search_engine.search(query, top_k=5, min_similarity=0.1)
            
            if not results:
                print("Результатов не найдено (попробуйте другие ключевые слова)")
            else:
                print(f"Найдено {len(results)} релевантных результатов:")
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Файл: {result['filename']} (сходство: {result['similarity']:.3f})")
                    
                    # Показываем больше текста для интерактивного режима
                    text = result['text']
                    if len(text) > 300:
                        text = text[:300] + "..."
                    print(f"   {text}")
            
            print("\n" + "="*60 + "\n")
            
    except KeyboardInterrupt:
        print("\n\nВыход по Ctrl+C")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    print("ТЕСТИРОВАНИЕ ПОИСКОВОЙ СИСТЕМЫ")
    print("="*50)
    
    # Простой тест
    if test_search_simple():
        print("✅ Простой тест прошел успешно")
    else:
        print("❌ Простой тест не прошел")
        sys.exit(1)
    
    # Детальный тест
    if test_search_detailed():
        print("✅ Детальный тест прошел успешно")
    else:
        print("❌ Детальный тест не прошел")
        sys.exit(1)
    
    # Предложение интерактивного режима
    print("\n" + "="*50)
    choice = input("Хотите протестировать интерактивный поиск? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes', 'да', 'д']:
        interactive_search()
    else:
        print("Тестирование завершено!")