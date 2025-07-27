#!/usr/bin/env python3
"""
Тестовый скрипт для проверки RAG системы
"""

import sys
import os

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from rag_system import GeologicalRAGSystem

def check_openai_key():
    """Проверка наличия OpenAI API ключа"""
    api_key = config.OPENAI_API_KEY
    if not api_key or api_key == "your_api_key_here":
        print("❌ OpenAI API ключ не настроен!")
        print("\nДля работы RAG системы необходимо:")
        print("1. Получить API ключ на https://platform.openai.com/")
        print("2. Создать/обновить файл .env в корне проекта:")
        print("   OPENAI_API_KEY=your_actual_api_key")
        return False
    
    print(f"✅ OpenAI API ключ найден: {api_key[:10]}...")
    return True

def test_rag_basic():
    """Базовое тестирование RAG системы"""
    print("=== БАЗОВОЕ ТЕСТИРОВАНИЕ RAG ===\n")
    
    try:
        # Инициализация
        print("🚀 Инициализация RAG системы...")
        rag = GeologicalRAGSystem()
        print("✅ RAG система инициализирована\n")
        
        # Простые тестовые вопросы
        test_questions = [
            "месторождение",
            "Борисово", 
            "песок и гравий"
        ]
        
        for question in test_questions:
            print(f"❓ Тестовый вопрос: '{question}'")
            print("-" * 50)
            
            result = rag.ask_question(question, max_chunks=3)
            
            print(f"💬 Ответ получен: {len(result['answer'])} символов")
            print(f"🎯 Уверенность: {result['confidence']}")
            print(f"📚 Источников использовано: {len(result['sources'])}")
            
            if result["error"]:
                print(f"⚠️ Ошибка: {result['error']}")
                return False
            
            # Показываем первые 200 символов ответа
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"🔤 Начало ответа: {answer_preview}")
            
            print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

def test_rag_detailed():
    """Детальное тестирование RAG системы"""
    print("=== ДЕТАЛЬНОЕ ТЕСТИРОВАНИЕ RAG ===\n")
    
    try:
        rag = GeologicalRAGSystem()
        
        # Более сложные вопросы
        detailed_questions = [
            "Что известно о Борисовском месторождении песка и гравия?",
            "В каком году проводились геологические работы?",
            "Кто был ответственным за проект в районе реки Протвы?"
        ]
        
        for question in detailed_questions:
            print(f"❓ Вопрос: {question}")
            print("="*70)
            
            result = rag.ask_question(question, max_chunks=5, min_similarity=0.01)
            
            print(f"💬 Полный ответ:")
            print(result['answer'])
            
            print(f"\n📊 Метаинформация:")
            print(f"   Уверенность: {result['confidence']}")
            print(f"   Чанков использовано: {result.get('chunks_used', 'неизвестно')}")
            
            if result["sources"]:
                print(f"\n📚 Источники:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"   {i}. {source['filename']} (сходство: {source['similarity']})")
            
            if result["error"]:
                print(f"\n⚠️ Ошибка: {result['error']}")
                return False
            
            print("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при детальном тестировании: {e}")
        return False

def test_edge_cases():
    """Тестирование граничных случаев"""
    print("=== ТЕСТИРОВАНИЕ ГРАНИЧНЫХ СЛУЧАЕВ ===\n")
    
    try:
        rag = GeologicalRAGSystem()
        
        edge_cases = [
            "",  # Пустой вопрос
            "абракадабра несуществующее слово",  # Нерелевантный вопрос
            "что" * 100,  # Очень длинный вопрос
        ]
        
        for i, question in enumerate(edge_cases, 1):
            question_desc = f"граничный случай {i}"
            if question == "":
                question_desc = "пустой вопрос"
            elif "абракадабра" in question:
                question_desc = "нерелевантный вопрос"
            elif len(question) > 50:
                question_desc = "очень длинный вопрос"
            
            print(f"🧪 Тест: {question_desc}")
            print("-" * 40)
            
            if question == "":
                print("⏭️ Пропускаем пустой вопрос")
                continue
            
            result = rag.ask_question(question, max_chunks=2, min_similarity=0.1)
            
            print(f"📝 Результат обработан: {len(result['answer'])} символов")
            print(f"🎯 Уверенность: {result['confidence']}")
            print(f"📚 Источников: {len(result['sources'])}")
            
            if result["error"]:
                print(f"⚠️ Ошибка (ожидаемо): {result['error']}")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании граничных случаев: {e}")
        return False

def performance_test():
    """Тест производительности"""
    print("=== ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ===\n")
    
    try:
        import time
        
        rag = GeologicalRAGSystem()
        
        question = "месторождение песок гравий"
        num_requests = 3
        
        print(f"🚀 Выполняю {num_requests} запросов для оценки производительности...")
        
        times = []
        for i in range(num_requests):
            start_time = time.time()
            result = rag.ask_question(question, max_chunks=3)
            end_time = time.time()
            
            request_time = end_time - start_time
            times.append(request_time)
            
            print(f"   Запрос {i+1}: {request_time:.2f} сек")
            
            if result["error"]:
                print(f"   ⚠️ Ошибка: {result['error']}")
                return False
        
        avg_time = sum(times) / len(times)
        print(f"\n📊 Статистика:")
        print(f"   Среднее время запроса: {avg_time:.2f} сек")
        print(f"   Минимальное время: {min(times):.2f} сек") 
        print(f"   Максимальное время: {max(times):.2f} сек")
        
        if avg_time < 10:
            print("✅ Производительность хорошая")
        elif avg_time < 20:
            print("⚠️ Производительность приемлемая")
        else:
            print("❌ Производительность низкая")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании производительности: {e}")
        return False

if __name__ == "__main__":
    print("ТЕСТИРОВАНИЕ RAG СИСТЕМЫ")
    print("="*50)
    
    # Проверка API ключа
    if not check_openai_key():
        print("\n❌ Тестирование невозможно без API ключа")
        sys.exit(1)
    
    print()
    
    # Базовое тестирование
    if test_rag_basic():
        print("✅ Базовое тестирование прошло успешно\n")
    else:
        print("❌ Базовое тестирование не прошло")
        sys.exit(1)
    
    # Детальное тестирование  
    if test_rag_detailed():
        print("✅ Детальное тестирование прошло успешно\n")
    else:
        print("❌ Детальное тестирование не прошло")
        sys.exit(1)
    
    # Тестирование граничных случаев
    if test_edge_cases():
        print("✅ Тестирование граничных случаев прошло успешно\n")
    else:
        print("❌ Тестирование граничных случаев не прошло")
        sys.exit(1)
    
    # Тест производительности
    if performance_test():
        print("✅ Тест производительности прошел успешно\n")
    else:
        print("❌ Тест производительности не прошел")
        sys.exit(1)
    
    print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("\n" + "="*50)
    print("RAG система готова к использованию!")
    print("Для интерактивного режима запустите:")
    print("python src/rag_system.py")