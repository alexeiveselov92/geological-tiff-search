import os
from typing import List, Dict, Optional
from search_engine import GeologicalSearchEngine
import config

try:
    import openai
except ImportError:
    print("❌ Ошибка: библиотека openai не установлена")
    print("Установите её командой: pip install openai")
    raise

class GeologicalRAGSystem:
    """RAG система для ответов на вопросы по геологическим документам"""
    
    def __init__(self, openai_api_key: str = None):
        # Настройка OpenAI
        if openai_api_key is None:
            openai_api_key = config.OPENAI_API_KEY
        
        if not openai_api_key or openai_api_key == "your_api_key_here":
            raise ValueError(
                "OpenAI API ключ не настроен! "
                "Добавьте ключ в .env файл: OPENAI_API_KEY=ваш_ключ"
            )
        
        # Простая инициализация без дополнительных параметров
        try:
            self.client = openai.OpenAI(api_key=openai_api_key)
        except Exception as e:
            # Попробуем альтернативный способ для старых версий
            try:
                openai.api_key = openai_api_key
                self.client = openai
                self.use_legacy_api = True
            except Exception as e2:
                raise ValueError(f"Не удалось инициализировать OpenAI клиент: {e}")
        else:
            self.use_legacy_api = False
        
        # Инициализация поисковой системы
        self.search_engine = GeologicalSearchEngine()
        
        print("RAG система инициализирована успешно!")
    
    def create_context_from_chunks(self, chunks: List[Dict], max_tokens: int = 6000) -> str:
        """Создание контекста из найденных чанков с учетом лимита токенов"""
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"\n--- Документ {chunk['filename']}, фрагмент {chunk['chunk_index']} ---\n{chunk['text']}\n"
            
            # Простая оценка количества токенов (примерно 1 токен = 4 символа для русского)
            estimated_tokens = len(chunk_text) // 4
            
            if current_length + estimated_tokens > max_tokens:
                break
            
            context_parts.append(chunk_text)
            current_length += estimated_tokens
        
        return "\n".join(context_parts)
    
    def create_system_prompt(self) -> str:
        """Создание системного промпта для OpenAI"""
        return """Ты - эксперт-геолог, который анализирует исторические геологические отчеты.

Твоя задача:
1. Внимательно изучить предоставленные фрагменты геологических документов
2. Ответить на вопрос пользователя на основе информации из документов
3. Если информации недостаточно, честно сказать об этом
4. Указать, из каких документов взята информация
5. Использовать профессиональную геологическую терминологию

Правила ответа:
- Отвечай только на русском языке
- Будь точен и конкретен
- Ссылайся на источники информации
- Если в документах нет ответа на вопрос, так и скажи
- Не придумывай информацию, которой нет в документах"""
    
    def ask_question(self, question: str, max_chunks: int = 5, 
                    min_similarity: float = 0.01) -> Dict:
        """
        Задать вопрос системе RAG
        
        Args:
            question: Вопрос пользователя
            max_chunks: Максимальное количество чанков для контекста
            min_similarity: Минимальный порог сходства для включения чанка
        
        Returns:
            Словарь с ответом и метаинформацией
        """
        try:
            # 1. Поиск релевантных документов
            search_results = self.search_engine.search(
                question, 
                top_k=max_chunks, 
                min_similarity=min_similarity
            )
            
            if not search_results:
                return {
                    "question": question,
                    "answer": "К сожалению, я не нашел релевантной информации в доступных геологических документах для ответа на ваш вопрос.",
                    "sources": [],
                    "confidence": "низкая",
                    "error": None
                }
            
            # 2. Создание контекста
            context = self.create_context_from_chunks(search_results)
            
            # 3. Формирование промпта
            user_prompt = f"""
Контекст из геологических документов:
{context}

Вопрос пользователя: {question}

Проанализируй предоставленные документы и дай развернутый ответ на вопрос. Обязательно укажи, из каких документов взята информация.
"""
            
            # 4. Запрос к OpenAI
            try:
                if hasattr(self, 'use_legacy_api') and self.use_legacy_api:
                    # Старый API
                    response = self.client.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": self.create_system_prompt()},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content.strip()
                else:
                    # Новый API
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": self.create_system_prompt()},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content.strip()
            except Exception as api_error:
                raise Exception(f"Ошибка при обращении к OpenAI API: {api_error}")
            
            # 5. Подготовка метаинформации
            sources = []
            for chunk in search_results:
                sources.append({
                    "filename": chunk["filename"],
                    "chunk_id": chunk["chunk_id"],
                    "similarity": round(chunk["similarity"], 3)
                })
            
            # Оценка уверенности на основе качества найденных документов
            avg_similarity = sum(chunk["similarity"] for chunk in search_results) / len(search_results)
            if avg_similarity > 0.3:
                confidence = "высокая"
            elif avg_similarity > 0.1:
                confidence = "средняя"
            else:
                confidence = "низкая"
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "chunks_used": len(search_results),
                "error": None
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Произошла ошибка при обработке вопроса: {str(e)}",
                "sources": [],
                "confidence": "ошибка",
                "error": str(e)
            }
    
    def interactive_session(self):
        """Интерактивная сессия вопросов-ответов"""
        print("\n=== ИНТЕРАКТИВНАЯ СЕССИЯ ГЕОЛОГИЧЕСКИХ ЗАПРОСОВ ===")
        print("Задавайте вопросы о геологических документах.")
        print("Для выхода введите 'выход', 'exit' или 'quit'\n")
        
        while True:
            try:
                question = input("\n❓ Ваш вопрос: ").strip()
                
                if question.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("До свидания!")
                    break
                
                if not question:
                    print("Пожалуйста, введите вопрос.")
                    continue
                
                print("\n🔍 Ищу информацию в документах...")
                
                result = self.ask_question(question)
                
                print(f"\n📋 Ответ:")
                print("=" * 50)
                print(result["answer"])
                print("=" * 50)
                
                if result["sources"]:
                    print(f"\n📚 Источники (уверенность: {result['confidence']}):")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"{i}. {source['filename']} (сходство: {source['similarity']})")
                
                if result["error"]:
                    print(f"\n⚠️ Ошибка: {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n\nДо свидания!")
                break
            except Exception as e:
                print(f"\nОшибка: {e}")

def test_rag_system():
    """Тестирование RAG системы"""
    try:
        # Проверяем наличие API ключа
        api_key = config.OPENAI_API_KEY
        if not api_key or api_key == "your_api_key_here":
            print("❌ OpenAI API ключ не настроен!")
            print("Для тестирования RAG системы необходимо:")
            print("1. Получить API ключ на https://platform.openai.com/")
            print("2. Добавить его в файл .env:")
            print("   OPENAI_API_KEY=your_actual_api_key")
            return None
        
        print("🚀 Инициализация RAG системы...")
        rag = GeologicalRAGSystem()
        
        # Тестовые вопросы
        test_questions = [
            "Что известно о Борисовском месторождении?",
            "Какие полезные ископаемые упоминаются в документах?",
            "В каком году проводились работы?",
            "Где расположено месторождение?",
            "Кто был главным геологом проекта?"
        ]
        
        print("\n=== ТЕСТИРОВАНИЕ RAG СИСТЕМЫ ===\n")
        
        for question in test_questions:
            print(f"\n📝 Вопрос: {question}")
            print("-" * 60)
            
            result = rag.ask_question(question)
            
            print(f"💬 Ответ: {result['answer']}")
            print(f"🎯 Уверенность: {result['confidence']}")
            
            if result["sources"]:
                print(f"📚 Источники: {', '.join([s['filename'] for s in result['sources']])}")
        
        return rag
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании RAG системы: {e}")
        return None

if __name__ == "__main__":
    rag_system = test_rag_system()
    if rag_system:
        # Запуск интерактивной сессии если система работает
        rag_system.interactive_session()