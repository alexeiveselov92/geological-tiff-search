#!/usr/bin/env python3
import sys
import os
from typing import Optional

# Добавляем текущую директорию в PATH для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search_engine import GeologicalSearchEngine
from rag_system import GeologicalRAGSystem
import config

class GeologicalInterface:
    """Простой интерфейс для взаимодействия с геологической системой"""
    
    def __init__(self):
        self.search_engine = None
        self.rag_system = None
        self.mode = "search"  # "search" или "rag"
        
        self.initialize_systems()
    
    def initialize_systems(self):
        """Инициализация систем поиска и RAG"""
        try:
            print("🔄 Инициализация систем...")
            
            # Инициализация поисковой системы
            self.search_engine = GeologicalSearchEngine()
            print("✅ Поисковая система готова")
            
            # Попытка инициализации RAG системы
            try:
                api_key = config.OPENAI_API_KEY
                if api_key and api_key != "your_api_key_here":
                    self.rag_system = GeologicalRAGSystem()
                    print("✅ RAG система готова")
                else:
                    print("⚠️  RAG система недоступна (не настроен OpenAI API ключ)")
                    print("   Доступен только режим поиска")
            except Exception as e:
                print(f"⚠️  RAG система недоступна: {e}")
                print("   Доступен только режим поиска")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            sys.exit(1)
    
    def show_main_menu(self):
        """Показать главное меню"""
        print("\n" + "="*60)
        print("🏔️  СИСТЕМА АНАЛИЗА ГЕОЛОГИЧЕСКИХ ДОКУМЕНТОВ")
        print("="*60)
        print("Доступные режимы:")
        print("1. 🔍 Поиск - быстрый поиск по документам")
        if self.rag_system:
            print("2. 🤖 RAG - ответы на вопросы с помощью ИИ")
        print("3. 📊 Статистика - информация о базе данных")
        print("4. ❓ Помощь - инструкции по использованию")
        print("5. 🚪 Выход")
        print("-"*60)
    
    def show_help(self):
        """Показать справку"""
        print("\n📖 СПРАВКА ПО ИСПОЛЬЗОВАНИЮ")
        print("="*50)
        print("🔍 РЕЖИМ ПОИСКА:")
        print("- Введите ключевые слова для поиска")
        print("- Примеры: 'месторождение', 'песок гравий', 'Борисово'")
        print("- Система найдет релевантные фрагменты документов")
        
        if self.rag_system:
            print("\n🤖 РЕЖИМ RAG (ИИ):")
            print("- Задавайте вопросы на естественном языке")
            print("- Примеры: 'Что известно о Борисовском месторождении?'")
            print("- ИИ даст развернутый ответ на основе документов")
        
        print("\n💡 СОВЕТЫ:")
        print("- Используйте русские термины")
        print("- Формулируйте конкретные вопросы")
        print("- В режиме поиска можно искать по именам, датам, местам")
        print("\n")
    
    def show_statistics(self):
        """Показать статистику базы данных"""
        try:
            index_data = self.search_engine.index_data
            print("\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
            print("="*40)
            print(f"Всего чанков: {index_data['total_chunks']}")
            print(f"Размерность векторов: {index_data['embedding_dim']}")
            print(f"Модель эмбеддингов: {index_data['model_name']}")
            
            # Подсчет уникальных файлов
            unique_files = set()
            for chunk in index_data['chunks']:
                unique_files.add(chunk['filename'])
            
            print(f"Уникальных документов: {len(unique_files)}")
            print(f"Документы: {', '.join(sorted(unique_files))}")
            
        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
    
    def search_mode(self):
        """Режим поиска"""
        print("\n🔍 РЕЖИМ ПОИСКА")
        print("Введите 'назад' для возврата в главное меню")
        print("-"*40)
        
        while True:
            try:
                query = input("\n🔎 Поисковый запрос: ").strip()
                
                if query.lower() in ['назад', 'back', 'exit']:
                    break
                
                if not query:
                    print("Пожалуйста, введите запрос")
                    continue
                
                print(f"\n🔍 Ищу: '{query}'...")
                
                results = self.search_engine.search_with_details(query, top_k=5)
                
                if not results['results']:
                    print("❌ Ничего не найдено")
                    continue
                
                print(f"\n📋 Найдено: {results['total_results']} результатов")
                print(f"📁 Файлов: {results['stats']['files_count']}")
                print("-"*50)
                
                for i, result in enumerate(results['results'], 1):
                    print(f"\n{i}. 📄 {result['filename']}")
                    print(f"   🎯 Сходство: {result['similarity']:.3f}")
                    print(f"   📝 Текст: {result['text'][:300]}{'...' if len(result['text']) > 300 else ''}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Ошибка поиска: {e}")
    
    def rag_mode(self):
        """Режим RAG (вопросы-ответы с ИИ)"""
        if not self.rag_system:
            print("❌ RAG система недоступна")
            return
        
        print("\n🤖 РЕЖИМ RAG (ВОПРОСЫ-ОТВЕТЫ)")
        print("Задавайте вопросы на естественном языке")
        print("Введите 'назад' для возврата в главное меню")
        print("-"*50)
        
        while True:
            try:
                question = input("\n❓ Ваш вопрос: ").strip()
                
                if question.lower() in ['назад', 'back', 'exit']:
                    break
                
                if not question:
                    print("Пожалуйста, введите вопрос")
                    continue
                
                print("\n🤔 Обрабатываю вопрос...")
                
                result = self.rag_system.ask_question(question)
                
                print("\n" + "="*60)
                print("🤖 ОТВЕТ:")
                print("="*60)
                print(result['answer'])
                print("="*60)
                
                if result['sources']:
                    print(f"\n📚 Источники (уверенность: {result['confidence']}):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['filename']} (релевантность: {source['similarity']:.3f})")
                
                if result['error']:
                    print(f"\n⚠️ Ошибка: {result['error']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Ошибка RAG системы: {e}")
    
    def run(self):
        """Основной цикл интерфейса"""
        print("🚀 Система готова к работе!")
        
        while True:
            try:
                self.show_main_menu()
                
                choice = input("\nВыберите режим (1-5): ").strip()
                
                if choice == '1':
                    self.search_mode()
                elif choice == '2' and self.rag_system:
                    self.rag_mode()
                elif choice == '2' and not self.rag_system:
                    print("❌ RAG система недоступна")
                elif choice == '3':
                    self.show_statistics()
                elif choice == '4':
                    self.show_help()
                elif choice == '5':
                    print("\n👋 До свидания!")
                    break
                else:
                    print("❌ Неверный выбор")
                
                # Пауза перед возвратом в меню
                if choice in ['3', '4']:
                    input("\nНажмите Enter для продолжения...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Неожиданная ошибка: {e}")

def main():
    """Точка входа в программу"""
    try:
        interface = GeologicalInterface()
        interface.run()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()