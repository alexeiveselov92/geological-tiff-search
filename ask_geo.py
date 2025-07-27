#!/usr/bin/env python3
"""
Простой интерфейс для работы с геологическими документами
Автор: Claude Code Assistant
"""

import sys
import os

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import config
from rag_system import GeologicalRAGSystem


def main():
    """Главная функция программы"""
    print("🏔️  СИСТЕМА АНАЛИЗА ГЕОЛОГИЧЕСКИХ ДОКУМЕНТОВ")
    print("=" * 60)
    print("Анализ 267 отсканированных геологических отчетов")
    print("Задавайте вопросы на русском языке о содержимом документов")
    print("=" * 60)

    # Проверка API ключа
    api_key = config.OPENAI_API_KEY
    if not api_key or api_key == "your_api_key_here":
        print("❌ ОШИБКА: OpenAI API ключ не настроен!")
        print("\nДля работы системы необходимо:")
        print("1. Получить API ключ на https://platform.openai.com/")
        print("2. Создать файл .env в этой папке:")
        print("   OPENAI_API_KEY=ваш_ключ_здесь")
        print("\nИли задать переменную окружения:")
        print("   export OPENAI_API_KEY=ваш_ключ_здесь")
        return

    try:
        # Инициализация системы
        print("\n🚀 Загружаю систему...")
        rag = GeologicalRAGSystem()
        print("✅ Система готова к работе!\n")

        # Показать примеры вопросов
        print("💡 Примеры вопросов:")
        examples = [
            "Что известно о Борисовском месторождении?",
            "Какие полезные ископаемые упоминаются?",
            "В каком году проводились работы?",
            "Где расположены месторождения?",
            "Кто был ответственным за проекты?",
        ]
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")

        print("\n" + "=" * 60)
        print("Введите 'выход' для завершения работы")
        print("=" * 60)

        # Основной цикл
        while True:
            try:
                question = input("\n❓ Ваш вопрос: ").strip()

                # Проверка на выход
                if question.lower() in ["выход", "exit", "quit", "q", "стоп"]:
                    print("\n👋 До свидания!")
                    break

                # Проверка на пустой ввод
                if not question:
                    print("💭 Пожалуйста, введите вопрос или 'выход' для завершения")
                    continue

                # Показать что система работает
                print("\n🔍 Анализирую документы...")

                # Получить ответ
                result = rag.ask_question(question)

                # Показать результат
                print("\n📋 ОТВЕТ:")
                print("=" * 60)
                print(result["answer"])
                print("=" * 60)

                # Показать источники
                if result["sources"]:
                    print(f"\n📚 Источники информации (уверенность: {result['confidence']}):")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"   {i}. {source['filename']} (релевантность: {source['similarity']:.1%})")
                else:
                    print("\n📚 Источники: информация не найдена в документах")

                # Показать ошибки если есть
                if result.get("error"):
                    print(f"\n⚠️ Предупреждение: {result['error']}")

            except KeyboardInterrupt:
                print("\n\n👋 Завершение работы по Ctrl+C")
                break
            except Exception as e:
                print(f"\n❌ Произошла ошибка: {e}")
                print("Попробуйте переформулировать вопрос или обратитесь к администратору")

    except Exception as e:
        print(f"\n❌ Ошибка инициализации системы: {e}")
        print("\nВозможные причины:")
        print("- Неверный API ключ OpenAI")
        print("- Отсутствуют необходимые файлы данных")
        print("- Проблемы с интернет-соединением")
        return


def show_help():
    """Показать справку"""
    help_text = """
🏔️  СИСТЕМА АНАЛИЗА ГЕОЛОГИЧЕСКИХ ДОКУМЕНТОВ

ИСПОЛЬЗОВАНИЕ:
    python ask_geo.py          - запуск интерактивного режима
    python ask_geo.py --help   - показать эту справку

ТРЕБОВАНИЯ:
    - Python 3.7+
    - OpenAI API ключ в файле .env
    - Обработанные геологические документы

ПРИМЕРЫ ВОПРОСОВ:
    - "Что известно о месторождениях песка и гравия?"
    - "В каких районах проводились работы?"
    - "Кто был ответственным за проекты в 1959 году?"
    - "Какие полезные ископаемые упоминаются в отчетах?"

КОМАНДЫ В ИНТЕРАКТИВНОМ РЕЖИМЕ:
    выход, exit, quit, q - завершение работы
    
ФАЙЛЫ КОНФИГУРАЦИИ:
    .env - содержит OPENAI_API_KEY=ваш_ключ
    
ТЕХНИЧЕСКАЯ ПОДДЕРЖКА:
    Проверьте наличие всех файлов в папках data/
    При ошибках API проверьте ключ OpenAI
    """
    print(help_text)


if __name__ == "__main__":
    # Проверка аргументов командной строки
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h", "help"]:
            show_help()
        else:
            print(f"❌ Неизвестный аргумент: {sys.argv[1]}")
            print("Используйте --help для справки")
    else:
        main()
