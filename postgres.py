import psycopg2
from psycopg2 import sql
from datetime import date


# Подключение к базе данных
def get_connection():
    return psycopg2.connect(
        dbname="hseproject",
        user="postgres",
        password="603065",
        host="localhost",
        port="5432"
    )


# Функция для добавления записи в knowledge_base
def add_to_knowledge_base(conn):
    print("\nДобавление записи в таблицу knowledge_base")
    print("Оставьте поле пустым, чтобы установить NULL\n")

    fields = {
        'selected_role': input("Выберите роль (selected_role): "),
        'campus': input("Кампус (campus): "),
        'education_level': input("Уровень образования (education_level): "),
        'question_category': input("Категория вопроса (question_category): "),
        'user_question': input("Вопрос пользователя (user_question): "),
        'user_filters': input("Фильтры пользователя (user_filters, через запятую): "),
        'question_filters': input("Фильтры вопроса (question_filters, через запятую): "),
        'saiga_answer': input("Ответ Saiga (saiga_answer): "),
        'giga_answer': input("Ответ Giga (giga_answer): "),
        'winner': input("Победитель (winner): "),
        'comment': input("Комментарий (comment): "),
        'contexts': input("Контексты (contexts, в JSON формате): "),
        'refined_question': input("Уточненный вопрос (refined_question): "),
        'refined_answer': input("Уточненный ответ (refined_answer): "),
        'refined_contexts': input("Уточненные контексты (refined_contexts, в JSON формате): "),
        'response_time': input("Время ответа (response_time): "),
        'refined_response_time': input("Время уточненного ответа (refined_response_time): "),
        'comment_status': input("Статус комментария (comment_status): ")
    }

    # Обработка пустых значений и специальных типов
    for key, value in fields.items():
        if value == '':
            fields[key] = None
        elif key in ['user_filters', 'question_filters'] and value is not None:
            fields[key] = [v.strip() for v in value.split(',')] if value else None
        elif key in ['contexts', 'refined_contexts'] and value is not None:
            fields[key] = value if value else None
        elif key in ['response_time', 'refined_response_time'] and value is not None:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = None

    # Формирование SQL запроса
    columns = []
    values = []
    for col, val in fields.items():
        if val is not None:
            columns.append(sql.Identifier(col))
            values.append(val)

    query = sql.SQL("INSERT INTO knowledge_base ({}) VALUES ({})").format(
        sql.SQL(', ').join(columns),
        sql.SQL(', ').join([sql.Placeholder()] * len(columns))
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, values)
            conn.commit()
            print("Запись успешно добавлена в knowledge_base!")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при добавлении записи: {e}")


# Функция для добавления записи в metrics
def add_to_metrics(conn):
    print("\nДобавление записи в таблицу metrics")
    print("Оставьте поле пустым, чтобы установить NULL\n")

    fields = {
        'date': input("Дата (date, YYYY-MM-DD): "),
        'mean_answer_correctness_literal': input("Средняя корректность ответа (литеральный метод): "),
        'mean_answer_correctness_neural': input("Средняя корректность ответа (нейросетевой метод): "),
        'mean_context_precision': input("Средняя точность контекста: "),
        'mean_context_recall': input("Средняя полнота контекста: ")
    }

    # Обработка пустых значений и специальных типов
    for key, value in fields.items():
        if value == '':
            fields[key] = None
        elif key == 'date' and value is not None:
            try:
                fields[key] = date.fromisoformat(value)
            except ValueError:
                print("Неверный формат даты. Используйте YYYY-MM-DD")
                fields[key] = None
        elif key in ['mean_answer_correctness_literal', 'mean_answer_correctness_neural',
                     'mean_context_precision', 'mean_context_recall'] and value is not None:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = None

    # Формирование SQL запроса
    columns = []
    values = []
    for col, val in fields.items():
        if val is not None:
            columns.append(sql.Identifier(col))
            values.append(val)

    query = sql.SQL("INSERT INTO metrics ({}) VALUES ({})").format(
        sql.SQL(', ').join(columns),
        sql.SQL(', ').join([sql.Placeholder()] * len(columns))
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, values)
            conn.commit()
            print("Запись успешно добавлена в metrics!")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при добавлении записи: {e}")


# Главное меню
def main():
    conn = None
    try:
        conn = get_connection()
        print("Успешное подключение к базе данных")

        while True:
            print("\nГлавное меню:")
            print("1. Добавить запись в knowledge_base")
            print("2. Добавить запись в metrics")
            print("3. Выход")

            choice = input("Выберите действие: ")

            if choice == '1':
                add_to_knowledge_base(conn)
            elif choice == '2':
                add_to_metrics(conn)
            elif choice == '3':
                print("Выход из программы")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")

    except Exception as e:
        print(f"Ошибка подключения к базе данных: {e}")
    finally:
        if conn is not None:
            conn.close()
            print("Соединение с базой данных закрыто")


if __name__ == "__main__":
    main()