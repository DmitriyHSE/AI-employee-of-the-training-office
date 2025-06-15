from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd
import os


def get_mongo_connection():
    try:
        # Подключение к MongoDB
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        client.server_info()  # Проверка подключения

        db = client['chatbot_analysis']
        collection = db['user_responses']

        # Если коллекция пуста - загружаем данные из Excel
        if collection.count_documents({}) == 0 and os.path.exists('data.xlsx'):
            df = pd.read_excel('data.xlsx')
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)

            # Конвертируем DataFrame в словари для MongoDB
            data = df.to_dict('records')
            collection.insert_many(data)

        return collection

    except ConnectionFailure as e:
        print(f"MongoDB недоступен: {e}")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None