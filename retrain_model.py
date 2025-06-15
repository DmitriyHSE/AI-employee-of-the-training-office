import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
from pathlib import Path


class CorrectnessModelRetrainer:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
            )),
            ('regressor', LinearRegression())
        ])

    def load_data(self, filename='correctness_dataset.csv'):
        try:
            # Проверка существования файла
            if not Path(filename).exists():
                raise FileNotFoundError(f"Файл {filename} не найден")

            # Загрузка данных
            df = pd.read_csv(filename, encoding='utf-8')

            # Проверка структуры данных
            if df.empty:
                raise ValueError("Файл с данными пуст")

            # Приведение названий колонок к нижнему регистру
            df.columns = df.columns.str.lower()

            # Проверка наличия нужных колонок
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Файл должен содержать колонки 'text' и 'label'")

            # Очистка данных
            df = df.dropna(subset=['text', 'label'])
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df.dropna(subset=['label'])
            df['label'] = np.clip(df['label'], 0, 1)

            # Проверка минимального количества данных
            if len(df) < 2:
                raise ValueError(f"Недостаточно данных после очистки: {len(df)} строк")

            return df['text'], df['label']

        except Exception as e:
            print(f"\nОшибка загрузки данных: {str(e)}")
            print("Проверьте, что файл с данными:")
            print("1. Существует и доступен для чтения")
            print("2. Содержит колонки 'text' и 'label'")
            print("3. Содержит достаточное количество корректных данных")
            raise

    def train(self):
        try:
            # Загрузка данных
            X, y = self.load_data()

            print(f"\nДанные успешно загружены:")
            print(f"- Всего примеров: {len(X)}")

            # Для маленьких датасетов используем кросс-валидацию
            if len(X) < 10:
                print("- Используется кросс-валидация на всех данных")
                from sklearn.model_selection import cross_val_predict
                y_pred = cross_val_predict(self.model, X, y, cv=min(3, len(X)))

                print("\n=== Результаты кросс-валидации ===")
                print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
                print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
                print(f"R2 Score: {r2_score(y, y_pred):.4f}")
            else:
                # Для больших датасетов используем стандартное разделение
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                print(f"- Обучающих примеров: {len(X_train)}")
                print(f"- Тестовых примеров: {len(X_test)}")

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

                print("\n=== Результаты обучения ===")
                print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
                print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
                print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

            # Обучение финальной модели на всех данных
            self.model.fit(X, y)

            # Сохранение модели
            joblib.dump(self.model, 'correctness_model_v2.joblib')
            print("\nМодель успешно сохранена как 'correctness_model_v2.joblib'")

            return True

        except Exception as e:
            print(f"\nОшибка при обучении модели: {str(e)}")
            return False


if __name__ == '__main__':
    trainer = CorrectnessModelRetrainer()
    print("\nНачинаем переобучение модели...")
    success = trainer.train()

    if success:
        print("\nПереобучение завершено успешно!")
    else:
        print("\nПереобучение завершено с ошибками")