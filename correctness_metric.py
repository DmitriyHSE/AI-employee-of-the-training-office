import joblib
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression


class CorrectnessValidator:
    def __init__(self, model_path='correctness_model_v2.joblib'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            self._create_dummy_model()
            return False

    def _create_dummy_model(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('regressor', LinearRegression())
        ])
        print("Создана модель-заглушка. Все вопросы будут считаться корректными (1.0).")

    def predict_correctness(self, text: str) -> float:
        if not self.model:
            return 1.0

        try:
            score = max(0, min(1, float(self.model.predict([text])[0])))
            return round(score, 2)
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            return 1.0

    def calculate_correctness_batch(self, questions: List[str]) -> List[float]:
        if not questions:
            return []

        if not self.model:
            return [1.0] * len(questions)

        return [self.predict_correctness(q) for q in questions]