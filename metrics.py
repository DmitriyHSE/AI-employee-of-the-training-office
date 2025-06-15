import numpy as np
import pandas as pd
from typing import List
import ast
import os
import time
from correctness_metric import CorrectnessValidator

class ValidatorSimple:
    def __init__(self, neural: bool = False, batch_size: int = 16):
        self.correctness_validator = CorrectnessValidator()
        if not self.correctness_validator.model:
            print("Обучение модели корректности")
            self.correctness_validator.train_model()

    def validate_rag(self, test_set: pd.DataFrame) -> dict:
        # Проверяем необходимые колонки
        required_cols = ['answer', 'ground_truth', 'contexts']
        for col in required_cols:
            if col not in test_set.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")

        results = {
            'mean_answer_correctness_literal': self._calculate_answer_literal_batch(
                test_set['answer'].tolist(),
                test_set['ground_truth'].tolist()
            ),
            'mean_context_precision': self._calculate_context_precision_batch(
                test_set['ground_truth'].tolist(),
                test_set['contexts'].apply(self._safe_extract_texts).tolist()
            ),
            'mean_correctness_score': np.mean([
                self.correctness_validator.predict_correctness(q)
                for q in test_set['user_question'].fillna('')
            ])
        }

        if self.neural:
            results['mean_answer_correctness_neural'] = self._calculate_answer_neural_batch(
                test_set['answer'].tolist(),
                test_set['ground_truth'].tolist()
            )

        # Рассчитываем incorrect_ratio
        correctness_scores = [
            self.correctness_validator.predict_correctness(q)
            for q in test_set['user_question'].fillna('')
        ]
        results['incorrect_ratio'] = sum(1 for x in correctness_scores if x < 0.4) / len(correctness_scores)

        return results

    def _safe_extract_texts(self, ctx_list):
        try:
            contexts = ast.literal_eval(ctx_list) if isinstance(ctx_list, str) else ctx_list
            return [item.get('text', '') for item in contexts if isinstance(item, dict)]
        except:
            return []

    def _calculate_context_recall_batch(self, ground_truths: List[str], contexts_batch: List[List[str]]) -> List[float]:
        predictions = []
        references = []
        for gt, ctxs in zip(ground_truths, contexts_batch):
            predictions.extend(ctxs)
            references.extend([gt] * len(ctxs))

        scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=False,
            rouge_types=["rouge2"]
        )["rouge2"]

        result = []
        index = 0
        for ctxs in contexts_batch:
            n = len(ctxs)
            if n == 0:
                result.append(0.0)
            else:
                result.append(np.mean(scores[index:index+n]))
                index += n
        return result

    def _calculate_context_precision_batch(self, ground_truths: List[str], contexts_batch: List[List[str]]) -> List[float]:
        predictions = []
        references = []
        for gt, ctxs in zip(ground_truths, contexts_batch):
            predictions.extend(ctxs)
            references.extend([gt] * len(ctxs))

        scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = self.bleu.compute(
                    predictions=[pred],
                    references=[ref],
                    max_order=2
                )["precisions"][1]
            except ZeroDivisionError:
                score = 0.0
            scores.append(score)

        result = []
        index = 0
        for ctxs in contexts_batch:
            n = len(ctxs)
            if n == 0:
                result.append(0.0)
            else:
                result.append(np.mean(scores[index:index+n]))
                index += n
        return result

    def _calculate_answer_literal_batch(self, answers: List[str], ground_truths: List[str]) -> List[float]:
        # Возвращаем список результатов для каждого ответа
        return [
            self.chrf.compute(
                predictions=[ans],
                references=[gt]
            )["score"] for ans, gt in zip(answers, ground_truths)
        ]

    def _calculate_answer_neural_batch(self, answers: List[str], ground_truths: List[str]) -> List[float]:
        scores = self.bertscore.compute(
            predictions=answers,
            references=ground_truths,
            model_type=self.model_type,
            batch_size=self.batch_size,
            num_layers=11
        )["f1"]
        return scores

    def save_metrics(metrics: dict):
        try:
            # Загружаем существующие метрики
            try:
                with pd.ExcelFile(DATA_FILE, engine='openpyxl') as excel:
                    metrics_df = pd.read_excel(excel, sheet_name='metrics')
            except:
                metrics_df = pd.DataFrame()

            # Добавляем новую запись
            new_record = {
                'date': pd.to_datetime('today').date(),
                **{k: v for k, v in metrics.items() if not pd.isna(v)}
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_record])], ignore_index=True)

            # Сохраняем
            book = None
            try:
                book = pd.ExcelFile(DATA_FILE, engine='openpyxl').book
            except:
                pass

            with pd.ExcelWriter(DATA_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                if book:
                    writer.book = book
                metrics_df.to_excel(writer, sheet_name='metrics', index=False)
        except Exception as e:
            print(f"Ошибка сохранения метрик: {e}")

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        metrics = {
            'date': pd.to_datetime('today').date(),
            'mean_correctness_score': 0,
            'incorrect_ratio': 0,
            'mean_answer_quality': 0,
            'mean_response_time': 0
        }

        # 1. Метрики корректности вопросов
        if 'user_question' in df.columns:
            questions = df['user_question'].fillna('').astype(str).tolist()
            correctness_scores = self.correctness_validator.calculate_correctness_batch(questions)
            metrics.update({
                'mean_correctness_score': np.mean(correctness_scores),
                'incorrect_ratio': sum(1 for x in correctness_scores if x < 0.4) / len(correctness_scores)
            })

        # 2. Метрики качества ответов
        if 'winner' in df.columns:
            metrics['mean_answer_quality'] = (df['winner'] == 'giga_answer').mean()

        # 3. Метрики времени ответа
        if 'response_time' in df.columns:
            metrics['mean_response_time'] = df['response_time'].mean()

        return metrics


SOURCE_FILE = 'data.xlsx'  # Исходный файл данных
DATA_FILE = 'intermediate_data.xlsx'  # Промежуточный файл
BATCH_SIZE = 5  # Количество строк, добавляемых за один раз
INTERVAL = 60  # Интервал между добавлениями

def load_source_data():
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(f"Файл {SOURCE_FILE} не найден.")
    return pd.read_excel(SOURCE_FILE)

def simulate_data_upload(source_data: pd.DataFrame, intermediate_file: str):
    total_rows = len(source_data)
    uploaded_rows = 0

    while uploaded_rows < total_rows:
        start_index = uploaded_rows
        end_index = min(uploaded_rows + BATCH_SIZE, total_rows)

        # Выбираем пакет данных
        batch = source_data.iloc[start_index:end_index]

        # Загружаем текущие данные из промежуточного файла
        try:
            existing_data = pd.read_excel(intermediate_file)
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=batch.columns)

        # Объединяем существующие данные с новыми
        updated_data = pd.concat([existing_data, batch], ignore_index=True)

        updated_data.to_excel(intermediate_file, index=False)
        print(f"Добавлено {len(batch)} строк. Всего в промежуточном файле: {len(updated_data)} строк.")

        uploaded_rows += len(batch)

        time.sleep(INTERVAL)

if __name__ == "__main__":
    print("Начинаем симуляцию подгрузки данных...")
    source_data = load_source_data()
    simulate_data_upload(source_data, DATA_FILE)
    print("Симуляция завершена. Все данные загружены.")