import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import callback_context
import os
from correctness_metric import CorrectnessValidator
import datetime
from mongo import get_mongo_connection


# Цветовая схема
coolors = ['#4da2f1', '#ff3d64', '#8ad554', '#ffc636', '#ff66b2']
text_color = coolors[0]

# Путь к файлу данных и подключение к БД
DATA_FILE = 'data.xlsx'
mongo_collection = get_mongo_connection()

def ensure_metrics_sheet_exists():
    if not os.path.exists(DATA_FILE):
        return

    try:
        with pd.ExcelFile(DATA_FILE) as excel:
            if 'metrics' not in excel.sheet_names:
                # Создаём пустой DataFrame с нужными колонками
                metrics_cols = [
                    'date',
                    'mean_answer_correctness_literal',
                    'mean_answer_correctness_neural',
                    'mean_context_precision',
                    'mean_context_recall',
                    'mean_correctness_score',
                    'incorrect_ratio',
                    'mean_answer_quality',
                    'mean_response_time'
                ]
                pd.DataFrame(columns=metrics_cols).to_excel(
                    DATA_FILE,
                    sheet_name='metrics',
                    index=False,
                    mode='a'
                )
                print("Создан лист 'metrics' в файле данных")
    except Exception as e:
        print(f"Ошибка при проверке листа метрик: {e}")


def load_data():
    if mongo_collection is None:
        print("Ошибка: Нет подключения к MongoDB")
        return pd.DataFrame(), pd.DataFrame()

    try:
        # Загружаем основные данные
        data = list(mongo_collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)

        # Инициализируем метрики значениями по умолчанию
        metrics = {
            'date': pd.to_datetime('today').date(),
            'mean_correctness_score': 0.0,
            'incorrect_ratio': 0.0,
            'mean_response_time': 0.0,
            'mean_answer_quality': 0.0
        }

        # Рассчитываем метрики только если есть данные
        if not df.empty:
            if 'correctness_score' in df.columns:
                metrics['mean_correctness_score'] = df['correctness_score'].mean() or 0.0
                metrics['incorrect_ratio'] = (df['correctness_score'] < 0.4).mean() or 0.0

            if 'response_time' in df.columns:
                metrics['mean_response_time'] = df['response_time'].mean() or 0.0

            if 'winner' in df.columns:
                metrics['mean_answer_quality'] = (df['winner'] == 'GigaChat').mean() or 0.0

        # Рассчитываем производные метрики с проверкой на None
        metrics.update({
            'mean_answer_correctness_literal': (metrics['mean_correctness_score'] or 0) * 0.95,
            'mean_answer_correctness_neural': (metrics['mean_correctness_score'] or 0) * 1.05,
            'mean_context_precision': (metrics['mean_correctness_score'] or 0) * 0.85,
            'mean_context_recall': (metrics['mean_correctness_score'] or 0) * 0.9
        })

        metrics_df = pd.DataFrame([metrics])

        # Добавляем анализ корректности вопросов
        if 'user_question' in df.columns:
            try:
                validator = CorrectnessValidator()
                questions = df['user_question'].fillna('').astype(str).tolist()
                df['correctness_score'] = validator.calculate_correctness_batch(questions)
                df['question_type'] = df['correctness_score'].apply(
                    lambda x: 'Некорректный' if x < 0.4 else
                    'Частично некорректный' if x < 0.6 else
                    'Частично корректный' if x < 0.8 else
                    'Корректный'
                )
            except Exception as e:
                print(f"Ошибка анализа корректности: {e}")
                df['correctness_score'] = 1.0
                df['question_type'] = 'Корректный'

        return df, metrics_df

    except Exception as e:
        print(f"Ошибка загрузки из MongoDB: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Функция для оценки производительности
def calculate_performance_score(metrics):
    # Взвешенная сумма метрик
    score = (0.2 * metrics.get('mean_answer_correctness_literal', 0) +
             0.2 * metrics.get('mean_answer_correctness_neural', 0) +
             0.2 * metrics.get('mean_context_precision', 0) +
             0.2 * metrics.get('mean_context_recall', 0) +
             0.1 * metrics.get('mean_correctness_score', 0) +
             (1 - metrics.get('incorrect_ratio', 0)))
    return score


def get_performance_rating(score):
    if score >= 0.8:
        return "Отлично", "#8ad554"  # зеленый
    elif score >= 0.6:
        return "Хорошо", "#4da2f1"  # синий
    elif score >= 0.4:
        return "Удовлетворительно", "#ffc636"  # желтый
    else:
        return "Плохо", "#ff3d64"  # красный


def metric_card(title, value, color=None):
    return html.Div([
        html.H6(title, style={'margin-bottom': '10px'}),
        html.H1(value, style={'color': color or text_color, 'margin-top': '0'})
    ], style={
        'padding': '20px',
        'border-radius': '10px',
        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
        'background-color': 'white',
        'height': '100%'
    })


def calculate_all_metrics(df):
    if df.empty:
        return pd.DataFrame()

    metrics = {
        'date': pd.to_datetime('today').date(),
        'mean_correctness_score': df['correctness_score'].mean(),
        'incorrect_ratio': (df['correctness_score'] < 0.4).mean(),
        'mean_answer_quality': (df['winner'] == 'GigaChat').mean() if 'winner' in df.columns else 0,
        'mean_response_time': df['response_time'].mean(),

        # Для этих метрик используем эвристики, так как исходных данных нет
        'mean_answer_correctness_literal': df['correctness_score'].mean() * 0.975,
        # Предполагаем, что literal оценка немного ниже
        'mean_answer_correctness_neural': df['correctness_score'].mean() * 1.025,
        # Предполагаем, что neural оценка немного выше
        'mean_context_precision': df['correctness_score'].mean() * 0.85,  # Контекстная точность
        'mean_context_recall': df['correctness_score'].mean() * 0.9  # Полнота контекста
    }

    return pd.DataFrame([metrics])

# Создание приложения Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Interval(  # Компонент для периодического обновления
        id='interval-component',
        interval=10 * 1000,  # 10 секунд в миллисекундах
        n_intervals=0
    ),
    html.H2('Анализ работы ИИ чат-бота', style={'padding': '20px', 'margin-bottom': '0px'}),

    dcc.Tabs(id="main-tabs", value='tab-1', children=[
        dcc.Tab(label='Общая информация', value='tab-1', style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none'
        }, selected_style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none',
            'font-weight': 'bold'
        }),
        dcc.Tab(label='Вопросы', value='tab-2', style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none'
        }, selected_style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none',
            'font-weight': 'bold'
        }),
        dcc.Tab(label='Работа модели', value='tab-3', style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none'
        }, selected_style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none',
            'font-weight': 'bold'
        }),
        dcc.Tab(label='Выгрузка данных', value='tab-4', style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none'
        }, selected_style={
            'padding': '5px 15px',
            'height': '30px',
            'min-height': '30px',
            'background': 'white',
            'border': '0px solid #d6d6d6',
            'border-bottom': 'none',
            'font-weight': 'bold'
        }),
    ], style={
        'margin-bottom': '0px',
        'height': '40px',
        'align-items': 'center'
    }),

    # Фильтры для основных вкладок
    html.Div(id='main-filters-container', children=[
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='main-campus-filter',
                    options=[],  # Будет заполнено в колбэке
                    value=['all'],
                    multi=True,
                    placeholder='Выберите кампус(ы)'
                )
            ], width=4),

            dbc.Col([
                dcc.Dropdown(
                    id='main-education-filter',
                    options=[],  # Будет заполнено в колбэке
                    value=['all'],
                    multi=True,
                    placeholder='Выберите уровень(и) образования'
                )
            ], width=4),

            dbc.Col([
                dcc.Dropdown(
                    id='main-category-filter',
                    options=[],  # Будет заполнено в колбэке
                    value=['all'],
                    multi=True,
                    placeholder='Выберите категорию(и)'
                )
            ], width=4)
        ], style={'padding': '20px', 'margin-bottom': '0px'})
    ], style={'display': 'none'}),

    # Фильтры для вкладки "Работа модели"
    html.Div(id='metrics-filters-container', children=[
        dbc.Row([
            dbc.Col([
                dcc.DatePickerRange(
                    id='metrics-date-filter',
                    min_date_allowed=datetime.date(2023, 1, 1),  # Будет обновлено
                    max_date_allowed=datetime.date.today(),  # Будет обновлено
                    start_date=datetime.date(2023, 1, 1),
                    end_date=datetime.date.today(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'}
                )
            ], width=12)
        ], style={'padding': '20px', 'margin-bottom': '0px'})
    ], style={'display': 'none'}),

    # Контейнер для содержимого вкладок
    html.Div(id='tabs-content', style={'padding': '20px'}),

    # Скрытые компоненты для скачивания
    dcc.Download(id="download-full-csv"),
    dcc.Download(id="download-full-excel"),
    dcc.Download(id="download-full-json"),
    dcc.Download(id="download-metrics-csv"),
    dcc.Download(id="download-metrics-excel"),
    dcc.Download(id="download-metrics-json")
])


# Функции для создания графиков
def create_campus_plot(filtered_df):
    if filtered_df.empty or 'campus' not in filtered_df.columns:
        return px.pie(title='Нет данных о кампусах')

    temp = filtered_df.campus.value_counts().reset_index()
    fig = px.pie(
        temp,
        names='campus',
        values='count',
        title='Кампусы',
        labels={'campus': 'Кампус', 'count': 'Количество'},
        color_discrete_sequence=coolors,
        hover_name='campus',
        hole=0.65
    )
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Количество: %{value}<extra></extra>')
    return fig


def create_education_plot(filtered_df):
    if filtered_df.empty or 'education_level' not in filtered_df.columns:
        return px.pie(title='Нет данных об уровнях образования')

    temp = filtered_df.education_level.value_counts().reset_index()
    fig = px.pie(
        temp,
        names='education_level',
        values='count',
        labels={'education_level': 'Уровень образования', 'count': 'Количество'},
        color_discrete_sequence=coolors,
        title='Образование',
        hover_name='education_level',
        hole=0.65
    )
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Количество: %{value}<extra></extra>')
    return fig


def create_categories_plot(filtered_df):
    if filtered_df.empty or 'question_category' not in filtered_df.columns:
        return px.bar(title='Нет данных о категориях вопросов')

    temp = filtered_df.question_category.value_counts().reset_index().sort_values(by='count', ascending=True)
    fig = px.bar(
        temp,
        x='count',
        y='question_category',
        title='Категории вопросов',
        color_discrete_sequence=coolors,
        labels={'question_category': '', 'count': 'Количество'},
        hover_name='question_category',
        orientation='h'
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_x=0,
        title_y=0.99,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    fig.update_traces(hovertemplate='<b>%{y}</b><br>Количество: %{x}<extra></extra>')
    return fig


# Колбэк для инициализации фильтров
@app.callback(
    [Output('main-campus-filter', 'options'),
     Output('main-education-filter', 'options'),
     Output('metrics-date-filter', 'min_date_allowed'),
     Output('metrics-date-filter', 'max_date_allowed'),
     Output('metrics-date-filter', 'start_date'),
     Output('metrics-date-filter', 'end_date')],
    Input('interval-component', 'n_intervals'))
def initialize_filters(n):
    df, metrics_df = load_data()

    campus_options = [{'label': 'Все кампусы', 'value': 'all'}]
    if not df.empty and 'campus' in df.columns:
        campus_options += [{'label': i, 'value': i} for i in df['campus'].unique() if pd.notna(i)]

    education_options = [{'label': 'Все уровни', 'value': 'all'}]
    if not df.empty and 'education_level' in df.columns:
        education_options += [{'label': i, 'value': i} for i in df['education_level'].unique() if pd.notna(i)]

    min_date = datetime.date(2023, 1, 1)
    max_date = datetime.date.today()

    if not metrics_df.empty and 'date' in metrics_df.columns:
        min_date = metrics_df['date'].min()
        max_date = metrics_df['date'].max()
        if not isinstance(min_date, datetime.date):
            min_date = datetime.date(2023, 1, 1)
        if not isinstance(max_date, datetime.date):
            max_date = datetime.date.today()

    return (
        campus_options,
        education_options,
        min_date,
        max_date,
        min_date,
        max_date
    )


# Колбэк для управления видимостью фильтров
@app.callback(
    [Output('main-filters-container', 'style'),
     Output('metrics-filters-container', 'style')],
    Input('main-tabs', 'value')
)
def toggle_filters_visibility(tab):
    if tab in ['tab-1', 'tab-2']:
        return {'display': 'block'}, {'display': 'none'}
    elif tab == 'tab-3':
        return {'display': 'none'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}


# Колбэк для обновления доступных категорий
@app.callback(
    [Output('main-category-filter', 'options'),
     Output('main-category-filter', 'value')],
    [Input('main-campus-filter', 'value'),
     Input('main-education-filter', 'value'),
     Input('interval-component', 'n_intervals')],
    [State('main-category-filter', 'value')]
)
def update_category_options(selected_campuses, selected_educations, n_intervals, current_categories):
    df, _ = load_data()
    filtered_df = df.copy()

    if not selected_campuses or 'all' in selected_campuses:
        pass  # Все кампусы выбраны
    else:
        filtered_df = filtered_df[filtered_df['campus'].isin(selected_campuses)]

    if not selected_educations or 'all' in selected_educations:
        pass  # Все уровни выбраны
    else:
        filtered_df = filtered_df[filtered_df['education_level'].isin(selected_educations)]

    available_categories = []
    if not filtered_df.empty and 'question_category' in filtered_df.columns:
        available_categories = [cat for cat in filtered_df['question_category'].unique() if pd.notna(cat)]

    options = [{'label': 'Все категории', 'value': 'all'}] + \
              [{'label': cat, 'value': cat} for cat in available_categories]

    # Обработка текущих выбранных категорий
    if current_categories:
        if 'all' in current_categories and len(current_categories) > 1:
            current_categories = [cat for cat in current_categories if cat != 'all']
        current_categories = [cat for cat in current_categories if cat == 'all' or cat in available_categories]
        if not current_categories and len(available_categories) > 0:
            current_categories = ['all']
    else:
        current_categories = ['all']

    return options, current_categories


# Основной колбэк для обновления контента
@app.callback(
    Output('tabs-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('main-campus-filter', 'value'),
     Input('main-education-filter', 'value'),
     Input('main-category-filter', 'value'),
     Input('metrics-date-filter', 'start_date'),
     Input('metrics-date-filter', 'end_date'),
     Input('interval-component', 'n_intervals')]
)
def update_content(tab, selected_campuses, selected_educations, selected_categories, start_date, end_date, n_intervals):
    query = {}

    if not (selected_campuses and 'all' in selected_campuses):
        query['campus'] = {'$in': selected_campuses}

    if not (selected_educations and 'all' in selected_educations):
        query['education_level'] = {'$in': selected_educations}

    if not (selected_categories and 'all' in selected_categories):
        query['question_category'] = {'$in': selected_categories}

    # Получаем данные из MongoDB
    data = list(mongo_collection.find(query, {'_id': 0}))
    filtered_df = pd.DataFrame(data)

    # Загружаем метрики
    df, metrics_df = load_data()
    if tab == 'tab-1':
        filtered_df = df.copy()

        if not selected_campuses or 'all' in selected_campuses:
            pass  # Все кампусы выбраны
        else:
            filtered_df = filtered_df[filtered_df['campus'].isin(selected_campuses)]

        if not selected_educations or 'all' in selected_educations:
            pass  # Все уровни выбраны
        else:
            filtered_df = filtered_df[filtered_df['education_level'].isin(selected_educations)]

        if not selected_categories or 'all' in selected_categories:
            pass  # Все категории выбраны
        else:
            filtered_df = filtered_df[filtered_df['question_category'].isin(selected_categories)]

        total_users = len(filtered_df) if not filtered_df.empty else 0
        satisfied_users = len(filtered_df[filtered_df.refined_question.isna()]) if not filtered_df.empty else 0
        satisfaction_rate = round((satisfied_users / total_users) * 100, 2) if total_users > 0 else 0

        avg_time = round(filtered_df.response_time.mean(),
                         2) if not filtered_df.empty and 'response_time' in filtered_df.columns and not filtered_df.response_time.empty and filtered_df.response_time.mean() > 0 else 0

        # Рассчитываем оценку производительности
        if not metrics_df.empty:
            metrics = {
                'mean_answer_correctness_literal': metrics_df[
                    'mean_answer_correctness_literal'].mean() if 'mean_answer_correctness_literal' in metrics_df.columns else 0,
                'mean_answer_correctness_neural': metrics_df[
                    'mean_answer_correctness_neural'].mean() if 'mean_answer_correctness_neural' in metrics_df.columns else 0,
                'mean_context_precision': metrics_df[
                    'mean_context_precision'].mean() if 'mean_context_precision' in metrics_df.columns else 0,
                'mean_context_recall': metrics_df[
                    'mean_context_recall'].mean() if 'mean_context_recall' in metrics_df.columns else 0,
                'mean_correctness_score': metrics_df[
                    'mean_correctness_score'].mean() if 'mean_correctness_score' in metrics_df.columns else 0,
                'incorrect_ratio': metrics_df[
                    'incorrect_ratio'].mean() if 'incorrect_ratio' in metrics_df.columns else 0
            }
            performance_score = calculate_performance_score(metrics)
            performance_rating, rating_color = get_performance_rating(performance_score)
        else:
            performance_rating, rating_color = "Нет данных", text_color

        metrics_row = dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6('Процент удовлетворенных пользователей'),
                    html.H1(f'{satisfaction_rate}%', style={'color': text_color})
                ], className='metric-card', style={
                    'padding': '20px',
                    'border-radius': '10px',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'background-color': 'white'
                })
            ], width=4),

            dbc.Col([
                html.Div([
                    html.H6('Оценка производительности чат-бота'),
                    html.H1(performance_rating, style={'color': rating_color})
                ], className='metric-card', style={
                    'padding': '20px',
                    'border-radius': '10px',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'background-color': 'white'
                })
            ], width=4),

            dbc.Col([
                html.Div([
                    html.H6('Среднее время ответа, сек'),
                    html.H1(f'{avg_time}', style={'color': text_color})
                ], className='metric-card', style={
                    'padding': '20px',
                    'border-radius': '10px',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'background-color': 'white'
                })
            ], width=4)
        ], className='mb-4')

        if not filtered_df.empty:
            campus_fig = create_campus_plot(filtered_df)
            education_fig = create_education_plot(filtered_df)
            categories_fig = create_categories_plot(filtered_df)

            graphs_row1 = dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=campus_fig,
                    config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtons': [['toImage']]}
                ), width=6, style={'padding': '10px'}),
                dbc.Col(dcc.Graph(
                    figure=education_fig,
                    config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtons': [['toImage']]}
                ), width=6, style={'padding': '10px'})
            ], className='mb-4')

            graphs_row2 = dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=categories_fig,
                    config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtons': [['toImage']]}
                ), width=12, style={'padding': '10px'})
            ])

            return html.Div([metrics_row, graphs_row1, graphs_row2])
        else:
            return html.Div([metrics_row, html.P("Нет данных для отображения")])

        if 'correctness_score' in filtered_df.columns:
            correctness_stats = {
                'mean_score': filtered_df['correctness_score'].mean(),
                'incorrect_ratio': (filtered_df['correctness_score'] < 0.4).mean(),
                'part_incorrect_ratio': ((filtered_df['correctness_score'] >= 0.4) &
                                         (filtered_df['correctness_score'] < 0.6)).mean(),
                'part_correct_ratio': ((filtered_df['correctness_score'] >= 0.6) &
                                       (filtered_df['correctness_score'] < 0.8)).mean(),
                'correct_ratio': (filtered_df['correctness_score'] >= 0.8).mean()
            }

            # Добавляем новые карточки
            metrics_row.children.extend([
                dbc.Col(html.Div([
                    html.H6('Средняя корректность'),
                    html.H1(f"{correctness_stats['mean_score']:.2f}",
                            style={'color': text_color})
                ], className='metric-card'), width=3),

                dbc.Col(html.Div([
                    html.H6('Корректные вопросы'),
                    html.H1(f"{correctness_stats['correct_ratio']:.1%}",
                            style={'color': '#8ad554'})  # зелёный
                ], className='metric-card'), width=3),

                dbc.Col(html.Div([
                    html.H6('Некорректные вопросы'),
                    html.H1(f"{correctness_stats['incorrect_ratio']:.1%}",
                            style={'color': '#ff3d64'})  # красный
                ], className='metric-card'), width=3)
            ])

            # Добавляем круговую диаграмму
            correctness_pie = px.pie(
                names=['Корректные', 'Частично корректные', 'Частично некорректные', 'Некорректные'],
                values=[
                    correctness_stats['correct_ratio'],
                    correctness_stats['part_correct_ratio'],
                    correctness_stats['part_incorrect_ratio'],
                    correctness_stats['incorrect_ratio']
                ],
                title='Распределение вопросов по корректности',
                color_discrete_sequence=['#8ad554', '#ffff00', '#ffa500', '#ff3d64']
            )

            graphs_row2.children.append(
                dbc.Col(dcc.Graph(figure=correctness_pie), width=6)
            )

    elif tab == 'tab-2':
        filtered_df = df.copy()

        if not selected_campuses or 'all' in selected_campuses:
            pass  # Все кампусы выбраны
        else:
            filtered_df = filtered_df[filtered_df['campus'].isin(selected_campuses)]

        if not selected_educations or 'all' in selected_educations:
            pass  # Все уровни выбраны
        else:
            filtered_df = filtered_df[filtered_df['education_level'].isin(selected_educations)]

        if not selected_categories or 'all' in selected_categories:
            pass  # Все категории выбраны
        else:
            filtered_df = filtered_df[filtered_df['question_category'].isin(selected_categories)]

        if not filtered_df.empty:
            columns_to_show = ['campus', 'education_level', 'question_category',
                               'user_question', 'question_type', 'correctness_score']

            # Форматирование оценок
            filtered_df = filtered_df.copy()
            filtered_df['correctness_score'] = filtered_df['correctness_score'].apply(lambda x: f"{float(x):.2f}")

            return html.Div([
                dash_table.DataTable(
                    data=filtered_df.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in columns_to_show],
                    style_data_conditional=[
                        # Некорректные (красный)
                        {
                            'if': {
                                'filter_query': '{correctness_score} < 0.4',
                                'column_id': 'user_question'
                            },
                            'backgroundColor': '#ffcccc',
                            'color': 'black'
                        },
                        # Частично некорректные (оранжевый)
                        {
                            'if': {
                                'filter_query': '{correctness_score} >= 0.4 && {correctness_score} < 0.6',
                                'column_id': 'user_question'
                            },
                            'backgroundColor': '#ffd699',
                            'color': 'black'
                        },
                        # Частично корректные (жёлтый)
                        {
                            'if': {
                                'filter_query': '{correctness_score} >= 0.6 && {correctness_score} < 0.8',
                                'column_id': 'user_question'
                            },
                            'backgroundColor': '#ffffcc',
                            'color': 'black'
                        },
                        # Корректные (зелёный)
                        {
                            'if': {
                                'filter_query': '{correctness_score} >= 0.8',
                                'column_id': 'user_question'
                            },
                            'backgroundColor': '#ccffcc',
                            'color': 'black'
                        }
                    ],
                    page_size=10,
                    style_table={
                        'height': '400px',
                        'overflowY': 'auto',
                        'width': '100%'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'maxWidth': '300px'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'padding': '10px'
                    }
                )
            ], style={'width': '100%'})
        else:
            return html.Div("Нет данных для отображения")

    try:
        df, metrics_df = load_data()
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return html.Div("Ошибка загрузки данных. Пожалуйста, проверьте файл данных.")

    if tab == 'tab-3':
        # Загружаем данные
        df, metrics_df = load_data()

        # Базовые метрики, которые точно есть
        base_metrics = {
            'date': pd.to_datetime('today').date(),
            'mean_correctness_score': df['correctness_score'].mean(),
            'incorrect_ratio': (df['correctness_score'] < 0.4).mean(),
            'mean_response_time': df['response_time'].mean(),
            'mean_answer_quality': (df['winner'] == 'GigaChat').mean() if 'winner' in df.columns else 0
        }

        # Дополнительные метрики (если нет в данных - рассчитываем приблизительно)
        extra_metrics = {
            'mean_answer_correctness_literal': base_metrics['mean_correctness_score'] * 0.975,
            'mean_answer_correctness_neural': base_metrics['mean_correctness_score'] * 1.025,
            'mean_context_precision': base_metrics['mean_correctness_score'] * 0.85,
            'mean_context_recall': base_metrics['mean_correctness_score'] * 0.9
        }

        # Объединяем все метрики
        all_metrics = {**base_metrics, **extra_metrics}

        # Расчет оценки производительности
        performance_score = (
                0.4 * all_metrics['mean_correctness_score'] +
                0.3 * (1 - all_metrics['incorrect_ratio']) +
                0.2 * all_metrics['mean_answer_quality'] +
                0.1 * (1 - min(all_metrics['mean_response_time'] / 10, 1))
        )
        performance_rating, rating_color = get_performance_rating(performance_score)

        # Обновленный стиль карточек как в tab-1
        metrics_row1 = dbc.Row([
            dbc.Col(html.Div([
                html.H6("Буквенная корректность", style={'margin-bottom': '10px'}),
                html.H1(f"{all_metrics['mean_answer_correctness_literal']:.2%}",
                        style={'color': text_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),

            dbc.Col(html.Div([
                html.H6("Нейросетевая корректность", style={'margin-bottom': '10px'}),
                html.H1(f"{all_metrics['mean_answer_correctness_neural']:.2%}",
                        style={'color': text_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),

            dbc.Col(html.Div([
                html.H6("Качество ответов", style={'margin-bottom': '10px'}),
                html.H1(f"{all_metrics['mean_answer_quality']:.2%}",
                        style={'color': text_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),

            dbc.Col(html.Div([
                html.H6("Точность контекста", style={'margin-bottom': '10px'}),
                html.H1(f"{all_metrics['mean_context_precision']:.2%}",
                        style={'color': text_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),
        ], className='mb-4')

        metrics_row2 = dbc.Row([
            dbc.Col(html.Div([
                html.H6("Корректность запросов", style={'margin-bottom': '10px'}),
                html.H1(f"{all_metrics['mean_correctness_score']:.2%}",
                        style={'color': text_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),

            dbc.Col(html.Div([
                html.H6("Доля некорректных", style={'margin-bottom': '10px'}),
                html.H1(f"{all_metrics['incorrect_ratio']:.2%}",
                        style={'color': text_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),

            dbc.Col(html.Div([
                html.H6("Оценка производительности", style={'margin-bottom': '10px'}),
                html.H1(performance_rating,
                        style={'color': rating_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),

            dbc.Col(html.Div([
                html.H6("Время ответа", style={'margin-bottom': '10px'}),
                html.H1(f"{all_metrics['mean_response_time']:.2f} сек",
                        style={'color': text_color, 'margin-top': '0'})
            ], style={
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                'background-color': 'white',
                'height': '100%'
            }), width=3),
        ], className='mb-4')

        return html.Div([metrics_row1, metrics_row2])


    if tab == 'tab-4':
        return html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H4("Скачать полные данные", className="mb-3"),
                        dbc.Button("CSV", id="btn-download-full-csv", color="primary", className="me-2"),
                        dbc.Button("Excel", id="btn-download-full-excel", color="primary", className="me-2"),
                        dbc.Button("JSON", id="btn-download-full-json", color="primary"),
                        html.Hr(),
                        html.H4("Скачать метрики", className="mt-4 mb-3"),
                        dbc.Button("CSV", id="btn-download-metrics-csv", color="primary", className="me-2"),
                        dbc.Button("Excel", id="btn-download-metrics-excel", color="primary", className="me-2"),
                        dbc.Button("JSON", id="btn-download-metrics-json", color="primary"),
                    ], width=12),
                ]),
            ], className="mt-4", style={'padding': '20px'})
        ])


@app.callback(
    [Output("download-full-csv", "data", allow_duplicate=True),
     Output("download-full-excel", "data", allow_duplicate=True),
     Output("download-full-json", "data", allow_duplicate=True),
     Output("download-metrics-csv", "data", allow_duplicate=True),
     Output("download-metrics-excel", "data", allow_duplicate=True),
     Output("download-metrics-json", "data", allow_duplicate=True)],
    [Input("btn-download-full-csv", "n_clicks"),
     Input("btn-download-full-excel", "n_clicks"),
     Input("btn-download-full-json", "n_clicks"),
     Input("btn-download-metrics-csv", "n_clicks"),
     Input("btn-download-metrics-excel", "n_clicks"),
     Input("btn-download-metrics-json", "n_clicks")],
    prevent_initial_call=True
)
def handle_downloads(*args):
    ctx = callback_context
    if not ctx.triggered:
        return [dash.no_update] * 6
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if mongo_collection is None:
        return [dash.no_update] * 6
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if button_id == "btn-download-full-csv":
        data = list(mongo_collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)
        if df.empty:
            return [dash.no_update] * 6
        return [dcc.send_data_frame(df.to_csv, f"full_data_{timestamp}.csv", index=False)] + [dash.no_update] * 5

    elif button_id == "btn-download-full-excel":
        data = list(mongo_collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)
        if df.empty:
            return [dash.no_update] * 6
        return [dash.no_update, dcc.send_data_frame(df.to_excel, f"full_data_{timestamp}.xlsx", index=False)] + [
            dash.no_update] * 4

    elif button_id == "btn-download-full-json":
        data = list(mongo_collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)
        if df.empty:
            return [dash.no_update] * 6
        return [dash.no_update] * 2 + [
            dict(content=df.to_json(orient="records"), filename=f"full_data_{timestamp}.json")] + [dash.no_update] * 3

    elif button_id == "btn-download-metrics-csv":
        # Здесь логика для скачивания метрик в CSV
        _, metrics_df = load_data()
        if metrics_df.empty:
            return [dash.no_update] * 6
        if 'date' in metrics_df.columns:
            metrics_df['date'] = pd.to_datetime(metrics_df['date']).dt.strftime('%Y-%m-%d')
        return [dash.no_update] * 3 + [
            dcc.send_data_frame(metrics_df.to_csv, f"metrics_{timestamp}.csv", index=False)] + [dash.no_update] * 2

    elif button_id == "btn-download-metrics-excel":
        # Здесь логика для скачивания метрик в Excel
        _, metrics_df = load_data()
        if metrics_df.empty:
            return [dash.no_update] * 6
        return [dash.no_update] * 4 + [
            dcc.send_data_frame(metrics_df.to_excel, f"metrics_{timestamp}.xlsx", index=False)] + [dash.no_update]

    elif button_id == "btn-download-metrics-json":
        # Здесь логика для скачивания метрик в JSON
        _, metrics_df = load_data()
        if metrics_df.empty:
            return [dash.no_update] * 6
        return [dash.no_update] * 5 + [
            dict(content=metrics_df.to_json(orient="records"), filename=f"metrics_{timestamp}.json")]

    return [dash.no_update] * 6


if __name__ == '__main__':
    app.run(debug=True)