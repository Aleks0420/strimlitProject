# cd E:\pythonProject\strimlitProject
# streamlit run .\main.py
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.express as px                             # графические обьекты со встроенными функциями
import plotly.graph_objects as go                       # графические обьекты более низкоуровневые


# созаем контейнер заголовка, данных, предсказания и обучение модели
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
interactive = st.container()                    # интерактивный контейнер


# изменим фон приложения для улучшения внешнего вида
st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


background_color = '#F5F5F5'                                    # цвет фона


# поместим код который мы хотим запускать один раз в функцию получения данных
# и кешируем его с помощью декоратора
# и до тех пор пока имя файла не изменится эта функция не будет запускаться
@st.cache
def get_data(filename):
    """функция получения данных из файла"""
    taxi_data = pd.read_csv(filename)  # загрузим данные
    return taxi_data


# заполним контейнеры
with header:
    st.title("Это основной жирный текст заголовка")
    st.text("Дополнительный текст пояснялка")

with dataset:
    st.header("Набор данных такси New York")
    st.text("Здесь я раскажу где я этот набор возьму")

    taxi_data = get_data("data/taxi_data.csv")                              # загрузим данные
    # st.write(taxi_data.head())                         # и отобразим в контейнере голову таблицы

    st.subheader("Этот подзаголовок для гистограммы")
    pulocation_dist = pd.DataFrame(taxi_data["start_trip_area"].value_counts())#.head(50))      # укажем сколько раз и с какого места забирали пассажира
    st.bar_chart(pulocation_dist)                                               # построим гистограмму


with features:
    st.header("Создадим функции")
    # создадим список функций
    st.markdown('* **Первая функция** я что то создаю чтобы было хорошо')
    st.markdown('* **Вторая функция** я еще что то создаю чтобы было лучше')

with model_training:
    st.header("Обучим модель")
    st.text("Опишу обучение модели")

    # создадим раздел пользовательского ввода

    # создадим колонки 2 в контейнере колонку выбора и колонку отображения
    sel_col, disp_col = st.columns(2)

    # созадим слайдер указав диапазон, минимум,максимум, значение по умолчанию и шаг
    max_depth = sel_col.slider('Какая максимальная глубина дерева?',
                               min_value=10,
                               max_value=100,
                               value=20,
                               step=10)

    # создадим выпадающий список указав параметры для выбора и индекс элемента по умолчанию
    n_estimators = sel_col.selectbox("Какое количество деревьев?",
                                    options=[100, 200, 300, 'No limits'],
                                    index=0)

    sel_col.text('Здесь лист в колонками для выбора данных')
    sel_col.write(taxi_data.columns)

    # создадим функцию ввода от пользователя указав идентификатор местоположения по умолчанию
    input_feature = sel_col.text_input('Введем значение функции в ручную',
                                       'dropoff_latitude')


    # добавим модель случайного леса

    if n_estimators == "No limits":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


    X = taxi_data[[input_feature]]
    y = taxi_data[['prev_trip_area']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    # отобразим производительность модели
    disp_col.subheader('Средняя абсолютная ошибка:')
    # напишем абсолютную ошибку дадим ей занчение и прогноз
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Средняя квадратичная ошибка:')
    # напишем абсолютную ошибку дадим ей занчение и прогноз
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('оценка модели в r2:')
    # напишем абсолютную ошибку дадим ей занчение и прогноз
    disp_col.write(r2_score(y, prediction))


# работа с таблицей
with interactive:
    st.title('Более пристальный взгляд на данные')
    # создаем таблицу передав заголовок (шапка таблицы) и ячейки (тело таблицы)
    fig = go.Figure(data=go.Table(
        columnwidth=[3, 1, 1],
        header=dict(values=list(taxi_data[['pickup_datetime', 'end_trip_area', 'total_amount']].columns),
                    fill_color='#FD8E72',
                    align='left'),
        cells=dict(values=[taxi_data.pickup_datetime, taxi_data.end_trip_area, taxi_data.total_amount],
                   fill_color='#E5ECF6',
                   align='left')))


    fig.update_layout(margin=dict(l=5, r=5, b=10, t=10),
                      paper_bgcolor=background_color)# обновить макет уменьшив поля
    st.write(fig)                                           # отобразим обьект



    # круговые диаграммы
    # столбец ввода и стобец круговой диаграммы
    input_col, pie_col = st.columns(2)
    pulocation_dist = pulocation_dist.reset_index()
    pulocation_dist.columns = ["start_trip_area", 'count']
    # print(pulocation_dist)

    top_n = input_col.text_input('Какое количество локаций хотим увидеть', 8)
    top_n = int(top_n)

    pulocation_dist= pulocation_dist.head(top_n)
    # изобразим на круговой диаграмме
    fig = px.pie(pulocation_dist, values='count', names='start_trip_area', hover_name="start_trip_area")
    # включим обновление макета отключим боковые индикаторы диаграммы, размер, изменим поля
    fig.update_layout(showlegend=False,
                      width=320,
                      height=320,
                      margin=dict(l=1, r=1, b=1, t=1),
                      paper_bgcolor=background_color,
                      font=dict(color='#383635', size=15))
    pie_col.write(fig)


    # линейные диаграммы
    line_chart_data = taxi_data.copy()                                  # скопируем данные
    # извлечем информацию из колонки с датой о часах
    line_chart_data['pickup_datetime'] = pd.to_datetime(line_chart_data['pickup_datetime'])
    line_chart_data['pickup_hour'] = line_chart_data['pickup_datetime'].dt.hour

    # создадим перекрестную вкладку: две колонки одна колонка место посадки вторая час
    # считаем сколько раз встретилось пересечение и заносим в перекрестную таблицу
    hour_cross_tab = pd.crosstab( line_chart_data['pickup_hour'], line_chart_data['start_trip_area'])
    # print(hour_cross_tab)

    # отобразим все на линейной диаграмме
    fig = px.line(hour_cross_tab)
    # настроим параметры отображения уберем легенду, изменим размер, поля, цвет фона и цвет шрифта
    fig.update_layout(showlegend=False,
                      width=800,
                      height=500,
                      margin=dict(l=1, r=1, b=1, t=1),
                      paper_bgcolor=background_color,
                      font=dict(color='#383635', size=15))

    # отрадактируем ось со временем чтоб было видно все часы сделав оси категориальными
    fig.update_xaxes(type='category')


    st.write(fig)
