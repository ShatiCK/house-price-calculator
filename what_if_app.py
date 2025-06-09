import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
st.set_page_config(page_title="Оценка стоимости загородной недвижимости", layout="centered")
# === Загрузка данных и обучение модели ===
@st.cache_data
def load_and_train():
    df = pd.read_csv("clean_dataset.csv")

    X = df.drop(columns=["saleprice"])
    y = df["saleprice"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return model, X, feature_importance


# === Загрузка модели и данных ===
model, X, importance_df = load_and_train()

# === Заголовок и описание ===

st.title("📊 Онлайн-калькулятор цены загородной недвижимости")
st.markdown("📋 Введите параметры недвижимости — и узнайте, за сколько такой дом обычно продаётся на рынке.")

# === Словарь переводов ===
feature_labels = {
    "overallqual": "Общее качество (1–10)",
    "grlivarea": "Жилая площадь (кв. футы)",
    "totalbsmtsf": "Общая площадь подвала (кв. футы)",
    "2ndflrsf": "Площадь второго этажа (кв. футы)",
    "bsmtfinsf1": "Площадь отделанного подвала (кв. футы)",
    "1stflrsf": "Площадь первого этажа (кв. футы)",
    "lotarea": "Площадь участка (кв. футы)",
    "garagearea": "Площадь гаража (кв. футы)",
    "yearbuilt": "Год постройки",
    "garagecars": "Количество машиномест"
}

# === Реалистичные значения по умолчанию ===
preset_values = {
    "overallqual": 8,
    "grlivarea": 2000,
    "totalbsmtsf": 800,
    "2ndflrsf": 400,
    "bsmtfinsf1": 600,
    "1stflrsf": 1200,
    "lotarea": 9500,
    "garagearea": 400,
    "yearbuilt": 2015,
    "garagecars": 2
}

# === Ограничения по диапазонам ===
feature_ranges = {
    "overallqual":   {"min": 1,    "max": 10,   "step": 1,    "type": "int"},
    "grlivarea":     {"min": 100,  "max": 4000, "step": 10,   "type": "float"},
    "totalbsmtsf":   {"min": 0,    "max": 3000, "step": 10,   "type": "float"},
    "2ndflrsf":      {"min": 0,    "max": 2000, "step": 10,   "type": "float"},
    "bsmtfinsf1":    {"min": 0,    "max": 2000, "step": 10,   "type": "float"},
    "1stflrsf":      {"min": 100,  "max": 2500, "step": 10,   "type": "float"},
    "lotarea":       {"min": 1000, "max": 25000,"step": 100,  "type": "float"},
    "garagearea":    {"min": 0,    "max": 1500, "step": 10,   "type": "float"},
    "yearbuilt":     {"min": 1800, "max": 2025, "step": 1,    "type": "int"},
    "garagecars":    {"min": 0,    "max": 5,    "step": 1,    "type": "int"}
}

# === Получение топ-10 признаков ===
top_features = importance_df.head(10)['Feature']
sample = X.mean().to_dict()

# === Боковая панель ввода параметров ===
st.sidebar.header("🔧 Ввод характеристик")

for feature in top_features:
    if feature not in X.columns:
        continue

    label = feature_labels.get(feature.lower(), feature)
    default = preset_values.get(feature.lower(), float(sample[feature]))
    fr = feature_ranges.get(feature.lower(), {"min": 0, "max": 10000, "step": 1, "type": "float"})

    if fr["type"] == "int":
        sample[feature] = st.sidebar.number_input(
            label,
            min_value=int(fr["min"]),
            max_value=int(fr["max"]),
            step=int(fr["step"]),
            value=int(round(default)),
            key=feature
        )
    else:
        sample[feature] = st.sidebar.number_input(
            label,
            min_value=float(fr["min"]),
            max_value=float(fr["max"]),
            step=float(fr["step"]),
            value=float(default),
            key=feature
        )

# === Предсказание и вывод ===
input_df = pd.DataFrame([sample])
price = model.predict(input_df)[0]

st.markdown("---")
st.markdown("### 💰 Прогнозируемая стоимость объекта:")
st.success(f"**≈ {price:,.0f} USD**")
top10 = importance_df.head(10).sort_values(by="Importance")

# === Инструкция / советы ===
with st.expander("ℹ️ Как пользоваться приложением?"):
    st.markdown("""
    - Укажите реальные параметры вашего дома слева
    - Все значения уже заполнены, но вы можете изменить их под себя
    - Ничего нажимать не нужно — результат обновляется автоматически
    - Вы можете увидеть, какие характеристики влияют на цену сильнее всего — они отображены на диаграмме ниже
    """)

# === График важности факторов ===
with st.expander("🔍 Посмотреть, что влияет на стоимость"):
    labels = [feature_labels.get(f.lower(), f) for f in top10['Feature']]

    fig, ax = plt.subplots()
    ax.barh(labels, top10['Importance'], color='green')
    ax.set_title("Топ-10 признаков, влияющих на цену")
    ax.set_xlabel("Важность признака")
    st.pyplot(fig)



st.markdown("---")
st.caption("📘Прогноз рассчитан на основе анализа более 1400 реальных объектов недвижимости и учитывает ключевые параметры вашего дома. Модель прогнозирования построена на открытом наборе данных House Prices (Kaggle).")

# === Футер ===
st.markdown("""
<hr style="border:1px solid gray">

<p style='text-align: center; font-size: 0.85em; color: gray;'>
Создатели: Шатравский Никита Дмитриевич, Горшков Андрей Максимович<br>
Группа ФБИ-33, НГТУ, 2025 г.<br>
Все права защищены ©
</p>
""", unsafe_allow_html=True)

