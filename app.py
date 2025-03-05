import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Cargar el dataset
data_path = r"C:\Users\Francesc\Documents\GitHub\won_analysis\datasets\days_to_start_new.csv"
df = pd.read_csv(data_path)

# Añadir features temporales
df['day_of_week'] = pd.to_datetime(df['buy_date'], dayfirst=True).dt.dayofweek
df['week_of_year'] = pd.to_datetime(df['buy_date'], dayfirst=True).dt.isocalendar().week
df['year_sale'] = pd.to_datetime(df['buy_date'], dayfirst=True).dt.year

# Función para generar rangos
def generar_rangos(max_dias):
    intervalo = 30
    bins = list(range(0, max_dias + 1, intervalo)) + [float('inf')]
    labels = [f"{i}-{i+intervalo-1}" for i in range(0, max_dias - intervalo + 1, intervalo)] + [f"{max_dias}+"]
    return bins, labels

# Preparar el modelo
max_dias = 150
bins, labels = generar_rangos(max_dias)
df['rango_dias'] = pd.cut(df['days_to_start'], bins=bins, labels=labels, right=False)
available_columns = ['Program', 'month_sale', 'month_diff', 'year_sale', 'day_of_week', 'week_of_year']
X = pd.get_dummies(df[available_columns], columns=['Program'])
y = df['rango_dias']
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X, y)

# Función de predicción híbrida (80% histórico + 20% modelo)
def predecir_distribucion_hibrida(programa, cantidad_ventas, mes, año=2025):
    historico_mes = df[(df['Program'] == programa) & (df['month_sale'] == mes)]
    if historico_mes.empty:
        return None
    
    nueva_data = pd.DataFrame({
        'Program': [programa],
        'month_sale': [mes],
        'month_diff': [0],
        'year_sale': [año],
        'day_of_week': [3],
        'week_of_year': [mes * 4]
    })
    nueva_data = pd.get_dummies(nueva_data, columns=['Program'])
    nueva_data = nueva_data.reindex(columns=X.columns, fill_value=0)
    probs = model.predict_proba(nueva_data)[0]
    prediccion = (probs * cantidad_ventas).round().astype(int)
    
    historico_dist = historico_mes['rango_dias'].value_counts(normalize=True).reindex(labels, fill_value=0)
    prediccion_corregida = (prediccion * 0.2 + historico_dist * cantidad_ventas * 0.8).round().astype(int)
    
    diferencia = cantidad_ventas - prediccion_corregida.sum()
    if diferencia != 0:
        ajuste = (historico_dist * diferencia).round().astype(int)
        prediccion_corregida += ajuste
        while prediccion_corregida.sum() != cantidad_ventas:
            if prediccion_corregida.sum() > cantidad_ventas:
                idx_max = prediccion_corregida.argmax()
                prediccion_corregida[idx_max] -= 1
            else:
                idx_min = prediccion_corregida.argmin()
                prediccion_corregida[idx_min] += 1
    
    return dict(zip(labels, prediccion_corregida))

# Función de predicción simple (promedio histórico)
def predecir_distribucion_historica(programa, cantidad_ventas, mes):
    historico_mes = df[(df['Program'] == programa) & (df['month_sale'] == mes)]
    if historico_mes.empty:
        return None
    
    historico_dist = historico_mes['rango_dias'].value_counts(normalize=True).reindex(labels, fill_value=0)
    prediccion_historica = (historico_dist * cantidad_ventas).round().astype(int)
    
    diferencia = cantidad_ventas - prediccion_historica.sum()
    if diferencia != 0:
        ajuste = (historico_dist * diferencia).round().astype(int)
        prediccion_historica += ajuste
        while prediccion_historica.sum() != cantidad_ventas:
            if prediccion_historica.sum() > cantidad_ventas:
                idx_max = prediccion_historica.argmax()
                prediccion_historica[idx_max] -= 1
            else:
                idx_min = prediccion_historica.argmin()
                prediccion_historica[idx_min] += 1
    
    return dict(zip(labels, prediccion_historica))

# Interfaz de Streamlit
st.title("Predicción de Ventas por Rango de Días (2025)")
st.write("Explora cómo se distribuyen las ventas previstas para 2025 según nuestro modelo híbrido y el promedio histórico.")

programa = st.selectbox("Programa", ["FS", "DS", "CS"])
cantidad_ventas = st.number_input("Cantidad de Ventas", min_value=1, value=50)
mes = st.slider("Mes", 1, 12, 8)

if st.button("Predecir"):
    # Calcular ambas predicciones
    resultado_hibrido = predecir_distribucion_hibrida(programa, cantidad_ventas, mes)
    resultado_historico = predecir_distribucion_historica(programa, cantidad_ventas, mes)
    
    if resultado_hibrido is None or resultado_historico is None:
        st.write("No hay datos suficientes para llevar a cabo una predicción.")
    else:
        # Crear un DataFrame comparativo
        df_comparativo = pd.DataFrame({
            "Días": labels,
            "Predicción Híbrida": [resultado_hibrido[rango] for rango in labels],
            "Predicción Simple": [resultado_historico[rango] for rango in labels]
        })
        
        st.subheader(f"Comparación de Predicciones para {cantidad_ventas} ventas de {programa} en el mes {mes} (2025):")
        st.dataframe(df_comparativo)
