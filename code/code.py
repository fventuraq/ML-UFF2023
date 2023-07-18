# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 08:15:44 2023

@author: FRANKLIN
"""
import pandas as pd
from dateutil.parser import parse
from datetime import datetime
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from scipy.interpolate import CubicSpline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

label_encoder = LabelEncoder()

#%%

df = pd.read_csv("F:/UFF/DOCTORADO/MACHINE_LEARNING/TRABALHO_FINAL/data/mergeData2.csv")

#%%
variables = df.columns.tolist()
print(variables)

#%%
#=========TRANSFORMAR 'DataIni', 'DataFim'  A VALORES DATETIME
df['DataIni'] = pd.to_datetime(df['DataIni'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['DataIni']) #delete 'NaN'

#=========TRANSFORMAR 'DataIni',  A VALORES DATETIME
df['DataFim'] = pd.to_datetime(df['DataFim'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['DataFim']) #delete 'NaN'

print(df['DataFim'].dtype)

#%%
#=========TRANSFORMAR 'HoraIni', 'HoraFim'  A VALORES DATETIME
df['HoraIniFormatted'] = pd.to_datetime(df['HoraIni'], format='%H:%M:%S', errors='coerce')
df['HoraFimFormatted'] = pd.to_datetime(df['HoraFim'], format='%H:%M:%S', errors='coerce')

print(df['HoraIni'])
print(df['HoraFim'])
print(df['HoraIniFormatted'].dtype)
print(df['HoraFimFormatted'].dtype)

#%%

# Paso 1: Convertir a columna 'DuraçãoViagem' a tipo timedelta
df['DuraÆoViagem'] = pd.to_timedelta(df['DuraÆoViagem'])

# Paso 2: Extraer la duracao em segundos
df['DuracaoSegundos'] = df['DuraÆoViagem'].dt.total_seconds()

df['DuracaoSegundos']
#%%
#===============ELIMINACAO DAS VARIAVEIS NAO USADAS=================

df = df.drop(['HoraIni', 'HoraFim', 'DuraÆoViagem'], axis=1)

#%%
#========TRANSFORMAR LA DISTANCIA A FORMATO (KM)===================

# Paso 1: Remover caracteres no numéricos
df['KmPerc'] = df['KmPerc'].str.replace(',', '')

# Paso 2: Convertir a tipo float
df['KmPerc'] = df['KmPerc'].astype(float)

# Paso 3: Dividir por 1000 para obtener la distancia en kilómetros
df['KmPerc'] = df['KmPerc'] / 1000  

print(df['HoraFimFormatted'])

#%%
#==============Plot classes Sentido=============== 

sentido_counts = df['Sentido'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(sentido_counts.index, sentido_counts.values)
plt.xlabel('Sentido')
plt.ylabel('Número de classes')
plt.title('Número de Classes para a Variável"Sentido"')
plt.show()

# Variable "Linha"
linha_counts = df['Linha'].value_counts()
plt.figure(figsize=(12, 6))
plt.bar(linha_counts.index, linha_counts.values)
plt.xlabel('Linha')
plt.ylabel('Número de classes')
plt.title('Número de Classes para a Variável "Linha"')
plt.xticks(rotation=90)
plt.show()

# variable 'Linha_encode'
linha_counts_2 = df['Linha_encoded'].value_counts()
plt.figure(figsize=(12, 6))
plt.bar(linha_counts_2.index, linha_counts_2.values)
plt.xlabel('Linha_2')
plt.ylabel('Número de classes')
plt.title('Número de Classes para a Variável "Linha_2"')
plt.xticks(rotation=90)
plt.show()

# Variable "NoVe¡culo"
noveiculo_counts = df['NoVe¡culo'].astype(str).value_counts()
plt.figure(figsize=(12, 6))
plt.bar(noveiculo_counts.index, noveiculo_counts.values)
plt.xlabel('NoVeiculo')
plt.ylabel('Número de Clases')
plt.title('Número de Classes para a Variável "NoVeiculo"')
plt.xticks(rotation=90)
plt.show()

noveiculo_counts_2 = df['NoVe¡culo_encoded'].astype(str).value_counts()
plt.figure(figsize=(12, 6))
plt.bar(noveiculo_counts_2.index, noveiculo_counts_2.values)
plt.xlabel('NoVeiculo_2')
plt.ylabel('Número de Clases')
plt.title('Número de Classes para a Variável "NoVeiculo_2"')
plt.xticks(rotation=90)
plt.show()

print(df['Linha'][50000])

#%%
#=======COLUMN 'LINHA'============================
one_hot_encoded_linha = pd.get_dummies(df['Linha'])
#merged_df = pd.concat([merged_df, one_hot_encoded], axis=1)

#frequencies_linha = df['Linha'].value_counts()
#df['Linha'] = df['Linha'].map(frequencies_linha)
#print(frequencies_linha['410'])

#ERROR AL APLICAR YA QUE PUEDE QUE ALGUNAS CLASES TENGAN LA MISMA FRECUENCIA....
#APLICAMOS CODIFICACION SIMPLE
df['Linha_encoded'] = label_encoder.fit_transform(df['Linha'].astype(str))

#parcear cuales tienen una frecuencia igual o menor a 10 y eliminarlas
#categorias_a_eliminar_linha = frequencies_linha[frequencies_linha < 10].index #NO EJECUTADO
#df = df[~df['Linha'].isin(categorias_a_eliminar_linha)] #NO EJECUTADO

#%%
#=======COLUMN 'NoVe¡culo'============================
#one_hot_encoded_nveiculo = pd.get_dummies(df['NoVe¡culo'])
#merged_df = pd.concat([merged_df, one_hot_encoded_nveiculo], axis=1)

#frequencies_nvehiculo = df['NoVe¡culo'].value_counts()
#df['NoVe¡culo'] = df['NoVe¡culo'].map(frequencies_nvehiculo)

#ERROR AL APLICAR YA QUE PUEDE QUE ALGUNAS CLASES TENGAN LA MISMA FRECUENCIA....
#APLICAMOS CODIFICACION SIMPLE

# Codificar la variable 'NoVe¡culo'
df['NoVe¡culo_encoded'] = label_encoder.fit_transform(df['NoVe¡culo'].astype(str))

#parcear cuales tienen una frecuencia igual o menor a 10 y eliminarlas
#categorias_a_eliminar_nvehiculo = frequencies_nvehiculo[frequencies_nvehiculo < 10].index #NO EJECUTADO
#df = df[~df['NoVe¡culo'].isin(categorias_a_eliminar_nvehiculo)] #NO EJECUTADO

#print(frequencies_nvehiculo)
#%%
df = df.drop(['Linha', 'NoVe¡culo'], axis=1)

#%%
frequencies_totalGiros= df['TotalGiros'].value_counts()

df['TotalGiros'] = pd.to_numeric(df['TotalGiros'], errors='coerce')

df = df.dropna(subset=['TotalGiros'])
df = df[df['TotalGiros'] >= 0]
df['TotalGiros'] = df['TotalGiros'].astype(int)

print(frequencies_totalGiros)
print(df['TotalGiros'])
#%%
frequencies_sentido = df['Sentido'].value_counts()

mapping_sentido = {'Ida': 0, 'Volta': 1}
df['Sentido'] = df['Sentido'].map(mapping_sentido)

print(frequencies_sentido)
print(df['Sentido'])
#%%

print(df)
#%%
#==================ELIMINACAO DE VALORES NULOS O CEROS======

df = df[df['DuracaoSegundos'] != 0]
df = df[df['KmPerc'] != 0]

#df = df[df['TotalGiros'] != 0] #PARA ANALISIS

#%%
# Realizar interpolación de datos usando Cubic Spline Interpolation
#interpolator = CubicSpline(df.index, df['DuracaoSegundos'])
#df['DuracaoSegundos'] = interpolator(df.index)

#%%
# Realizar análisis de detección de anomalías utilizando Local Outlier Factor
X_anomalias = df[['TotalGiros', 'KmPerc', 'DuracaoSegundos']]
outlier_detector = LocalOutlierFactor()
outliers = outlier_detector.fit_predict(X_anomalias)
anomaly_indices = pd.Series(outliers)[outliers == -1].index
valid_indices = anomaly_indices[anomaly_indices.isin(df.index)]
df = df.drop(valid_indices)
#%%

df['DataIni'] = df['DataIni'].astype('int64')
df['DataFim'] = df['DataFim'].astype('int64')
df['HoraIniFormatted'] = df['HoraIniFormatted'].astype('int64')
df['HoraFimFormatted'] = df['HoraFimFormatted'].astype('int64')
#%%
variables = df.columns.tolist()
#%%

scaler = MinMaxScaler()
df['Sentido_normalized'] = scaler.fit_transform(df[['Sentido']])
df['KmPerc_normalized'] = scaler.fit_transform(df[['KmPerc']])
df['TotalGiros_normalized'] = scaler.fit_transform(df[['TotalGiros']])
df['DuracaoSegundos_normalized'] = scaler.fit_transform(df[['DuracaoSegundos']])
df['DataIni_normalized'] = scaler.fit_transform(df[['DataIni']])
df['DataFim_normalized'] = scaler.fit_transform(df[['DataFim']])
df['HoraIniFormatted_normalized'] = scaler.fit_transform(df[['HoraIniFormatted']])
df['HoraFimFormatted_normalized'] = scaler.fit_transform(df[['HoraFimFormatted']])
df['Linha_encoded_normalized'] = scaler.fit_transform(df[['Linha_encoded']])
df['NoVe¡culo_encoded_normalized'] = scaler.fit_transform(df[['NoVe¡culo_encoded']])


print(anomaly_indices)
print(df.dtypes)
print(valid_indices)

print(df.dtypes)

#%%
df = df.drop(['Sentido'], axis=1)
df = df.drop(['TotalGiros', 'KmPerc', 'DuracaoSegundos', 'DataIni', 'DataFim', 'HoraIniFormatted', 'HoraFimFormatted', 'Linha_encoded', 'NoVe¡culo_encoded'], axis=1)
df.to_csv('F:/UFF/DOCTORADO/MACHINE_LEARNING/TRABALHO_FINAL/data/mergeFinal2.csv', index=False)
#%%
import pandas as pd
df = pd.read_csv('F:/UFF/DOCTORADO/MACHINE_LEARNING/TRABALHO_FINAL/data/mergeFinal2.csv')

print(df)

#%%

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
#from libsvm import svm

#%%
print(df.dtypes)

#%%

print(len(df))
df = df.dropna()
print(len(df))

#%%
# Función para imprimir mensaje de inicio
def print_start_message(function_name):
    print(f"Ejecutando {function_name}...")

# Función para imprimir mensaje de finalización
def print_end_message(function_name):
    print(f"{function_name} finalizado.")

# Función para imprimir mensaje de error
def print_error_message(function_name, error_message):
    print(f"Error en {function_name}: {error_message}")

# Tamaños de entrenamiento y prueba
train_sizes = [0.8, 0.7, 0.6, 0.5]
test_sizes = [0.2, 0.3, 0.4, 0.5]

# Array para almacenar los resultados
results = []

class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = Sequential([
            layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=5, batch_size=32, verbose=1)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

for train_size, test_size in zip(train_sizes, test_sizes):
    # Dividir los datos en conjunto de entrenamiento y prueba
    print_start_message("división de datos")
    X_train, X_test, y_train, y_test = train_test_split(df.drop('DuracaoSegundos_normalized', axis=1),
                                                        df['DuracaoSegundos_normalized'],
                                                        test_size=test_size,
                                                        train_size=train_size,
                                                        stratify=df['Sentido_normalized'],
                                                        random_state=42) 
    print_end_message("división de datos")

    # Modelos de regresión
    models = [
        #('Máquinas de Vectores de Soporte', svm.SVR(kernel='linear', C=1.0, epsilon=0.1, verbose=1, )),
        ('Redes Neurais Artificiais', KerasRegressor()),        
        ('Random Forest', RandomForestRegressor(n_estimators=10, random_state=42, verbose=1)),
        ('Árvore de Regressão', DecisionTreeRegressor(random_state=42)),
        ('Regressão Linear', LinearRegression()),
    ]

    # Array para almacenar los resultados de cada tamaño
    size_results = []

    for name, model in models:
        # Validación cruzada con 3 pliegues
        print_start_message(f"validación cruzada para el modelo {name}")
        try:
            scores = cross_val_score(model, X_train, y_train, cv=2, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            mean_rmse = rmse_scores.mean()
            std_rmse = rmse_scores.std()

            # Entrenamiento del modelo con todos los datos de entrenamiento            
            model.fit(X_train, y_train)            

            # Predicciones en el conjunto de prueba
            y_pred = model.predict(X_test)

            # Evaluación del rendimiento en el conjunto de prueba
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Almacenar los resultados en el array
            size_results.append((name, mean_rmse, std_rmse, rmse, r2))

            print_end_message(f"validación cruzada para el modelo {name}")
        except Exception as e:
            print_error_message(f"validación cruzada para el modelo {name}", str(e))

    # Almacenar los resultados de cada tamaño en el array general
    results.append((train_size, test_size, size_results))

# Imprimir los resultados
for train_size, test_size, size_results in results:
    print(f"Tamaño de entrenamiento: {train_size * 100}%")
    print(f"Tamaño de prueba: {test_size * 100}%")
    print("--------------------")
    for name, mean_rmse, std_rmse, rmse, r2 in size_results:
        print(f"Modelo: {name}")
        print(f"Error cuadrático medio (RMSE) promedio: {mean_rmse}")
        print(f"Desviación estándar del RMSE: {std_rmse}")
        print(f"Error cuadrático medio (RMSE) en el conjunto de prueba: {rmse}")
        print(f"Coeficiente de determinación R-cuadrado en el conjunto de prueba: {r2}")
        print("--------------------")
    print("====================")



#%%
#=================REPORT===================
print(results)

data = []
for train_size, test_size, size_results in results:
    for name, mean_rmse, std_rmse, rmse, r2 in size_results:
        data.append([train_size, test_size, name, mean_rmse, std_rmse, rmse, r2])

# Crear un DataFrame con los datos
dfx = pd.DataFrame(data, columns=['Tamaño de entrenamiento', 'Tamaño de prueba', 'Modelo', 'RMSE promedio', 'Desviación estándar del RMSE', 'RMSE en el conjunto de prueba', 'R2 en el conjunto de prueba'])

# Guardar el DataFrame en un archivo CSV
filename = 'F:/UFF/DOCTORADO/MACHINE_LEARNING/TRABALHO_FINAL/data/results.csv'
dfx.to_csv(filename, index=False)
#%%


#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Obtener los nombres de los modelos y las métricas de precisión
model_names = [name for name, _, _, _, _ in results[0][2]]
mean_rmse_values = [[mean_rmse for _, mean_rmse, _, _, _ in size_results] for _, _, size_results in results]
std_rmse_values = [[std_rmse for _, _, std_rmse, _, _ in size_results] for _, _, size_results in results]

# Configurar los datos del gráfico
x = np.arange(len(model_names))
width = 0.2

# Crear la figura y los subplots
fig, ax = plt.subplots()

custom_palette = sns.color_palette( "rocket", n_colors=len(model_names))

# Generar las barras para cada tamaño de entrenamiento
for i, (train_size, test_size, size_results) in enumerate(results):
    mean_rmse_values_i = mean_rmse_values[i]
    std_rmse_values_i = std_rmse_values[i]
    rects = ax.bar(x + i * width, mean_rmse_values_i, width, yerr=std_rmse_values_i, label=f"Train Size: {train_size*100}%", color=custom_palette[i])

# Configurar los ejes y el título
ax.set_ylabel('Erro quadrático médio (RMSE)')
ax.set_title('Comparação de Err do Modelo')
ax.set_xticks(x + (len(train_sizes) / 2) * width)
ax.set_xticklabels(model_names)
ax.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()

#%%
# Obtener los nombres de los modelos y las métricas de precisión
model_names = [name for name, _, _, _, _ in results[0][2]]
r2_values = [[r2*100 for _, _, _, _, r2 in size_results] for _, _, size_results in results]

# Configurar los datos del gráfico
x = np.arange(len(model_names))
width = 0.2

# Crear la figura y los subplots
fig, ax = plt.subplots()


# Generar las barras para cada tamaño de entrenamiento
for i, (train_size, test_size, size_results) in enumerate(results):
    r2_values_i = r2_values[i]
    rects = ax.bar(x + i * width, r2_values_i, width, label=f"Train Size: {train_size*100}%")

    # Mostrar el valor porcentual encima de cada barra
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

# Configurar los ejes y el título
ax.set_ylabel('Coeficiente de determinación R-cuadrado (%)')
ax.set_title('Comparación de R2 en el conjunto de prueba')
ax.set_xticks(x + (len(train_sizes) / 2) * width)
ax.set_xticklabels(model_names)
ax.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()

#%%
# Obtener los nombres de los modelos y las métricas de precisión
model_names = [name for name, _, _, _, _ in results[0][2]]
accuracy_values = [[accuracy for _, _, _, _, accuracy in size_results] for _, _, size_results in results]

# Configurar los datos del gráfico
x = np.arange(len(model_names))
width = 0.2

# Crear la figura y los subplots
fig, ax = plt.subplots()

# Paleta de colores personalizada
custom_palette = sns.color_palette( "rocket", n_colors=len(model_names))

# Generar las barras para cada tamaño de entrenamiento
for i, (train_size, test_size, size_results) in enumerate(results):
    accuracy_values_i = accuracy_values[i]
    rects = ax.bar(x + i * width, accuracy_values_i, width, label=f"Tamanho do treinamento: {train_size*100}%", color=custom_palette[i])

    # Agregar etiquetas de valor encima de cada barra
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2%}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')

# Configurar los ejes y el título
ax.set_ylabel('Coeficiente de determinação R-quadrado (%)')
ax.set_title('Comparação de R2 do Modelo')
ax.set_xticks(x + (len(train_sizes) / 2) * width)
ax.set_xticklabels(model_names)
ax.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()

#%%
# Obtener los nombres de los modelos y las métricas de precisión
model_names = [name for name, _, _, _, _ in results[0][2]]
mean_rmse_values = [[mean_rmse for _, mean_rmse, _, _, _ in size_results] for _, _, size_results in results]
std_rmse_values = [[std_rmse for _, _, std_rmse, _, _ in size_results] for _, _, size_results in results]

# Configurar los datos del gráfico
x = np.arange(len(model_names))
width = 0.2

# Crear la figura y los subplots
fig, ax = plt.subplots()

custom_palette = sns.color_palette("rocket", n_colors=len(model_names))

# Generar las barras para cada tamaño de entrenamiento
for i, (train_size, test_size, size_results) in enumerate(results):
    mean_rmse_values_i = mean_rmse_values[i]
    std_rmse_values_i = std_rmse_values[i]
    rects = ax.bar(x + i * width, mean_rmse_values_i, width, yerr=std_rmse_values_i, label=f"Tamanho do treinamento: {train_size*100}%", color=custom_palette[i])

    # Agregar etiquetas de valor encima de cada barra
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')

# Configurar los ejes y el título
ax.set_ylabel('RMSE médio no treino')
ax.set_title('Comparação do RMSE médio dos Modelos no treino')
ax.set_xticks(x + (len(train_sizes) / 2) * width)
ax.set_xticklabels(model_names)
ax.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()

#%%
# Obtener los nombres de los modelos y los valores de RMSE de prueba
model_names = [name for name, _, _, _, _ in results[0][2]]
rmse_test_values = [[rmse_test for _, _, _, rmse_test, _ in size_results] for _, _, size_results in results]

# Configurar los datos del gráfico
x = range(len(model_names))
width = 0.2

# Crear la figura y los subplots
fig, ax = plt.subplots()

custom_palette = sns.color_palette("rocket", n_colors=len(model_names))

# Generar las barras para cada tamaño de entrenamiento
for i, (train_size, test_size, size_results) in enumerate(results):
    rmse_test_values_i = rmse_test_values[i]
    rects = ax.bar([xi + i * width for xi in x], rmse_test_values_i, width, label=f"Tamanho do treinamento: {train_size*100}%", color=custom_palette[i])

    # Agregar etiquetas de valor encima de cada barra
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')

# Configurar los ejes y el título
ax.set_ylabel('RMSE no conjunto de teste')
ax.set_title('Comparação de RMSE no conjunto de teste')
ax.set_xticks([xi + (len(results) / 2) * width for xi in x])
ax.set_xticklabels(model_names)
ax.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()


#%%
# Crear el encabezado de la tabla
table = r"\begin{tabular}{ccccccc}" + "\n"
table += r"\hline" + "\n"
table += r"Tamaño de entrenamiento & Tamaño de prueba & Modelo & RMSE promedio & Desviación estándar del RMSE & RMSE en el conjunto de prueba & R2 en el conjunto de prueba \\" + "\n"
table += r"\hline" + "\n"

# Crear las filas de la tabla
for train_size, test_size, model_data in results:
    for model_name, mean_rmse, std_rmse, rmse, r2 in model_data:
        table += f"{train_size:.1f} & {test_size:.1f} & {model_name} & {mean_rmse:.4f} & {std_rmse:.4f} & {rmse:.4f} & {r2:.4f} \\\\" + "\n"

# Cerrar la tabla
table += r"\hline" + "\n"
table += r"\end{tabular}"

# Imprimir la tabla en LaTeX
print(table)