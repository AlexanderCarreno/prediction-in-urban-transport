import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

print('\n===================================================')
print('=============== LASSO Regresion ================== ')
print('===================================================\n')


# Read urban transport data from Nuevo Leon
df_transport = pd.read_csv('raw/etup_mensual_tr_cifra_1986_2024.csv')
df_entities = pd.read_csv('raw/tc_entidad.csv')
df_municipality = pd.read_csv('raw/tc_municipio.csv')

# Merge dataframes
df_te = df_transport.merge(df_entities, on='ID_ENTIDAD', how='left')
df_tem = df_te.merge(df_municipality, left_on=['ID_ENTIDAD','ID_MUNICIPIO'], right_on=['ID_ENTIDAD','ID_MUNICIPIO'], how='left')

# Filter columns and rows in order to analyse just Nuevo Leon
df_tem = df_tem.filter(items=['ANIO','ID_MES','TRANSPORTE','VARIABLE','VALOR','NOM_ENTIDAD','NOM_MUNICIPIO'])

# Data after 2020 because in February 2021 the subway line number 3 was open
df_tem = df_tem[(df_tem['NOM_ENTIDAD'] == 'Nuevo León') & (df_tem['VARIABLE'] == 'Ingresos por pasaje') & (df_tem['ANIO'] > 2020)]
df_tem = df_tem.filter(items=['ANIO','ID_MES','VALOR'])
df_tem['fecha'] = df_tem['ANIO'] + df_tem['ID_MES'] / 12

# Visual check
# print(df_tem)
# df_tem.to_csv('data2.txt', sep='\t', index=False)


x = df_tem[['fecha']]
y = df_tem['VALOR']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelo = Lasso(alpha=1.0) 
modelo.fit(x_train.values, y_train.values)

y_pred = modelo.predict(x_test.values)

mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')

print('Coeficiente del modelo:', modelo.coef_)
print('Intersección del modelo:', modelo.intercept_)

nuevo_año = 2024
nuevo_mes = 4
nueva_fecha = nuevo_año + nuevo_mes / 12
prediccion_uso = modelo.predict([[nueva_fecha]])

res = ('{:,}'.format(prediccion_uso[0])) 

print(f'Predicción del uso del metro en Nuevo Leon el {nuevo_año}-{nuevo_mes} es: {str(res)}') # Predicción del uso del metro en Nuevo Leon el 2024-4 es: 72,761,068.86664581
