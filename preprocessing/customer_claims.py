import pandas as pd
import STRING
from resources import statistics
import datetime
import numpy as np

pd.options.display.max_columns = 500

# SOURCE FILE
customer_df = pd.read_csv(STRING.file_claim, sep=',', encoding='utf-8', quotechar='"',
                          parse_dates=['poliza_fecha_inicio'])
customer_df = customer_df[customer_df['cliente_fechaini_zurich'] >= '2013-01-01']

# BAD ID
customer_df = customer_df[customer_df['cliente_tipo_doc'].isin(['N', 'R', 'P'])]
'''
customer_df['bad_id'] = pd.Series(0, index=customer_df.index)
print(len(customer_df.index))
customer_df['bad_id'] = customer_df.apply(lambda y: id_conversor(y['cliente_tipo_doc'], y['cliente_nif']), axis=1)
customer_df = customer_df[customer_df['bad_id'] != 1]
print(len(customer_df.index))
del customer_df['bad_id']
'''
# CLIENTE ANTIGUEDAD
customer_df['final_date'] = pd.Series(pd.to_datetime('2017-12-31', format='%Y-%m-%d', errors='coerce'),
                                      index=customer_df.index)
customer_df['cliente_fechaini_zurich'] = pd.to_datetime(customer_df['cliente_fechaini_zurich'], format='%Y-%m-%d',
                                                        errors='coerce')
customer_df = customer_df.dropna(subset=['cliente_fechaini_zurich'])
customer_df['cliente_antiguedad_days'] = pd.Series(
    (customer_df['final_date'] - customer_df['cliente_fechaini_zurich']).dt.days,
    index=customer_df.index)

customer_df['cliente_antiguedad'] = customer_df['cliente_antiguedad_days'] / 365

# SEX VALIDATE
# customer_df['cliente_sexo'] = customer_df['cliente_sexo'].replace('?', -1)
customer_df = customer_df[customer_df['cliente_sexo'] != '?']
customer_df['cliente_sexo'] = customer_df['cliente_sexo'].map(int)
customer_df = customer_df[customer_df['cliente_sexo'].isin([0, 1])]

# FILTER NATIONALITY
# customer_df = customer_df[customer_df['cliente_pais_residencia'] == 'ESPAÑA']

# FILTER CP
customer_df = customer_df[~customer_df['cliente_cp'].astype(str).str.startswith('AD')]  # CP from Andorra
customer_df = customer_df[customer_df['cliente_cp'] != 0]
customer_df = customer_df[customer_df['cliente_cp'] != '0']
# customer_df['cliente_cp'] = customer_df['cliente_cp'].replace('?', -1)
customer_df = customer_df[customer_df['cliente_cp'] != '?']

# REPLACE CLAIMS COST
customer_df.loc[customer_df['cliente_carga_siniestral'] == '?', 'cliente_carga_siniestral'] = 0
customer_df['cliente_carga_siniestral'] = customer_df['cliente_carga_siniestral'].map(float)

# REPLACE BIRTH
customer_df.loc[customer_df['vehiculo_fenacco_conductor1'] == '?', 'vehiculo_fenacco_conductor1'] = customer_df[
    'cliente_fecha_nacimiento']

customer_df.loc[customer_df['cliente_fecha_nacimiento'] == '?', 'cliente_fecha_nacimiento'] = customer_df[
    'vehiculo_fenacco_conductor1']

customer_df = customer_df[customer_df['vehiculo_fenacco_conductor1'] != '?']


# CALCULATE AGE
def calculate_age(birthdate, sep='/'):
    birthdate = datetime.datetime.strptime(birthdate, '%Y' + sep + '%m' + sep + '%d')
    today = datetime.date.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


customer_df['cliente_edad'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fenacco_conductor1']), axis=1)
customer_df.loc[~customer_df['cliente_edad'].between(18, 99, inclusive=True), 'cliente_edad'] = '?'

customer_test = customer_df[['cliente_edad', 'cliente_numero_siniestros', 'cliente_numero_siniestros_auto',
                             'cliente_carga_siniestral']]

customer_test['cliente_edad'] = np.where(customer_test['cliente_edad'] == '?', 1, 0)
customer_test = customer_test.applymap(float)

print(customer_test.groupby(['cliente_edad']).agg({'cliente_numero_siniestros': ['max', 'mean'],
                                                   'cliente_numero_siniestros_auto': 'mean'}))

# We use mean test
statistics.mean_diff_test(customer_df[customer_df['cliente_edad'] != '?'],
                          customer_df[customer_df['cliente_edad'] == '?'], 'cliente_numero_siniestros')

customer_df = customer_df[customer_df['cliente_edad'] != '?']
customer_df['cliente_edad'] = customer_df['cliente_edad'].map(int)

# CLIENTE ANTIGUEDAD
customer_df['cliente_antiguedad'] = customer_df['cliente_antiguedad'].map(float)

customer_df = customer_df[customer_df['cliente_edad'] - customer_df['cliente_antiguedad'] >= 17]

# AGE RANGES
customer_df['cliente_edad_18_30'] = np.where(customer_df['cliente_edad'] <= 30, 1, 0)
customer_df['cliente_edad_30_65'] = np.where(customer_df['cliente_edad'].between(31, 65), 1, 0)
customer_df['cliente_edad_65'] = np.where(customer_df['cliente_edad'] > 65, 1, 0)

# SECOND DRIVER
customer_df.loc[
    customer_df['vehiculo_fenacco_conductor2'] == '?', 'vehiculo_fenacco_conductor2'] = datetime.date.today().strftime(
    '%Y/%m/%d')

customer_df['edad_segundo_conductor'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fenacco_conductor2']),
                                                          axis=1)
customer_df['edad_segundo_conductor_riesgo'] = np.where(customer_df['edad_segundo_conductor'].between(18, 21), 1, 0)
del customer_df['edad_segundo_conductor']

customer_df['edad_conductor_riesgo'] = np.where(customer_df['cliente_edad'].between(18, 21), 1, 0)

# LICENSE YEARS FIRST DRIVER
customer_df['antiguedad_permiso'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fepecon_conductor1']), axis=1)
customer_df['antiguedad_permiso_riesgo'] = np.where(customer_df['antiguedad_permiso'] <= 1, 1, 0)
customer_df.loc[customer_df['antiguedad_permiso'].between(0, 5, inclusive=True), 'antiguedad_permiso_range'] = '[0-5]'
customer_df.loc[customer_df['antiguedad_permiso'].between(6, 10, inclusive=True), 'antiguedad_permiso_range'] = '[6-10]'
customer_df.loc[
    customer_df['antiguedad_permiso'].between(11, 20, inclusive=True), 'antiguedad_permiso_range'] = '[11-20]'
customer_df.loc[
    customer_df['antiguedad_permiso'].between(21, 30, inclusive=True), 'antiguedad_permiso_range'] = '[21-30]'
customer_df.loc[customer_df['antiguedad_permiso'] >= 31, 'antiguedad_permiso_range'] = '[31-inf]'

# LICENSE YEARS SECOND DRIVER
customer_df.loc[customer_df['vehiculo_fepecon_conductor2'] == '?', 'vehiculo_fepecon_conductor2'] = '1900/01/01'
customer_df['antiguedad_permiso_segundo'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fepecon_conductor2']),
                                                              axis=1)
customer_df['antiguedad_permiso_segundo_riesgo'] = np.where(customer_df['antiguedad_permiso_segundo'] <= 1, 1, 0)
customer_df = customer_df.drop(['vehiculo_fepecon_conductor2', 'antiguedad_permiso_segundo'], axis=1)

# VEHICULE USE CODE
customer_df['d_uso_particular'] = np.where(customer_df['vehiculo_uso_desc'].str.contains('PARTICULAR'), 1, 0)
customer_df['d_uso_alquiler'] = np.where(customer_df['vehiculo_uso_desc'].str.contains('ALQUILER'), 1, 0)
customer_df['d_uso_publico'] = np.where(customer_df['vehiculo_uso_desc'].str.contains('PUBLICO'), 1, 0)

# VEHICLE TYPE
tipo_dict = {'TURISMO PARTICULAR': 'car', 'CICLOMOTOR':'motorbike', 'FURGONETA': 'van-track',  'CAMION': 'van-track', 'AUTOCAR': 'autocar',
             'REMOLQUE': 'van-track',  'AGRICO':'agricola',  'INDUSTRIAL': 'industrial',  'TRICICLO': 'motorbike'}

for k, v in tipo_dict.items():
    customer_df['d_tipo_' + v] = pd.Series(0, index=customer_df.index)
for k, v in tipo_dict.items():
    customer_df.loc[customer_df['vehiculo_uso_desc'].str.contains(k), 'd_tipo_' + v] = 1

del tipo_dict

# VEHICLE HEAVY
customer_df['vehiculo_heavy'] = np.where(customer_df['vehiculo_clase_agrupacion_descripcion'].str.contains('>'), 1, 0)

# VEHICLE VALUE
# print(customer_df.vehiculo_valor[~customer_df['vehiculo_valor'].map(np.isreal)])
customer_df = customer_df[customer_df['vehiculo_valor'].map(float) >= 300]

# VEHICLE BONUS
customer_df = customer_df[customer_df['vehiculo_bonus_codigo'] != '?']
customer_df['vehiculo_bonus_codigo'] = customer_df['vehiculo_bonus_codigo'].map(int)

# VEHICLE CATEGORY
cat_dict = {'PRIMERA': '1', 'SEGUNDA': '2', 'TERCERA': '3'}
customer_df['vehiculo_categoria'] = customer_df['vehiculo_categoria'].map(str)
for k, v in cat_dict.items():
    customer_df.loc[customer_df['vehiculo_categoria'].str.contains(k), 'vehiculo_categoria'] = v

# PLATE LICENSE
customer_df = customer_df[customer_df['vehiculo_fecha_mat'].between(1900, 2018, inclusive=True)]
customer_df['antiguedad_vehiculo'] = pd.Series(2018 - customer_df['vehiculo_fecha_mat'], index=customer_df.index)
del customer_df['vehiculo_fecha_mat']

# INTERMEDIARY STATISTICS
# We readjust by the number of days
customer_df['mediador_fecha_alta'] = pd.to_datetime(customer_df['mediador_fecha_alta'], format='%Y-%m-%d',
                                                    errors='coerce')
customer_df['mediador_antiguedad_days'] = pd.Series(
    (customer_df['final_date'] - customer_df['mediador_fecha_alta']).dt.days,
    index=customer_df.index)
customer_df['mediador_numero_siniestros'] = customer_df['mediador_numero_siniestros'] / ((
        customer_df['mediador_antiguedad_days'] + 1) / 365)

customer_df['mediador_numero_polizas'] = customer_df['mediador_numero_polizas'] / ((
        customer_df['mediador_antiguedad_days'] + 1) / 365)
customer_df['mediador_numero_siniestros_AUTO'] = customer_df['mediador_numero_siniestros_AUTO'] / ((
        customer_df['mediador_antiguedad_days'] + 1) / 365)

customer_df['mediador_numero_polizas_AUTO'] = customer_df['mediador_numero_polizas_AUTO'] / ((
        customer_df['mediador_antiguedad_days'] + 1) / 365)
# We calculate the risk
customer_df['mediador_riesgo'] = pd.Series(customer_df.mediador_numero_siniestros / customer_df.mediador_numero_polizas,
                                           index=customer_df.index)
customer_df['mediador_riesgo_auto'] = pd.Series(
    customer_df.mediador_numero_siniestros_AUTO / customer_df.mediador_numero_polizas_AUTO, index=customer_df.index)
customer_df['mediador_share_auto'] = pd.Series(
    customer_df.mediador_numero_polizas_AUTO / customer_df.mediador_numero_polizas, index=customer_df.index)

customer_df = customer_df.drop(['mediador_clase_intermediario', 'mediador_fecha_alta',
                                'mediador_numero_polizas_vigor',
                                'mediador_numero_siniestros_fraude', 'mediador_numero_siniestros_pagados',
                                'mediador_numero_polizas_vigor_AUTO',
                                'mediador_numero_siniestros_fraude_AUTO',
                                'mediador_numero_siniestros_pagados_AUTO'], axis=1)

# ADDRESS
customer_df['address_complete'] = customer_df['cliente_nombre_via'].map(str) + ' ' + customer_df[
    'cliente_numero_hogar'].map(str) + ', ' + customer_df['cliente_cp'].map(str) + ', Spain'

# GROUPED NATIONALITY
country_file = pd.read_csv(STRING.file_country, sep=';', encoding='latin1')
customer_df = pd.merge(customer_df, country_file, left_on='cliente_nacionalidad', right_on='COUNTRY', how='left')
dummy_region = pd.get_dummies(customer_df['REGION'], prefix='cliente_region', dummy_na=True)
customer_df = pd.concat([customer_df, dummy_region], axis=1)
customer_df['cliente_extranjero'] = np.where(customer_df['cliente_nacionalidad'] != 'ESPAÑA', 1, 0)

# VEHICULO VALOR
threshold = customer_df['vehiculo_valor'].quantile(0.99)
customer_df.loc[customer_df['vehiculo_valor'] > threshold, 'vehiculo_valor'] = threshold
customer_df['vehiculo_valor'] = customer_df['vehiculo_valor'].round()
customer_df['vehiculo_valor'] = customer_df['vehiculo_valor'].map(int)
customer_df['vehiculo_valor_range'] = pd.cut(customer_df['vehiculo_valor'],
                                             range(0, customer_df['vehiculo_valor'].max(), 1000), right=True)
customer_df['vehiculo_valor_range'] = customer_df['vehiculo_valor_range'].fillna(
    customer_df['vehiculo_valor_range'].max())

# CLIENTE EDAD
customer_df['cliente_edad'] = customer_df['cliente_edad'].map(int)
customer_df['cliente_edad_range'] = pd.cut(customer_df['cliente_edad'],
                                           range(customer_df['cliente_edad'].min(), customer_df['cliente_edad'].max(),
                                                 5), right=True)
customer_df['cliente_edad_range'] = customer_df['cliente_edad_range'].fillna(customer_df['cliente_edad_range'].max())

# VEHICULO USO
customer_df['veh_uso'] = pd.Series('OTHER', index=customer_df.index)
customer_df.loc[customer_df['d_uso_particular'] == 1, 'veh_uso'] = 'PARTICULAR'
customer_df.loc[customer_df['d_uso_alquiler'] == 1, 'veh_uso'] = 'RENTAL'
customer_df.loc[customer_df['d_uso_publico'] == 1, 'veh_uso'] = 'PUBLIC'


# VEHICULO TIPO

customer_df['veh_tipo'] = pd.Series('OTHER', index=customer_df.index)
customer_df.loc[customer_df['d_tipo_car'] == 1, 'veh_tipo'] = 'CAR'
customer_df.loc[customer_df['d_tipo_motorbike'] == 1, 'veh_tipo'] = 'MOTORBIKE'
customer_df.loc[customer_df['d_tipo_van-track'] == 1, 'veh_tipo'] = 'VAN-TRACK'
customer_df.loc[customer_df['d_tipo_autocar'] == 1, 'veh_tipo'] = 'AUTOCAR'
customer_df.loc[customer_df['d_tipo_agricola'] == 1, 'veh_tipo'] = 'AGRICULTURAL'
customer_df.loc[customer_df['d_tipo_industrial'] == 1, 'veh_tipo'] = 'INDUSTRIAL'

# CLEAN EDAD PERMISO
customer_df = customer_df[customer_df['cliente_edad'] - customer_df['antiguedad_permiso'] >= 17]

# CLEAN CP
customer_df['cliente_cp'] = customer_df['cliente_cp'].str.strip()
customer_df = customer_df.dropna(subset=['cliente_cp'])
customer_df['cliente_cp'] = customer_df['cliente_cp'].map(int)
customer_df = customer_df[customer_df['cliente_cp'] != 0]

# KEEP CUSTOMERS
customer_df = customer_df.sort_values(by=['cliente_poliza'], ascending=[False])
customer_df = customer_df.drop_duplicates(subset=['cliente_codfiliacion', 'vehiculo_modelo_desc'], keep='first')

# CLEAN OUTLIERS
for i in ['cliente_numero_polizas', 'cliente_numero_polizas_auto', 'cliente_numero_siniestros',
          'cliente_numero_siniestros_auto']:
    customer_df[i] = customer_df[i].map(int)
customer_df = customer_df[customer_df['cliente_numero_polizas'] <= 20]
customer_df = customer_df[customer_df['cliente_numero_polizas_auto'] <= customer_df['cliente_numero_polizas']]
customer_df = customer_df[customer_df['cliente_numero_siniestros_auto'] <= customer_df['cliente_numero_siniestros']]
customer_df = customer_df[customer_df['cliente_numero_siniestros'] <= customer_df['cliente_numero_polizas'] * 10]

# Risk Variables
outlier_var = ['cliente_numero_siniestros_auto_culpa',
               'cliente_numero_siniestros',
               'cliente_numero_siniestros_auto',
               'cliente_carga_siniestral']

for i in outlier_var:
    customer_df[i] = customer_df[i] / ((customer_df['cliente_antiguedad_days'] + 1) / 365)

# SHARE AUTO
customer_df['cliente_numero_siniestros_auto_culpa_share'] = customer_df['cliente_numero_siniestros_auto_culpa'] * 100 / \
                                                            customer_df[
                                                                'cliente_numero_siniestros_auto']
customer_df.loc[customer_df['cliente_numero_siniestros_auto'] == 0, 'cliente_numero_siniestros_auto_culpa_share'] = 0
customer_df['cliente_numero_siniestros_auto_culpa_share'] = customer_df[
    'cliente_numero_siniestros_auto_culpa_share'].round()

customer_df.drop_duplicates(subset=['cliente_codfiliacion'], keep='first').describe(include='all').transpose().to_csv(
    STRING.path_db_extra + '\\summary_statistics.csv',
    sep=';', encoding='utf-8')

customer_df.to_csv(STRING.path_db_extra + '\\historical_data.csv', index=False, sep=';', encoding='utf-8')
