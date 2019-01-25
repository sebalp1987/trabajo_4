import pandas as pd
import STRING
import numpy as np
import datetime

from sklearn.cluster import AgglomerativeClustering

from resources.process_utils import calculate_age

pd.set_option('display.max_columns', 100)

# SOURCE FILE
offer_df = pd.read_csv(STRING.file_offer, sep=',', encoding='utf-8', quotechar='"', low_memory=False)

# FILTER RESULTS
offer_df = offer_df[offer_df['oferta_sim_resultado_sinco'].isin(['00'])]
offer_df = offer_df[offer_df['oferta_nivel_sinco'] != '?']
offer_df = offer_df[offer_df['oferta_bonus_simulacion'] != '?']
# offer_df = offer_df[offer_df['oferta_sim_anios_asegurado'] != '?']
# offer_df = offer_df[offer_df['oferta_sim_antiguedad_cia_actual'] != '?']

# DROP DUPLICATES
'''
offer_df = offer_df.sort_values(by=['oferta_poliza', 'oferta_id'], ascending=[True, True]).reset_index(drop=True)
print(offer_df)
offer_df = offer_df.drop_duplicates(subset=['oferta_veh_marca', 'oferta_veh_modelo', 'oferta_veh_version',
                                            'oferta_veh_valor', 'oferta_tomador_cp', 'oferta_conductor_fecha_nac',
                                           'oferta_conductor_fecha_carne'], keep='last')
'''

# INTERMEDIARY FILTER
offer_df = offer_df[offer_df['oferta_cod_intermediario'] != '?']
offer_df['oferta_cod_intermediario'] = offer_df['oferta_cod_intermediario'].map(int)
offer_df = offer_df[offer_df['oferta_cod_intermediario'] != 81083]  # ZURICH PRUEBAS OFERTA WEB

# Filter
offer_df = offer_df[offer_df['oferta_prod_tec'] == 721]
del offer_df['oferta_prod_tec']
offer_df = offer_df[offer_df['oferta_tomador_tipo_pers'] == 'F']
del offer_df['oferta_tomador_tipo_pers']

# SEX VALIDATE
offer_df['oferta_tomador_sexo'] = offer_df['oferta_tomador_sexo'].replace('V', 1)
offer_df['oferta_tomador_sexo'] = offer_df['oferta_tomador_sexo'].replace('M', 0)
offer_df = offer_df[offer_df['oferta_tomador_sexo'].isin([0, 1])]

# CP VALIDATE
offer_df = offer_df[offer_df['oferta_tomador_cp'] != '?']
offer_df = offer_df[offer_df.oferta_tomador_cp.apply(lambda x: x.isnumeric())]
# offer_df['oferta_tomador_cp'] = offer_df['oferta_tomador_cp'].replace('?', -1)
offer_df = offer_df[offer_df.oferta_tomador_cp != 0]
offer_df['oferta_tomador_cp'] = offer_df['oferta_tomador_cp'].map(int)

# STATE POLICY/OFFER
offer_df = offer_df[offer_df['oferta_estado'].isin(['1', '2', '3', 'V', 'P'])]
print(len(offer_df.index))

# 1: FORMALIZADA, 2: VIGOR OFERTA, 3: PENDIENTE OFERTA

# REPLACE BIRTH
offer_df.loc[offer_df['oferta_tomador_fecha_nac'] == 0, 'oferta_tomador_fecha_nac'] = offer_df[
    'oferta_propietario_fecha_nac']

offer_df.loc[offer_df['oferta_tomador_fecha_nac'] == 0, 'oferta_tomador_fecha_nac'] = offer_df[
    'oferta_conductor_fecha_nac']

offer_df.loc[offer_df['oferta_conductor_fecha_nac'] == 0, 'oferta_conductor_fecha_nac'] = offer_df[
    'oferta_tomador_fecha_nac']

offer_df.loc[offer_df['oferta_propietario_fecha_nac'] == 0, 'oferta_propietario_fecha_nac'] = offer_df[
    'oferta_tomador_fecha_nac']

# FILTER BIRTH
offer_df = offer_df[offer_df['oferta_tomador_fecha_nac'] != 0]
offer_df = offer_df[offer_df['oferta_conductor_fecha_nac'] != 0]

# Rare case of birth
offer_df['oferta_conductor_fecha_nac'] = offer_df['oferta_conductor_fecha_nac'].map(str)
offer_df = offer_df[~offer_df['oferta_conductor_fecha_nac'].str.endswith('99')]

# CALCULATE AGE
offer_df['cliente_edad'] = offer_df.apply(lambda y: calculate_age(y['oferta_conductor_fecha_nac']), axis=1)
offer_df = offer_df[offer_df['cliente_edad'].between(18, 99, inclusive=True)]
offer_df['cliente_edad'] = offer_df['cliente_edad'].map(int)
offer_df['cliente_edad_18_30'] = np.where(offer_df['cliente_edad'] <= 30, 1, 0)
offer_df['cliente_edad_30_65'] = np.where(offer_df['cliente_edad'].between(31, 65), 1, 0)
offer_df['cliente_edad_65'] = np.where(offer_df['cliente_edad'] > 65, 1, 0)

# LICENSE YEARS FIRST DRIVER
offer_df['oferta_conductor_fecha_carne'] = offer_df['oferta_conductor_fecha_carne'].map(str)
offer_df = offer_df[offer_df['oferta_conductor_fecha_carne'] != '0']
offer_df['antiguedad_permiso'] = offer_df.apply(lambda y: calculate_age(y['oferta_conductor_fecha_carne']), axis=1)
offer_df['antiguedad_permiso_riesgo'] = np.where(offer_df['antiguedad_permiso'] <= 1, 1, 0)
offer_df = offer_df[offer_df['cliente_edad'] - offer_df['antiguedad_permiso'] >= 17]
offer_df['edad_conductor_riesgo'] = np.where(offer_df['cliente_edad'].between(18, 21), 1, 0)
offer_df['cliente_edad'] = pd.cut(offer_df['cliente_edad'], range(18, offer_df['cliente_edad'].max(), 5), right=True)
offer_df['cliente_edad'] = offer_df['cliente_edad'].fillna(offer_df['cliente_edad'].max())
offer_df.loc[offer_df['antiguedad_permiso'].between(0, 5, inclusive=True), 'antiguedad_permiso_range'] = '[0-5]'
offer_df.loc[offer_df['antiguedad_permiso'].between(6, 10, inclusive=True), 'antiguedad_permiso_range'] = '[6-10]'
offer_df.loc[offer_df['antiguedad_permiso'].between(11, 20, inclusive=True), 'antiguedad_permiso_range'] = '[11-20]'
offer_df.loc[offer_df['antiguedad_permiso'].between(21, 30, inclusive=True), 'antiguedad_permiso_range'] = '[21-30]'
offer_df.loc[offer_df['antiguedad_permiso'] >= 31, 'antiguedad_permiso_range'] = '[31-inf]'

# SECOND DRIVER
offer_df.loc[
    offer_df['oferta_adicional_fecha_nac'].isin(['?', '0', 0]), 'oferta_adicional_fecha_nac'] = \
    datetime.date.today().strftime('%Y%m%d')

offer_df['oferta_adicional_fecha_nac'] = offer_df['oferta_adicional_fecha_nac'].map(str)
offer_df['edad_segundo_conductor'] = offer_df.apply(lambda y: calculate_age(y['oferta_adicional_fecha_nac']),
                                                    axis=1)
offer_df['edad_segundo_conductor_riesgo'] = np.where(offer_df['edad_segundo_conductor'].between(18, 21), 1, 0)
del offer_df['edad_segundo_conductor']

# LICENSE YEARS SECOND DRIVER
offer_df.loc[offer_df['oferta_adicional_fecha_carne'].isin(['?', '0', 0]), 'oferta_adicional_fecha_carne'] = '19000101'
offer_df['oferta_adicional_fecha_carne'] = offer_df['oferta_adicional_fecha_carne'].map(str)
offer_df['antiguedad_permiso_segundo'] = offer_df.apply(lambda y: calculate_age(y['oferta_adicional_fecha_carne']),
                                                        axis=1)
offer_df['antiguedad_permiso_segundo_riesgo'] = np.where(offer_df['antiguedad_permiso_segundo'] <= 1, 1, 0)

# WHO IS WHO
offer_df = offer_df[offer_df['oferta_tom_cond'].isin(['S', 'N'])]
offer_df = offer_df[offer_df['oferta_propietario_tom'].isin(['S', 'N'])]
offer_df = offer_df[offer_df['oferta_propietario_cond'].isin(['S', 'N'])]

offer_df['oferta_tom_cond'] = np.where(offer_df['oferta_tom_cond'] == 'S', 1, 0)
offer_df['oferta_propietario_tom'] = np.where(offer_df['oferta_propietario_tom'] == 'S', 1, 0)
offer_df['oferta_propietario_cond'] = np.where(offer_df['oferta_propietario_cond'] == 'S', 1, 0)

# FILTER DRIVING COUNTRY
offer_df = offer_df[offer_df['oferta_conductor_pais_circu'] == 'ESP']
print(len(offer_df.index))

# GROUPED NATIONALITY
country_file = pd.read_csv(STRING.file_country, sep=';', encoding='latin1')
offer_df = pd.merge(offer_df, country_file[['REGION', 'ISO']], left_on='oferta_conductor_pais_exped_carne',
                    right_on='ISO', how='left')
dummy_region = pd.get_dummies(offer_df['REGION'], prefix='cliente_region', dummy_na=True)
offer_df = pd.concat([offer_df, dummy_region], axis=1)
offer_df['cliente_extranjero'] = np.where(offer_df['oferta_conductor_pais_exped_carne'] != 'ESP', 1, 0)
print(len(offer_df.index))

# VEHICLE TYPE
for i in ['oferta_veh_marca', 'oferta_veh_modelo', 'oferta_veh_version']:
    offer_df[i] = offer_df[i].map(str)
offer_df = offer_df[offer_df['oferta_veh_valor'] != '?']
offer_df['oferta_veh_valor'] = offer_df['oferta_veh_valor'].map(float)
offer_df = offer_df[offer_df['oferta_veh_valor'] >= 300]

offer_df['veh_tipo_agrupacion'] = offer_df['oferta_veh_marca'].map(str) + '-' + offer_df['oferta_veh_modelo'].map(
    str) + '-' + offer_df['oferta_veh_version'].map(str) + '-' + offer_df['oferta_veh_accesorio'].map(str)
car_ranking = offer_df[['veh_tipo_agrupacion',
                        'oferta_veh_plazas', 'oferta_veh_potencia', 'oferta_veh_cilindrada', 'oferta_veh_tara',
                        'oferta_veh_valor']]

car_ranking = car_ranking.groupby(['veh_tipo_agrupacion',
                                   'oferta_veh_plazas', 'oferta_veh_potencia', 'oferta_veh_cilindrada',
                                   'oferta_veh_tara']).agg({'oferta_veh_valor': 'median'})

car_ranking = car_ranking.reset_index(drop=False)

ward = AgglomerativeClustering(n_clusters=10, linkage='ward',
                               connectivity=None)
ward.fit(car_ranking.drop('veh_tipo_agrupacion', axis=1))
labels = ward.labels_
df = pd.DataFrame(labels, columns=['car_ranking'], index=car_ranking.index)
car_ranking = pd.concat([car_ranking, df], axis=1)
del car_ranking['oferta_veh_valor']

offer_df = pd.merge(offer_df, car_ranking, how='left', on=['veh_tipo_agrupacion',
                                                           'oferta_veh_plazas', 'oferta_veh_potencia',
                                                           'oferta_veh_cilindrada',
                                                           'oferta_veh_tara'])

offer_df['oferta_veh_valor'] = offer_df['oferta_veh_valor'].round()
offer_df['oferta_veh_valor'] = offer_df['oferta_veh_valor'].map(int)
offer_df['oferta_veh_valor_unitary'] = offer_df['oferta_veh_valor'].copy()
offer_df.loc[offer_df['oferta_veh_valor'] > 71000, 'oferta_veh_valor'] = 71000
offer_df['oferta_veh_valor'] = pd.cut(offer_df['oferta_veh_valor'], range(0, offer_df['oferta_veh_valor'].max(), 1000),
                                      right=True)
offer_df['oferta_veh_valor'] = offer_df['oferta_veh_valor'].fillna(offer_df['oferta_veh_valor'].max())
offer_df['oferta_veh_valor'] = offer_df['oferta_veh_valor'].map(str)
print(len(offer_df.index))

# VEHICULE USE CODE
offer_df['oferta_veh_uso'] = offer_df['oferta_veh_uso'].map(int)
veh_use = pd.read_csv(STRING.file_veh_use, sep=';', encoding='latin1',
                      dtype={'oferta_veh_uso': int})
offer_df = pd.merge(offer_df, veh_use, how='left', on='oferta_veh_uso')

offer_df['d_uso_particular'] = np.where(offer_df['vehiculo_uso_desc'].str.contains('PARTICULAR'), 1, 0)
offer_df['d_uso_alquiler'] = np.where(offer_df['vehiculo_uso_desc'].str.contains('ALQUILER'), 1, 0)
offer_df['d_uso_publico'] = np.where(offer_df['vehiculo_uso_desc'].str.contains('PUBLICO'), 1, 0)

offer_df['oferta_veh_uso'] = pd.Series('OTHER', index=offer_df.index)
offer_df.loc[offer_df['d_uso_particular'] == 1, 'oferta_veh_uso'] = 'PARTICULAR'
offer_df.loc[offer_df['d_uso_alquiler'] == 1, 'oferta_veh_uso'] = 'ALQUILER'
offer_df.loc[offer_df['d_uso_publico'] == 1, 'oferta_veh_uso'] = 'PUBLIC'
print(len(offer_df.index))

# VEHICLE TYPE
tipo_dict = {'TURISMO PARTICULAR': 'car', 'CICLOMOTOR': 'motorbike', 'FURGONETA': 'van-track', 'CAMION': 'van-track',
             'AUTOCAR': 'autocar',
             'REMOLQUE': 'van-track', 'AGRICO': 'agricola', 'INDUSTRIAL': 'industrial', 'TRICICLO': 'motorbike'}

for k, v in tipo_dict.items():
    offer_df['d_tipo_' + v] = pd.Series(0, index=offer_df.index)
for k, v in tipo_dict.items():
    offer_df.loc[offer_df['vehiculo_uso_desc'].str.contains(k), 'd_tipo_' + v] = 1

del tipo_dict

offer_df['veh_tipo'] = pd.Series('OTHER', index=offer_df.index)
offer_df.loc[offer_df['d_tipo_car'] == 1, 'veh_tipo'] = 'CAR'
offer_df.loc[offer_df['d_tipo_motorbike'] == 1, 'veh_tipo'] = 'MOTORBIKE'
offer_df.loc[offer_df['d_tipo_van-track'] == 1, 'veh_tipo'] = 'VAN-TRACK'
offer_df.loc[offer_df['d_tipo_autocar'] == 1, 'veh_tipo'] = 'AUTOCAR'
offer_df.loc[offer_df['d_tipo_agricola'] == 1, 'veh_tipo'] = 'AGRICULTURAL'
offer_df.loc[offer_df['d_tipo_industrial'] == 1, 'veh_tipo'] = 'INDUSTRIAL'
print(len(offer_df.index))

# VEHICLE CLASE
offer_df['oferta_veh_tipo'] = offer_df['oferta_veh_tipo'].map(int)
veh_use = pd.read_csv(STRING.file_veh_clase, sep=';', encoding='latin1',
                      dtype={'oferta_veh_tipo': int})
offer_df = pd.merge(offer_df, veh_use, how='left', on='oferta_veh_tipo')
print(len(offer_df.index))

# VEHICLE MARCA
offer_df['oferta_veh_marca'] = offer_df['oferta_veh_marca'].map(str)
offer_df['oferta_veh_marca'] = offer_df['oferta_veh_marca'].str.lstrip("0")
veh_use = pd.read_csv(STRING.file_veh_marca, sep=';', encoding='latin1',
                      dtype={'oferta_veh_marca': str})
print(veh_use)
offer_df = pd.merge(offer_df, veh_use, how='left', on='oferta_veh_marca')
print(offer_df['oferta_veh_marca'])
print(len(offer_df.index))

# VEHICLE HEAVY
offer_df['vehiculo_heavy'] = np.where(offer_df['vehiculo_clase_agrupacion_descripcion'].str.contains('>'),
                                      1, 0)

offer_df['oferta_veh_tara'] = offer_df['oferta_veh_tara'].replace('?', 0)
offer_df['oferta_veh_tara'] = offer_df['oferta_veh_tara'].map(int)
offer_df['oferta_veh_tara'] = np.where(offer_df['oferta_veh_tara'] >= 3500,
                                       1, 0)

offer_df['oferta_veh_puertos'] = np.where(offer_df['oferta_veh_puertos'] == 'S', 1, 0)
print(len(offer_df.index))

# PLATE LICENSE
offer_df['oferta_fecha_matricula'] = offer_df['oferta_fecha_matricula'].map(int)
offer_df = offer_df[offer_df['oferta_fecha_matricula'].between(1900, 2018, inclusive=True)]
offer_df['antiguedad_vehiculo'] = pd.Series(2018 - offer_df['oferta_fecha_matricula'], index=offer_df.index)
del offer_df['oferta_fecha_matricula']

print(len(offer_df.index))
# MATCH BONUS
bonus_df = pd.read_csv(STRING.file_bonus, sep=';', encoding='latin1')
bonus_df = bonus_df.set_index('Code')['BONUS'].to_dict()

for i in ['oferta_nivel_sinco', 'oferta_bonus_simulacion']:

    offer_df[i + '_perc'] = pd.Series(0, index=offer_df.index)
    offer_df[i] = offer_df[i].map(int)
    for k, v in bonus_df.items():
        offer_df.loc[offer_df[i] == int(k), i + '_perc'] = v

# OFERTA ADICIONAL RIESGO
offer_df['oferta_adicional_riesgo'] = np.where(offer_df['oferta_adicional_riesgo'] == 'S', 1, 0)
print(len(offer_df.index))

# PROCESS SIMULATION VARIABLES
# Integer Var
int_var = ['oferta_sim_siniestro_5_anio_culpa', 'oferta_sim_anios_asegurado',
           'oferta_sim_antiguedad_cia_actual', 'oferta_sim_siniestro_1_anio_culpa']
for i in int_var:
    offer_df[i] = offer_df[i].replace('?', 0)
    offer_df[i] = offer_df[i].map(int)
print(len(offer_df.index))
# Before companies
dummy_var = pd.get_dummies(offer_df['oferta_sim_cia_actual'], dummy_na=True, prefix='cia_anterior')
offer_df = pd.concat([offer_df, dummy_var], axis=1)
print(len(offer_df.index))

# based on SINCO info we cluster the companies
cia_risk = offer_df[['oferta_sim_cia_actual', 'oferta_nivel_sinco_perc', 'oferta_bonus_simulacion_perc']]
cia_risk['oferta_nivel_sinco_perc'] = cia_risk['oferta_nivel_sinco_perc'].map(float) * 100
cia_risk['oferta_bonus_simulacion_perc'] = cia_risk['oferta_bonus_simulacion_perc'].map(float) * 100
cia_risk['oferta_diff'] = cia_risk['oferta_bonus_simulacion_perc'] - cia_risk['oferta_nivel_sinco_perc']
cia_risk = cia_risk[cia_risk['oferta_sim_cia_actual'] != '?']
cia_risk = cia_risk.groupby(['oferta_sim_cia_actual']).agg({'oferta_nivel_sinco_perc': 'median',
                                                            'oferta_sim_cia_actual': 'count'
                                                            })
print(len(offer_df.index))
cia_risk = cia_risk[cia_risk['oferta_sim_cia_actual'] > 5.0]
del cia_risk['oferta_sim_cia_actual']

cia_risk = cia_risk.reset_index(drop=False)

offer_df = offer_df.rename(columns={'cia_anterior_000': 'd_sin_cia'})
offer_df['risk_cia'] = offer_df['cia_anterior_039'] + offer_df['cia_anterior_040']  # Mutua Madrileña

offer_df = offer_df[offer_df.columns.drop(list(df.filter(regex='cia_anterior_')))]

print('CIA CODE CLUSTER')
# cluster_analysis.expl_hopkins(cia_risk.drop(['oferta_sim_cia_actual'], axis=1), num_iters=1000)
# cluster_analysis.cluster_internal_validation(cia_risk.drop(['oferta_sim_cia_actual'], axis=1), n_clusters=10)
# cluster_analysis.silhouette_coef(cia_risk.drop(['oferta_sim_cia_actual'], axis=1).values,
# range_n_clusters=range(7, 8, 1))
'''
cluster_analysis.kmeans_plus_plus(cia_risk, k=10, n_init=42, max_iter=500, drop='oferta_sim_cia_actual',
                                            show_plot=False,
                                            file_name='cia_risk')
'''

deL_col = ['oferta_sim_resultado_sinco']
offer_df.to_csv(STRING.processed_offer_before, sep=';', index=False)
# CLUSTERS ###########################################################################################################
cp_risk = pd.read_csv(STRING.cluster_risk, sep=';', dtype={'cliente_cp': int})
intm_risk = pd.read_csv(STRING.cluster_intm, sep=';', dtype={'mediador_cod_intermediario': int})
veh_risk = pd.read_csv(STRING.cluster_veh, sep=';',
                       dtype={'vehiculo_valor_range': str, 'vehiculo_categoria': int, 'vehiculo_marca_desc': str,
                              'veh_uso': str, 'veh_tipo': str, 'vehiculo_heavy': int, 'antiguedad_vehiculo': int})
customer_risk = pd.read_csv(STRING.cluster_customer, sep=';',
                            dtype={'cliente_edad_range': str, 'antiguedad_permiso_range': str,
                                   'antiguedad_permiso_riesgo': int, 'edad_segundo_conductor_riesgo': int,
                                   'cliente_sexo': int, 'antiguedad_permiso_segundo_riesgo': int,
                                   'cliente_extranjero': int,
                                   'edad_conductor_riesgo': int})
print(len(offer_df.index))
offer_df = pd.merge(offer_df, cp_risk[['cliente_cp', 'clusters_zip code_risk']], left_on='oferta_tomador_cp',
                    right_on='cliente_cp', how='left')
offer_df['clusters_zip code_risk'] = offer_df['clusters_zip code_risk'].fillna(-1)

print(len(offer_df.index))
offer_df = pd.merge(offer_df, intm_risk[['mediador_cod_intermediario', 'clusters_intermediary_risk']],
                    left_on='oferta_cod_intermediario',
                    right_on='mediador_cod_intermediario', how='left')
offer_df['clusters_intermediary_risk'] = offer_df['clusters_intermediary_risk'].fillna(-1)

offer_df['oferta_veh_categoria'] = offer_df['oferta_veh_categoria'].map(int)
offer_df['vehiculo_heavy'] = offer_df['vehiculo_heavy'].map(int)
offer_df['antiguedad_vehiculo'] = offer_df['antiguedad_vehiculo'].map(int)
offer_df['vehiculo_marca_desc'] = offer_df['vehiculo_marca_desc'].map(str)
print(len(offer_df.index))
offer_df = pd.merge(offer_df, veh_risk[['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc', 'veh_uso',
                                        'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo', 'clusters_vehicle_risk']],
                    left_on=['oferta_veh_valor', 'oferta_veh_categoria', 'vehiculo_marca_desc', 'oferta_veh_uso',
                             'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'],
                    right_on=['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc', 'veh_uso',
                              'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'], how='inner')
print(len(offer_df.index))

offer_df['cliente_edad'] = offer_df['cliente_edad'].map(str)
offer_df['antiguedad_permiso_range'] = offer_df['antiguedad_permiso_range'].map(str)
offer_df['antiguedad_permiso_riesgo'] = offer_df['antiguedad_permiso_riesgo'].map(int)
offer_df['edad_segundo_conductor_riesgo'] = offer_df['edad_segundo_conductor_riesgo'].map(int)
offer_df['oferta_tomador_sexo'] = offer_df['oferta_tomador_sexo'].map(int)
offer_df['antiguedad_permiso_segundo_riesgo'] = offer_df['antiguedad_permiso_segundo_riesgo'].map(int)
offer_df['cliente_extranjero'] = offer_df['cliente_extranjero'].map(int)
offer_df['edad_conductor_riesgo'] = offer_df['edad_conductor_riesgo'].map(int)

offer_df = pd.merge(offer_df, customer_risk[
    ['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo',
     'cliente_sexo', 'antiguedad_permiso_segundo_riesgo', 'cliente_extranjero', 'edad_conductor_riesgo',
     'clusters_customer_risk']],
                    left_on=['cliente_edad', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo',
                             'edad_segundo_conductor_riesgo', 'oferta_tomador_sexo',
                             'antiguedad_permiso_segundo_riesgo', 'cliente_extranjero', 'edad_conductor_riesgo'],
                    right_on=['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo',
                              'edad_segundo_conductor_riesgo',
                              'cliente_sexo', 'antiguedad_permiso_segundo_riesgo', 'cliente_extranjero',
                              'edad_conductor_riesgo'], how='inner')
print(len(offer_df.index))
remove_variables = ['cliente_cp', 'oferta_tomador_cp', 'oferta_cod_intermediario', 'mediador_cod_intermediario',
                    'vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc', 'veh_uso',
                    'cliente_edad_range', 'cliente_sexo']

deL_col += remove_variables

offer_df = offer_df.drop(deL_col, axis=1)
offer_df = offer_df.replace('?', 0)
offer_df = offer_df.dropna()
print(len(offer_df.index))

drop_variables = ['oferta_estado', 'oferta_inicio', 'oferta_prod_com', 'oferta_prod_nombre', 'oferta_tomador_tipo_doc',
                  'oferta_tomador_fecha_nac', 'oferta_tomador_lugar_nac', 'oferta_idioma', 'oferta_propietario_fecha_nac',
                  'oferta_propietario_fecha_carne', 'oferta_conductor_fecha_nac', 'oferta_conductor_fecha_carne',
                  'oferta_conductor_cp_circ', 'oferta_conductor_pais_exped_carne', 'oferta_conductor_pais_circu',
                  'oferta_adicional_fecha_nac', 'oferta_adicional_fecha_carne', 'oferta_veh_pais_matricula',
                  'oferta_veh_tipo_matricula', 'oferta_veh_marca', 'oferta_veh_modelo', 'oferta_veh_version',
                  'oferta_veh_accesorio', 'oferta_veh_uso', 'oferta_veh_tipo', 'oferta_veh_clase',
                  'oferta_veh_grupo_tarifa', 'oferta_veh_pma', 'oferta_veh_plazas', 'oferta_veh_potencia',
                  'oferta_veh_motor', 'oferta_veh_cilindrada', 'oferta_veh_remolque', 'oferta_veh_ambito',
                  'audit_oferta_version', 'audit_codigo_proc_oferta', 'REGION', 'ISO', 'veh_tipo_agrupacion',
                  'vehiculo_uso_desc', 'veh_tipo', 'vehiculo_clase_agrupacion_descripcion']

offer_df = offer_df.drop(drop_variables, axis=1)

offer_df.describe(include='all').transpose().to_csv(STRING.summary_statistics_offers, sep=';', encoding='utf-8')
offer_df.to_csv(STRING.processed_offer, sep=';', index=False)

