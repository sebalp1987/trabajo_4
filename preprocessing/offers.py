import pandas as pd
import STRING
import numpy as np
import datetime
from sklearn.cluster import AgglomerativeClustering
from resources.process_utils import calculate_age

# SOURCE FILE
offer_df = pd.read_csv(STRING.path_db + STRING.file_offer, sep=',', encoding='utf-8', quotechar='"')

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
offer_df['edad_segundo_conductor_riesgo'] = np.where(offer_df['edad_segundo_conductor'].between(18, 25), 1, 0)
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
country_file = pd.read_csv(STRING.path_db_aux + STRING.file_country, sep=';', encoding='latin1')
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
car_ranking.to_csv(STRING.path_db_aux + '\\cluster_ward_car.csv', index=False, sep=';')
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
print(len(offer_df.index))
# VEHICULE USE CODE
offer_df['oferta_veh_uso'] = offer_df['oferta_veh_uso'].map(int)
veh_use = pd.read_csv(STRING.path_db_aux + STRING.file_veh_use, sep=';', encoding='latin1',
                      dtype={'oferta_veh_uso': int})
offer_df = pd.merge(offer_df, veh_use, how='left', on='oferta_veh_uso')

offer_df['d_uso_particular'] = np.where(offer_df['vehiculo_uso_desc'].str.contains('PARTICULAR'), 1, 0)
offer_df['d_uso_alquiler'] = np.where(offer_df['vehiculo_uso_desc'].str.contains('ALQUILER'), 1, 0)
offer_df['veh_uso'] = pd.Series('OTRO', index=offer_df.index)
offer_df.loc[offer_df['d_uso_particular'] == 1, 'veh_uso'] = 'PARTICULAR'
offer_df.loc[offer_df['d_uso_alquiler'] == 1, 'veh_uso'] = 'ALQUILER'
print(len(offer_df.index))
# VEHICLE TYPE
tipo_dict = {'ciclomotor': 'PARTICULAR', 'furgoneta': 'FURGONETA', 'camion': 'CAMION', 'autocar': 'AUTOCAR',
             'remolque': 'REMOLQUE', 'agricola': 'AGRICO', 'industrial': 'INDUSTRIAL', 'triciclo': 'TRICICLO'}

for k, v in tipo_dict.items():
    offer_df['d_tipo_' + k] = np.where(offer_df['vehiculo_uso_desc'].str.contains(v), 1, 0)
del tipo_dict

offer_df['veh_tipo'] = pd.Series('OTRO', index=offer_df.index)
offer_df.loc[offer_df['d_tipo_ciclomotor'] == 1, 'veh_tipo'] = 'CICLOMOTOR'
offer_df.loc[offer_df['d_tipo_furgoneta'] == 1, 'veh_tipo'] = 'FURGONETA'
offer_df.loc[offer_df['d_tipo_camion'] == 1, 'veh_tipo'] = 'CAMION'
offer_df.loc[offer_df['d_tipo_autocar'] == 1, 'veh_tipo'] = 'AUTOCAR'
offer_df.loc[offer_df['d_tipo_remolque'] == 1, 'veh_tipo'] = 'REMOLQUE'
offer_df.loc[offer_df['d_tipo_agricola'] == 1, 'veh_tipo'] = 'AGRICOLA'
offer_df.loc[offer_df['d_tipo_industrial'] == 1, 'veh_tipo'] = 'INDUSTRIAL'
offer_df.loc[offer_df['d_tipo_triciclo'] == 1, 'veh_tipo'] = 'TRICICLO'
print(len(offer_df.index))
# VEHICLE CLASE
offer_df['oferta_veh_tipo'] = offer_df['oferta_veh_tipo'].map(int)
veh_use = pd.read_csv(STRING.path_db_aux + STRING.file_veh_clase, sep=';', encoding='latin1',
                      dtype={'oferta_veh_tipo': int})
offer_df = pd.merge(offer_df, veh_use, how='left', on='oferta_veh_tipo')
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
bonus_df = pd.read_csv(STRING.path_db_aux + STRING.file_bonus, sep=';', encoding='latin1')
bonus_df = bonus_df.set_index('Code')['BONUS'].to_dict()

for i in ['oferta_nivel_sinco', 'oferta_bonus_simulacion']:

    offer_df[i + '_perc'] = pd.Series(0, index=offer_df.index)
    offer_df[i] = offer_df[i].map(int)
    for k, v in bonus_df.items():
        offer_df.loc[offer_df[i] == int(k), i + '_perc'] = v

# OFERTA ADICIONAL RIESGO
offer_df['oferta_adicional_riesgo'] = np.where(offer_df['oferta_adicional_riesgo']=='S', 1, 0)
print(len(offer_df.index))
# KEEP VARIABLES
region = offer_df.columns.values.tolist()
region = [var for var in region if var.startswith('cliente_region')]
keep_var = ['oferta_id', 'oferta_poliza', 'oferta_tomador_cp', 'oferta_tomador_sexo',
            'oferta_cod_intermediario',
            'oferta_tom_cond', 'oferta_propietario_tom', 'oferta_propietario_cond', 'oferta_veh_valor',
            'oferta_veh_tara', 'oferta_veh_categoria',
            'oferta_veh_puertos', 'oferta_poliza', 'cliente_edad',
            'antiguedad_permiso', 'antiguedad_permiso_range',
            'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo',
            'REGION',
            'cliente_extranjero', 'car_ranking',
            'd_uso_particular', 'd_uso_alquiler', 'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion',
            'd_tipo_autocar', 'd_tipo_remolque', 'd_tipo_agricola', 'd_tipo_industrial',
            'd_tipo_triciclo', 'vehiculo_heavy', 'oferta_sim_cia_actual', 'oferta_sim_siniestro_5_anio_culpa',
            'oferta_sim_anios_asegurado', 'oferta_sim_antiguedad_cia_actual',
            'oferta_sim_siniestro_1_anio_culpa', 'oferta_sim_resultado_sinco',
            'oferta_nivel_sinco', 'oferta_nivel_sinco_perc', 'oferta_bonus_simulacion',
            'oferta_bonus_simulacion_perc', 'veh_uso', 'veh_tipo', 'oferta_veh_valor_unitary',
            'cliente_edad_18_30', 'cliente_edad_30_65', 'cliente_edad_65', 'antiguedad_vehiculo', 'oferta_sim_bonus_rc',
            'oferta_sim_bonus_danio', 'oferta_adicional_riesgo', 'oferta_veh_marca', 'oferta_veh_modelo', 'oferta_veh_version',
            'oferta_veh_accesorio', 'oferta_veh_uso', 'oferta_veh_tipo', 'oferta_veh_grupo_tarifa',
            'oferta_veh_plazas', 'oferta_veh_potencia', 'oferta_veh_tara', 'oferta_veh_cilindrada'
            ] + region
offer_df = offer_df[keep_var]
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

print('CIA CODE CLUSTER')
print('total cias ', len(cia_risk.index))
# cluster_analysis.expl_hopkins(cia_risk.drop(['oferta_sim_cia_actual'], axis=1), num_iters=1000)
#cluster_analysis.cluster_internal_validation(cia_risk.drop(['oferta_sim_cia_actual'], axis=1), n_clusters=10)
# cluster_analysis.silhouette_coef(cia_risk.drop(['oferta_sim_cia_actual'], axis=1).values, range_n_clusters=range(7, 8, 1))
'''
cluster_analysis.kmeans_plus_plus(cia_risk, k=10, n_init=42, max_iter=500, drop='oferta_sim_cia_actual',
                                            show_plot=False,
                                            file_name='cia_risk')

'''
deL_col = ['oferta_sim_resultado_sinco']

offer_df = offer_df.drop(deL_col, axis=1)
offer_df = offer_df.replace('?', 0)
offer_df = offer_df.dropna()
###################################################################################
print(len(offer_df.index))
offer_df.to_csv(STRING.path_db_aux + '\\oferta_processed.csv', sep=';', index=False)


