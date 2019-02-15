import pandas as pd
import STRING
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np
import random

import resources.process_utils as putils

pd.options.display.max_columns = 500
np.random.seed(19680801)
sns.set()

df = pd.read_csv(STRING.processed_offer, sep=';')
print(len(df.index))

# ANOMALY DEFINITION
'''
3.	Si el BONUS Simulado es un nivel mejor que el BONUS de consulta SINCO  ->  el BONUS de Emisión será el Simulado, 
siempre que el BONUS de SINCO sea 30% o mejor (KT3F3OT).

4.	Si el BONUS Simulado es más de un nivel mejor que el BONUS de consulta SINCO -> el BONUS de Emisión será el Simulado.
'''
df['target'] = pd.Series(0, index=df.index)
df['oferta_bonus_simulacion_perc'] = (df['oferta_bonus_simulacion_perc'].map(float) * 100).map(int)
df['oferta_nivel_sinco_perc'] = (df['oferta_nivel_sinco_perc'].map(float) * 100).map(int)
print(df[['oferta_bonus_simulacion_perc', 'oferta_nivel_sinco_perc']])
df = df[df['oferta_bonus_simulacion_perc'] >= 0].reset_index(drop=True)

df.loc[(df['oferta_bonus_simulacion'] - df['oferta_nivel_sinco'] == 1) & (df['oferta_nivel_sinco_perc'] < 30),
       'target'] = 1

df.loc[(df['oferta_bonus_simulacion'] - df['oferta_nivel_sinco'] > 1),
       'target'] = 1

print(len(df.index))

# PLOTS
plot.subplot(2, 1, 1)
plot.hist(df['oferta_bonus_simulacion_perc'])
plot.ylabel('Simulated Bonus')

plot.subplot(2, 1, 2)
plot.hist(df['oferta_nivel_sinco_perc'])
plot.xlabel('Percentage')
plot.ylabel('Adjusted Bonus')
plot.show()

N = len(df.index)
area = (25 * np.random.rand(N)) ** 2
plot.scatter(df['oferta_bonus_simulacion_perc'], df['oferta_nivel_sinco_perc'], s=area, alpha=0.5)
plot.xlabel('Simulated Bonus')
plot.ylabel('Adjusted Bonus')
plot.show()

# FINAL CLEANING
del_var = ['oferta_veh_valor', 'cliente_edad', 'antiguedad_permiso_range']
df = df.drop(del_var, axis=1)
df = df.dropna()
print(len(df.index))
df.to_csv(STRING.processed_target, sep=';', index=False)

df = df.drop(['oferta_nivel_sinco_perc', 'oferta_nivel_sinco', 'oferta_poliza',
              'oferta_sim_bonus_rc',
              'oferta_sim_bonus_danio'
              ], axis=1)

df = df.drop(['oferta_bonus_simulacion'], axis=1)
df = df.drop_duplicates(subset=['oferta_id'])
df['oferta_id'] = pd.Series(random.sample(range(0, len(df.index), 1), len(df.index)), index=df.index)

keep_var = ['oferta_id', 'oferta_tomador_sexo', 'oferta_tom_cond', 'oferta_propietario_tom', 'oferta_propietario_cond',
            'oferta_veh_tara', 'oferta_veh_categoria', 'oferta_veh_puertos', 'antiguedad_permiso',
            'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo', 'cliente_extranjero', 'car_ranking',
            'd_uso_particular', 'd_uso_alquiler',
            'd_uso_publico',
            'd_tipo_car', 'd_tipo_motorbike', 'd_tipo_van-track', 'd_tipo_autocar', 'd_tipo_agricola',
            'd_tipo_industrial',
            'vehiculo_heavy','oferta_sim_siniestro_5_anio_culpa',
            'oferta_sim_anios_asegurado', 'oferta_sim_antiguedad_cia_actual', 'oferta_sim_siniestro_1_anio_culpa',
            'oferta_bonus_simulacion_perc', 'oferta_veh_valor_unitary', 'cliente_edad_18_30', 'cliente_edad_30_65',
            'cliente_edad_65', 'antiguedad_vehiculo', 'oferta_adicional_riesgo',  'oferta_veh_plazas',
            'oferta_veh_potencia', 'oferta_veh_cilindrada',
            'cliente_region_AFRICAARABE', 'cliente_region_AFRICASUBSHARIANA', 'cliente_region_AMERICADELSUR',
            'cliente_region_ASIAORIENTAL', 'cliente_region_EUROPACENTRAL', 'cliente_region_EUROPADELNORTE',
            'cliente_region_EUROPADELSUR', 'cliente_region_EUROPAOCCIDENTAL', 'cliente_region_EUROPAORIENTAL',
            'cliente_region_MARCARIBE', 'cliente_region_OCEANIA', 'cliente_region_nan', 'd_sin_cia',
            'clusters_zip code_risk',
            'clusters_intermediary_risk', 'clusters_customer_risk', 'clusters_vehicle_risk', 'target']

extras = ['oferta_veh_marca',
            'oferta_veh_modelo', 'oferta_veh_version', 'oferta_veh_accesorio', 'oferta_veh_tipo',
            'oferta_veh_grupo_tarifa',  'oferta_sim_cia_actual']
df = df[keep_var]
print(df.shape)
putils.output_normal_anormal_new(df)
putils.training_test_valid(df)

count_classes = pd.value_counts(df['target'], sort=True)
count_classes.plot(kind='bar', rot=0)
plot.xticks(range(2), ['Normal', 'Abnormal'])
plot.xlabel('Class')
plot.ylabel('Frequency')
plot.show()
