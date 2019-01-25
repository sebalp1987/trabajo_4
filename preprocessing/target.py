import pandas as pd
import STRING
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np

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
df['oferta_bonus_simulacion_perc'] = df['oferta_bonus_simulacion_perc'].map(float) * 100
df['oferta_nivel_sinco_perc'] = df['oferta_nivel_sinco_perc'].map(float) * 100

df.loc[(df['oferta_bonus_simulacion'] - df['oferta_nivel_sinco'] == 1) & (df['oferta_nivel_sinco_perc'] < 30),
       'target'] = 1

df.loc[(df['oferta_bonus_simulacion'] - df['oferta_nivel_sinco'] > 1),
       'target'] = 1
print(len(df.index))

# PLOTS
plot.hist(df['oferta_bonus_simulacion_perc'])
plot.show()

plot.hist(df['oferta_nivel_sinco_perc'])
plot.show()

N = len(df.index)
area = (25 * np.random.rand(N)) ** 2
plot.scatter(df['oferta_bonus_simulacion_perc'], df['oferta_nivel_sinco_perc'], s=area, alpha=0.5)
plot.xlabel('Simulation Bonus')
plot.ylabel('SINCO Bonus')
plot.show()

plot.scatter(df['oferta_bonus_simulacion_perc'], df['oferta_bonus_simulacion_perc'] - df['oferta_nivel_sinco_perc'],
             s=area, alpha=0.5)
plot.xlabel('Simulation Bonus')
plot.ylabel('Bonus Diff = Sim - SINCO')
plot.show()

df2 = df[df['oferta_bonus_simulacion_perc'] == 60]
plot.hist(df2['oferta_nivel_sinco_perc'])

plot.show()
df2 = df[df['oferta_nivel_sinco_perc'] < 0]
plot.hist(df2['oferta_bonus_simulacion_perc'])
plot.show()

# CORRELATION
ax = sns.heatmap(df[['oferta_sim_siniestro_5_anio_culpa', 'oferta_sim_anios_asegurado',
                     'oferta_sim_antiguedad_cia_actual', 'oferta_sim_siniestro_1_anio_culpa',
                     'oferta_bonus_simulacion']].corr())
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
df['oferta_id'] = np.random.randint(1, len(df.index), df.shape[0])


putils.output_normal_anormal_new(df)
putils.training_test_valid(df)
