import pandas as pd
import STRING
from resources import cluster_analysis

customer_df = pd.read_csv(STRING.path_db_extra + '\\historical_data.csv', sep=';', encoding='latin1')

x = customer_df[
    ['cliente_codfiliacion', 'cliente_poliza', 'vehiculo_modelo_desc',
     'cliente_numero_siniestros', 'cliente_carga_siniestral', 'cliente_numero_siniestros_auto',
     'cliente_sexo', 'vehiculo_valor', 'cliente_edad_18_30', 'cliente_edad_30_65', 'cliente_edad_65',
     'edad_segundo_conductor_riesgo', 'antiguedad_permiso', 'antiguedad_permiso_riesgo',
     'antiguedad_permiso_segundo_riesgo',
     'd_uso_particular', 'd_uso_alquiler', 'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion', 'd_tipo_autocar',
     'd_tipo_remolque', 'd_tipo_agricola',
     'd_tipo_industrial', 'd_tipo_triciclo', 'antiguedad_vehiculo', 'cliente_extranjero', 'cliente_edad',
     'vehiculo_categoria',
     'vehiculo_heavy', 'mediador_riesgo', 'mediador_riesgo_auto', 'mediador_share_auto', 'REGION',
     'cliente_numero_siniestros_auto_culpa', 'antiguedad_permiso_range', 'cliente_numero_polizas_auto'
     ]]

# x = x.sort_values(by=['cliente_poliza'], ascending=[False])
# x = x.drop_duplicates(subset=['cliente_codfiliacion', 'vehiculo_modelo_desc'], keep='first')

# RISK CLUSTERING BY OBJECT
threshold = x['vehiculo_valor'].quantile(0.99)
print(threshold)
x.loc[x['vehiculo_valor'] > threshold, 'vehiculo_valor'] = threshold
x['vehiculo_valor'] = x['vehiculo_valor'].round()
x['vehiculo_valor'] = x['vehiculo_valor'].map(int)
x['vehiculo_valor'] = pd.cut(x['vehiculo_valor'], range(0, x['vehiculo_valor'].max(), 1000), right=True)
x['vehiculo_valor'] = x['vehiculo_valor'].fillna(x['vehiculo_valor'].max())

x['veh_uso'] = pd.Series('OTRO', index=x.index)
x.loc[x['d_uso_particular'] == 1, 'veh_uso'] = 'PARTICULAR'
x.loc[x['d_uso_alquiler'] == 1, 'veh_uso'] = 'ALQUILER'

x['veh_tipo'] = pd.Series('OTRO', index=x.index)
x.loc[x['d_tipo_ciclomotor'] == 1, 'veh_tipo'] = 'CICLOMOTOR'
x.loc[x['d_tipo_furgoneta'] == 1, 'veh_tipo'] = 'FURGONETA'
x.loc[x['d_tipo_camion'] == 1, 'veh_tipo'] = 'CAMION'
x.loc[x['d_tipo_autocar'] == 1, 'veh_tipo'] = 'AUTOCAR'
x.loc[x['d_tipo_remolque'] == 1, 'veh_tipo'] = 'REMOLQUE'
x.loc[x['d_tipo_agricola'] == 1, 'veh_tipo'] = 'AGRICOLA'
x.loc[x['d_tipo_industrial'] == 1, 'veh_tipo'] = 'INDUSTRIAL'
x.loc[x['d_tipo_triciclo'] == 1, 'veh_tipo'] = 'TRICICLO'

x['counter'] = pd.Series(1, index=x.index)

x_object = x.groupby(
    ['vehiculo_valor', 'vehiculo_categoria',
     'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo']).agg(
    {
        'cliente_numero_siniestros_auto': ['mean'], 'cliente_numero_polizas_auto': ['mean'], 'counter': 'count'})

x_object = x_object[x_object[('counter', 'count')] > 5.0]
del x_object['counter']
print('VEHICLE CLUSTER')
H = cluster_analysis.hopkins(x_object)
print(H)
cluster_analysis.expl_hopkins(x_object, num_iters=1000)
cluster_analysis.cluster_internal_validation(x_object, n_clusters=50)
cluster_analysis.silhouette_coef(x_object.values, range_n_clusters=range(10, 11, 1))
# cluster_analysis.kmeans_plus_plus(x_object, k=10, n_init=42, max_iter=500, drop=None, show_plot=False,
#                                  file_name='cluster_veh')

# DROP DUPLICATES CUSTOMERS
# customer_df = customer_df.sort_values(by=['cliente_poliza'], ascending=[False])
# customer_df = customer_df.drop_duplicates(subset=['cliente_codfiliacion'], keep='first')
print(len(customer_df.index))

# DEL VARIABLE
columns_to_drop = ['cliente_fecha_nacimiento', 'cliente_nacionalidad', 'cliente_pais_residencia',
                   'cliente_fechaini_zurich', 'cliente_tipo_doc', 'vehiculo_uso_codigo', 'vehiculo_uso_desc',
                   'vehiculo_clase_codigo', 'vehiculo_clase_descripcion', 'vehiculo_clase_agrupacion_descripcion',
                   'vehiculo_potencia', 'vehiculo_marca_codigo', 'vehiculo_marca_desc',
                   'vehiculo_modelo_codigo', 'vehiculo_modelo_desc', 'vehiculo_bonus_desc',
                   'vehiculo_fenacco_conductor1',
                   'cliente_nombre_via', 'cliente_numero_hogar', 'vehiculo_fenacco_conductor2',
                   'vehiculo_tipo_combustible', 'COUNTRY', 'vehiculo_fepecon_conductor1'
                   ]

customer_df = customer_df.drop(columns_to_drop, axis=1)
customer_df = customer_df[customer_df['cliente_edad'] - customer_df['antiguedad_permiso'] >= 17]

# RISK INTERMEDIARY
x_mediador = customer_df[['mediador_cod_intermediario', 'mediador_riesgo_auto']]

x_mediador = x_mediador.sort_values(by=['mediador_riesgo_auto'], ascending=[False])
x_mediador = x_mediador.drop_duplicates(subset=['mediador_cod_intermediario'], keep='first')
x_mediador = x_mediador.set_index('mediador_cod_intermediario')
for i in x_mediador.columns.values.tolist():
    x_mediador[i] = x_mediador[i] * 100 / x_mediador[i].max()
    x_mediador[i] = x_mediador[i].round()

print('MEDIADOR CLUSTER')
H = cluster_analysis.hopkins(x_mediador)
print(H)
cluster_analysis.expl_hopkins(x_mediador, num_iters=1000)
cluster_analysis.cluster_internal_validation(x_mediador, n_clusters=50)
# cluster_analysis.silhouette_coef(x_mediador.values, range_n_clusters=range(10, 11, 1))


cluster_analysis.kmeans_plus_plus(x_mediador, k=10, n_init=42, max_iter=500, drop=None, show_plot=False,
                                  file_name='cluster_intm')

# RISK CLUSTERING BY CP
cp_risk = customer_df[(customer_df['cliente_numero_siniestros'] < customer_df[
    'cliente_numero_siniestros'].quantile(0.99))]
print(len(customer_df.index))
cp_risk = cp_risk[['cliente_cp',
                   'cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto']]

cp_risk = cp_risk[cp_risk['cliente_cp'] != 0]

cp_risk['cliente_cp'] = cp_risk['cliente_cp'].map(int)
cp_risk = cp_risk.sort_values(by=['cliente_cp'], ascending=True)

cp_risk = cp_risk.groupby(['cliente_cp']).agg({
    'cliente_numero_siniestros_auto': ['mean'],
    'cliente_numero_polizas_auto': ['mean'],
    'cliente_cp': 'count'
})

print(cp_risk)
cp_risk = cp_risk[cp_risk[('cliente_cp', 'count')] >= 10.0]
del cp_risk[('cliente_cp', 'count')]

cp_risk = cp_risk.reset_index(drop=False)

print('POSTAL CODE CLUSTER')
H = cluster_analysis.hopkins(x_mediador)
print(H)
cluster_analysis.expl_hopkins(cp_risk.drop(['cliente_cp'], axis=1), num_iters=1000)
cluster_analysis.cluster_internal_validation(cp_risk.drop(['cliente_cp'], axis=1), n_clusters=50)
# cluster_analysis.silhouette_coef(cp_risk.drop(['cliente_cp'], axis=1).values, range_n_clusters=range(10, 11, 1))

cp_risk = cluster_analysis.kmeans_plus_plus(cp_risk, k=10, n_init=42, max_iter=500, drop='cliente_cp', show_plot=False,
                                            file_name='clusters_cp')

cp_risk = cp_risk[[('cliente_cp', ''), 'labels']]
cp_risk = cp_risk.rename(columns={('cliente_cp', ''): 'cliente_cp', 'labels': 'cp_risk'})
customer_df['cliente_cp'] = customer_df['cliente_cp'].map(int)
customer_df = pd.merge(customer_df, cp_risk, on='cliente_cp', how='inner')

# RISK CLUSTERING BY CUSTOMER
# First we normalize THE RISK VARIABLES by AGE IN THE COMPANY

# Risk Variables
target_variables = ['cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto']

for i in target_variables:
    customer_df[i] = customer_df[i] / ((customer_df['policy_days'] + 1) / 365)
    customer_df[i] = customer_df[i].round()

customer_df = customer_df[(customer_df['cliente_numero_siniestros_auto'] < customer_df[
    'cliente_numero_siniestros_auto'].quantile(0.99))]

# Cluster at customer
x = customer_df[
    ['cliente_codfiliacion', 'cliente_numero_siniestros', 'cliente_carga_siniestral', 'cliente_numero_siniestros_auto',
     'cliente_sexo', 'cp_risk', 'vehiculo_valor', 'cliente_edad_18_30', 'cliente_edad_30_65', 'cliente_edad_65',
     'edad_segundo_conductor_riesgo', 'antiguedad_permiso', 'antiguedad_permiso_riesgo',
     'antiguedad_permiso_segundo_riesgo',
     'd_uso_particular', 'd_uso_alquiler', 'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion', 'd_tipo_autocar',
     'd_tipo_remolque', 'd_tipo_agricola',
     'd_tipo_industrial', 'd_tipo_triciclo', 'antiguedad_vehiculo', 'cliente_extranjero', 'cliente_edad',
     'vehiculo_categoria',
     'vehiculo_heavy', 'mediador_riesgo', 'mediador_riesgo_auto', 'mediador_share_auto', 'REGION',
     'cliente_numero_siniestros_auto_culpa', 'antiguedad_permiso_range', 'cliente_numero_polizas_auto'
     ]]

x['cliente_edad'] = x['cliente_edad'].map(int)
x['cliente_edad'] = pd.cut(x['cliente_edad'], range(x['cliente_edad'].min(), x['cliente_edad'].max(), 5), right=True)
x['cliente_edad'] = x['cliente_edad'].fillna(x['cliente_edad'].max())

x['cliente_numero_siniestros_auto_culpa_share'] = x['cliente_numero_siniestros_auto_culpa'] * 100 / x[
    'cliente_numero_siniestros_auto']
x.loc[x['cliente_numero_siniestros_auto'] == 0, 'cliente_numero_siniestros_auto_culpa_share'] = 0
x['cliente_numero_siniestros_auto_culpa_share'] = x['cliente_numero_siniestros_auto_culpa_share'].round()
x['counter'] = pd.Series(1, index=x.index)
x_customer = x.groupby(
    ['cliente_edad', 'antiguedad_permiso_range', 'edad_segundo_conductor_riesgo', 'cliente_sexo',
     ]).agg(
    {
        'cliente_numero_siniestros_auto': ['mean'],
        'cliente_numero_polizas_auto': ['mean'],
        'counter': ['count']
    })

x_customer = x_customer[x_customer[('counter', 'count')] > 5]
del x_customer['counter']

print('CUSTOMER CLUSTER')
H = cluster_analysis.hopkins(x_mediador)
print(H)
cluster_analysis.expl_hopkins(x_customer, num_iters=1000)
cluster_analysis.cluster_internal_validation(x_customer, n_clusters=50)
cluster_analysis.silhouette_coef(x_customer.values, range_n_clusters=range(10, 11, 1))

cluster_analysis.kmeans_plus_plus(x_customer, k=10, n_init=42, max_iter=500, drop=None, show_plot=False,
                                  file_name='clusters_customer')
