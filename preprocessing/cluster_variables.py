import pandas as pd
import STRING
from resources import cluster_analysis

pd.set_option('max_colwidth', 10)

customer_df = pd.read_csv(STRING.path_db_extra + '\\historical_data.csv', sep=';', encoding='latin1')

# SUBSET VARIABLES
x = customer_df[[
    'cliente_codfiliacion', 'cliente_poliza', 'cliente_sexo', 'cliente_edad', 'cliente_antiguedad', 'cliente_cp',
    'cliente_numero_siniestros', 'cliente_numero_siniestros_auto', 'cliente_carga_siniestral', 'vehiculo_uso_codigo',
    'vehiculo_valor', 'vehiculo_marca_codigo', 'vehiculo_modelo_codigo', 'vehiculo_categoria',
    'mediador_cod_intermediario', 'cliente_siniestro_rehusado', 'cliente_numero_siniestros_auto_culpa',
    'cliente_numero_polizas',
    'cliente_numero_polizas_auto', 'cliente_antiguedad_days', 'cliente_edad_18_30', 'cliente_edad_30_65',
    'cliente_edad_65', 'edad_segundo_conductor_riesgo', 'antiguedad_permiso', 'antiguedad_permiso_riesgo',
    'antiguedad_permiso_range', 'antiguedad_permiso_segundo_riesgo', 'd_uso_particular', 'd_uso_alquiler',
    'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion', 'd_tipo_autocar', 'd_tipo_remolque', 'd_tipo_agricola',
    'd_tipo_industrial', 'd_tipo_triciclo', 'vehiculo_heavy', 'antiguedad_vehiculo', 'mediador_riesgo',
    'mediador_riesgo_auto', 'mediador_share_auto', 'cliente_region_AFRICAARABE', 'cliente_region_AFRICASUBSHARIANA',
    'cliente_region_AMERICACENTRAL', 'cliente_region_AMERICADELNORTE', 'cliente_region_AMERICADELSUR',
    'cliente_region_ASIACENTRAL', 'cliente_region_ASIAOCCIDENTAL', 'cliente_region_ASIAORIENTAL',
    'cliente_region_EUROPACENTRAL', 'cliente_region_EUROPADELNORTE', 'cliente_region_EUROPADELSUR',
    'cliente_region_EUROPAOCCIDENTAL', 'cliente_region_EUROPAORIENTAL', 'cliente_region_MARCARIBE',
    'cliente_region_OCEANIA', 'cliente_region_nan', 'cliente_extranjero', 'vehiculo_valor_range', 'cliente_edad_range',
    'cliente_numero_siniestros_auto_culpa_share',
    'mediador_numero_siniestros', 'mediador_numero_polizas', 'mediador_numero_polizas_AUTO',
    'mediador_numero_siniestros_AUTO']]

# x = x.sort_values(by=['cliente_poliza'], ascending=[False])
# x = x.drop_duplicates(subset=['cliente_codfiliacion', 'vehiculo_modelo_desc'], keep='first')

n_clusters_max = 10
# H --> 1
# SC --> 1
# CH --> +inf
# DB --> 0

'''
# RISK CLUSTERING BY CP #############################################################################################
cp_risk = customer_df[['cliente_cp',
                       'cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto']]

cp_risk['cliente_cp'] = cp_risk['cliente_cp'].map(int)
cp_risk = cp_risk.sort_values(by=['cliente_cp'], ascending=True)

cp_risk = cp_risk.groupby(['cliente_cp']).agg({
    'cliente_numero_siniestros_auto': ['sum'],
    'cliente_numero_polizas_auto': ['sum'],
    'cliente_cp': 'count'
})

cp_risk.columns = cp_risk.columns.droplevel(1)
cp_risk = cp_risk[cp_risk['cliente_cp'] >= 10.0]
del cp_risk['cliente_cp']

cp_risk['siniestros_polizas'] = cp_risk['cliente_numero_siniestros_auto'] / cp_risk['cliente_numero_polizas_auto']
cp_risk['siniestros_polizas'] = cp_risk['siniestros_polizas']*100
cp_risk = cp_risk[['siniestros_polizas']]
cp_risk = cp_risk.reset_index(drop=False)

print('POSTAL CODE CLUSTER')
cluster_analysis.hopkins(cp_risk.drop(['cliente_cp'], axis=1))
cluster_analysis.cluster_internal_validation(cp_risk.drop(['cliente_cp'], axis=1), n_clusters=n_clusters_max)
cluster_analysis.kmeans_plus_plus(cp_risk, k=4, n_init=100, max_iter=500, drop='cliente_cp',
                                            show_plot=False,
                                            file_name='clusters_cp', reindex_label='siniestros_polizas')


# RISK CLUSTERING BY INTERMEDIARY #####################################################################################
x_mediador = customer_df[
    ['mediador_cod_intermediario', 'mediador_riesgo_auto', 'mediador_share_auto', 'mediador_numero_siniestros',
     'mediador_numero_polizas', 'mediador_numero_polizas_AUTO',
     'mediador_numero_siniestros_AUTO']]

x_mediador = x_mediador.sort_values(by=['mediador_riesgo_auto'], ascending=[False])
x_mediador = x_mediador.drop_duplicates(subset=['mediador_cod_intermediario'], keep='first')
x_mediador = x_mediador[['mediador_cod_intermediario', 'mediador_numero_polizas_AUTO',
                         'mediador_numero_siniestros_AUTO']]
x_mediador = x_mediador[x_mediador['mediador_numero_polizas_AUTO'] >= 10]

print('MEDIADOR CLUSTER')
cluster_analysis.hopkins(x_mediador.drop(['mediador_cod_intermediario'], axis=1))
cluster_analysis.cluster_internal_validation(x_mediador.drop(['mediador_cod_intermediario'], axis=1), n_clusters=n_clusters_max)
cluster_analysis.kmeans_plus_plus(x_mediador, k=7, n_init=100, max_iter=500, drop='mediador_cod_intermediario', show_plot=False,
                                  file_name='cluster_intm', reindex_label=['mediador_numero_siniestros_AUTO','mediador_numero_polizas_AUTO'
                         ])

'''

# AGREGAR MAS VARIABLES PARA EL OBJETIVO DEL CLUSTER!!!!!!!!!!!!!!!!!!!


# RISK CLUSTERING BY OBJECT ###########################################################################################
x_object = customer_df[
    ['cliente_codfiliacion', 'cliente_poliza',
     'cliente_numero_siniestros_auto', 'cliente_carga_siniestral', 'vehiculo_uso_codigo',
     'vehiculo_valor', 'vehiculo_marca_codigo', 'vehiculo_modelo_codigo', 'vehiculo_categoria', 'vehiculo_modelo_desc',
     'd_uso_particular', 'd_uso_alquiler',
     'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion', 'd_tipo_autocar', 'd_tipo_remolque', 'd_tipo_agricola',
     'd_tipo_industrial', 'd_tipo_triciclo', 'vehiculo_heavy', 'antiguedad_vehiculo', 'vehiculo_valor_range',
     'vehiculo_marca_desc', 'veh_uso', 'veh_tipo']]

# We keep the last policy of the customer with the vehicle
x_object = x_object.sort_values(by=['cliente_poliza'], ascending=[False])
x_object = x_object.drop_duplicates(subset=['cliente_codfiliacion', 'vehiculo_modelo_desc'], keep='first')
x_object['counter'] = pd.Series(1, index=x_object.index)

x_object = x_object.groupby(
    ['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
     'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo']).agg(
    {
        'cliente_numero_siniestros_auto': ['mean'], 'cliente_carga_siniestral': ['mean'], 'counter': 'count'})

x_object.columns = x_object.columns.droplevel(1)
x_object = x_object[x_object['counter'] > 5.0]
del x_object['counter']
x_object = x_object.reset_index(drop=False)
print(x_object.columns.values.tolist())
print('VEHICLE CLUSTER')
cluster_analysis.hopkins(x_object.drop(['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
     'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'], axis=1))
cluster_analysis.cluster_internal_validation(x_object.drop(['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
     'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'], axis=1), n_clusters=n_clusters_max)
cluster_analysis.kmeans_plus_plus(x_object, k=7, n_init=100, max_iter=500, drop=['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
     'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'], show_plot=False,
                                  file_name='cluster_obj', reindex_label='cliente_numero_siniestros_auto')


# RISK CLUSTERING BY CUSTOMER ##########################################################################################

# Cluster at customer
x_customer = customer_df[
    ['cliente_codfiliacion', 'cliente_poliza', 'cliente_sexo', 'cliente_edad', 'cliente_antiguedad',
     'cliente_numero_siniestros_auto', 'cliente_carga_siniestral',
     'cliente_numero_polizas_auto', 'cliente_antiguedad_days', 'cliente_edad_18_30', 'cliente_edad_30_65',
     'cliente_edad_65', 'edad_segundo_conductor_riesgo', 'antiguedad_permiso', 'antiguedad_permiso_riesgo',
     'antiguedad_permiso_range', 'antiguedad_permiso_segundo_riesgo', 'cliente_region_AFRICAARABE', 'edad_conductor_riesgo',
     'cliente_region_AFRICASUBSHARIANA',
     'cliente_region_AMERICACENTRAL', 'cliente_region_AMERICADELNORTE', 'cliente_region_AMERICADELSUR',
     'cliente_region_ASIACENTRAL', 'cliente_region_ASIAOCCIDENTAL', 'cliente_region_ASIAORIENTAL',
     'cliente_region_EUROPACENTRAL', 'cliente_region_EUROPADELNORTE', 'cliente_region_EUROPADELSUR',
     'cliente_region_EUROPAOCCIDENTAL', 'cliente_region_EUROPAORIENTAL', 'cliente_region_MARCARIBE',
     'cliente_region_OCEANIA', 'cliente_region_nan', 'cliente_extranjero', 'cliente_edad_range',
     'cliente_numero_siniestros_auto_culpa_share', 'vehiculo_modelo_desc'
     ]]

# First we sum the customer policies
x_customer = x_customer.sort_values(by=['cliente_poliza'], ascending=[False])
x_customer = x_customer.drop_duplicates(subset=['cliente_codfiliacion', 'vehiculo_modelo_desc'], keep='first')

x_customer['counter'] = pd.Series(1, index=x_customer.index)
x_customer = x_customer.groupby(
    ['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo', 'cliente_sexo',
     'antiguedad_permiso_segundo_riesgo',
     'cliente_extranjero', 'edad_conductor_riesgo'
     ]).agg(
    {
        'cliente_numero_siniestros_auto': ['mean'],
        'counter': ['count']
    })
x_customer.columns = x_customer.columns.droplevel(1)
x_customer = x_customer[x_customer['counter'] > 5]
del x_customer['counter']
x_customer = x_customer.reset_index(drop=False)

print('CUSTOMER CLUSTER')
cluster_analysis.hopkins(x_customer.drop(['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo', 'cliente_sexo',
     'antiguedad_permiso_segundo_riesgo',
     'cliente_extranjero', 'edad_conductor_riesgo'
     ], axis=1))
cluster_analysis.cluster_internal_validation(x_customer.drop(['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo', 'cliente_sexo',
     'antiguedad_permiso_segundo_riesgo',
     'cliente_extranjero'
     ], axis=1), n_clusters=n_clusters_max)
cluster_analysis.kmeans_plus_plus(x_customer, k=6, n_init=100, max_iter=500, drop=['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo', 'cliente_sexo',
     'antiguedad_permiso_segundo_riesgo',
     'cliente_extranjero', 'edad_conductor_riesgo'
     ], show_plot=False,
                                  file_name='cluster_customer', reindex_label='cliente_numero_siniestros_auto')
