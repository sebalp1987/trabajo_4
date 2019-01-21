import pandas as pd
import STRING
from resources import cluster_analysis
import matplotlib.pyplot as plot
import seaborn as sns

sns.set()
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
    'd_uso_publico',
    'd_tipo_car', 'd_tipo_motorbike', 'd_tipo_van-track', 'd_tipo_autocar', 'd_tipo_agricola', 'd_tipo_industrial',
    'vehiculo_heavy', 'antiguedad_vehiculo', 'mediador_riesgo',
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
config_names = STRING.configure_names
# H --> 1
# SC --> 1
# CH --> +inf
# DB --> 0


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
cp_risk['siniestros_polizas'] = cp_risk['siniestros_polizas'] * 100
cp_risk = cp_risk[['cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto']]
cp_risk = cp_risk.reset_index(drop=False)

print('POSTAL CODE CLUSTER')
cluster_analysis.hopkins(cp_risk.drop(['cliente_cp'], axis=1))
cluster_analysis.cluster_internal_validation(cp_risk.drop(['cliente_cp'], axis=1), n_clusters=n_clusters_max)
fig_cp = cluster_analysis.kmeans_plus_plus(cp_risk, k=6, n_init=1, max_iter=1000, drop='cliente_cp',
                                  show_plot=True,
                                  file_name='clusters_zip code', reindex_label=['cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto'])

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
cluster_analysis.cluster_internal_validation(x_mediador.drop(['mediador_cod_intermediario'], axis=1),
                                             n_clusters=n_clusters_max)
cluster_analysis.kmeans_plus_plus(x_mediador, k=4, n_init=1, max_iter=1000, drop='mediador_cod_intermediario',
                                  show_plot=True,
                                  file_name='clusters_intermediary',
                                  reindex_label=['mediador_numero_siniestros_AUTO', 'mediador_numero_polizas_AUTO'
                                                 ])

# RISK CLUSTERING BY OBJECT ###########################################################################################
x_object = customer_df[
    ['cliente_codfiliacion', 'cliente_poliza', 'cliente_numero_polizas_auto',
     'cliente_numero_siniestros_auto', 'cliente_carga_siniestral', 'vehiculo_uso_codigo',
     'vehiculo_valor', 'vehiculo_marca_codigo', 'vehiculo_modelo_codigo', 'vehiculo_categoria', 'vehiculo_modelo_desc',
     'd_uso_particular', 'd_uso_alquiler', 'd_uso_publico',
     'd_tipo_car', 'd_tipo_motorbike', 'd_tipo_van-track', 'd_tipo_autocar', 'd_tipo_agricola', 'd_tipo_industrial',
     'vehiculo_heavy', 'antiguedad_vehiculo', 'vehiculo_valor_range',
     'vehiculo_marca_desc', 'veh_uso', 'veh_tipo']]

# We keep the last policy of the customer with the vehicle
x_object = x_object.sort_values(by=['cliente_poliza'], ascending=[False])
x_object = x_object.drop_duplicates(subset=['cliente_codfiliacion', 'vehiculo_modelo_desc'], keep='first')
x_object['counter'] = pd.Series(1, index=x_object.index)

x_object = x_object.groupby(
    ['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
     'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo']).agg(
    {
        'cliente_numero_siniestros_auto': ['sum'], 'cliente_numero_polizas_auto': ['sum'], 'counter': 'count'})

x_object.columns = x_object.columns.droplevel(1)
# We quit one outlier case
x_object = x_object[(x_object['cliente_numero_siniestros_auto'] / x_object['cliente_numero_polizas_auto']) < 10]
x_object = x_object[x_object['counter'] > 5]
del x_object['counter']
x_object = x_object.reset_index(drop=False)

print('VEHICLE CLUSTER')
cluster_analysis.hopkins(x_object.drop(['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
                                        'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'], axis=1))
cluster_analysis.cluster_internal_validation(
    x_object.drop(['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
                   'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'], axis=1), n_clusters=n_clusters_max)
x_object = cluster_analysis.kmeans_plus_plus(x_object, k=3, n_init=1, max_iter=1000,
                                             drop=['vehiculo_valor_range', 'vehiculo_categoria', 'vehiculo_marca_desc',
                                                   'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo'],
                                             show_plot=True,
                                             file_name='clusters_vehicle',
                                             reindex_label=['cliente_numero_siniestros_auto',
                                                            'cliente_numero_polizas_auto'])

# EXTRA PLOTS
x_object['vehiculo_valor_range'] = [x.replace(')', ']') for x in x_object['vehiculo_valor_range']]
x_object['vehiculo_valor_range'] = [x.replace('(', '[') for x in x_object['vehiculo_valor_range']]
x_object = x_object.sort_values(by=['cliente_numero_siniestros_auto_cliente_numero_polizas_auto'], ascending=[True])

for col in x_object.drop(['cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto', 'cliente_numero_siniestros_auto_cliente_numero_polizas_auto'], axis=1).columns.values.tolist():
    print(col)
    
    try:
        x_label = config_names[col]
    except KeyError:
        x_label = col
    try:
        if col is not 'vehiculo_marca_desc':
            g = sns.catplot(y='cliente_numero_siniestros_auto_cliente_numero_polizas_auto', x=col,
                       data=x_object)
            g.set(ylabel='Claims / Policies per year', xlabel=x_label)
        else:
            g = sns.catplot(x='cliente_numero_siniestros_auto_cliente_numero_polizas_auto', y=col, data=x_object)
            g.set(xlabel='Claims / Policies per year', ylabel=x_label)
    except TypeError:
        plot.close()
        if col is 'vehiculo_valor_range':
            range_values = []
            for i in range(0, 69000, 1000):
                range_values.append('[' + str(i) + ', ' + str(i + 1000) + ']')
            steps = 5
        else:
            range_values = None
            steps = None

        g = sns.catplot(y='cliente_numero_siniestros_auto_cliente_numero_polizas_auto', x=col, data=x_object,
                        order=range_values)
        g.set_xticklabels(rotation=30, step=steps)
        g.set(ylabel='Claims / Policies per year', xlabel=x_label)

    plot.show()

# RISK CLUSTERING BY CUSTOMER ##########################################################################################

# Cluster at customer
x_customer = customer_df[
    ['cliente_codfiliacion', 'cliente_poliza', 'cliente_sexo', 'cliente_edad', 'cliente_antiguedad',
     'cliente_numero_siniestros_auto', 'cliente_carga_siniestral',
     'cliente_numero_polizas_auto', 'cliente_antiguedad_days', 'cliente_edad_18_30', 'cliente_edad_30_65',
     'cliente_edad_65', 'edad_segundo_conductor_riesgo', 'antiguedad_permiso', 'antiguedad_permiso_riesgo',
     'antiguedad_permiso_range', 'antiguedad_permiso_segundo_riesgo', 'cliente_region_AFRICAARABE',
     'edad_conductor_riesgo',
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
    ['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo',
     'cliente_sexo',
     'antiguedad_permiso_segundo_riesgo',
     'cliente_extranjero', 'edad_conductor_riesgo'
     ]).agg(
    {
        'cliente_numero_siniestros_auto': ['sum'],
        'cliente_numero_polizas_auto': ['sum'],
        'counter': ['count']
    })
x_customer.columns = x_customer.columns.droplevel(1)
x_customer = x_customer[x_customer['counter'] > 10]
del x_customer['counter']
x_customer = x_customer.reset_index(drop=False)

print('CUSTOMER CLUSTER')
cluster_analysis.hopkins(x_customer.drop(
    ['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo',
     'cliente_sexo',
     'antiguedad_permiso_segundo_riesgo',
     'cliente_extranjero', 'edad_conductor_riesgo'
     ], axis=1))
cluster_analysis.cluster_internal_validation(x_customer.drop(
    ['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo', 'edad_segundo_conductor_riesgo',
     'cliente_sexo',
     'antiguedad_permiso_segundo_riesgo',
     'cliente_extranjero'
     ], axis=1), n_clusters=n_clusters_max)
x_customer = cluster_analysis.kmeans_plus_plus(x_customer, k=6, n_init=1, max_iter=1000,
                                  drop=['cliente_edad_range', 'antiguedad_permiso_range', 'antiguedad_permiso_riesgo',
                                        'edad_segundo_conductor_riesgo', 'cliente_sexo',
                                        'antiguedad_permiso_segundo_riesgo',
                                        'cliente_extranjero', 'edad_conductor_riesgo'
                                        ], show_plot=True,
                                  file_name='clusters_customer',
                                  reindex_label=['cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto'])

# EXTRA PLOTS
for col in x_customer.drop(['cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto', 'cliente_numero_siniestros_auto_cliente_numero_polizas_auto'], axis=1).columns.values.tolist():
    print(col)
    try:
        x_label = config_names[col]
    except KeyError:
        x_label = col
    range_values=None
    steps=None

    g = sns.catplot(y='cliente_numero_siniestros_auto_cliente_numero_polizas_auto', x=col, data=x_customer,
                    order=range_values)
    g.set_xticklabels(rotation=0, step=steps)
    g.set(ylabel='Claims / Policies per year', xlabel=x_label)
    plot.show()