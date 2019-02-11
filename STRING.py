import os

LOCAL = os.path.dirname(os.path.abspath(__file__))

# PATH NAMES
path_training = LOCAL + '/data_training/'
path_db_extra = LOCAL + '/data_extra/'

file_claim = LOCAL + '/data_extra/clientesiniestro_od.csv'
file_offer = LOCAL + '/data_extra/sincoofertas_od (3).csv'
file_country = LOCAL + '/data_extra/country_list.csv'
file_veh_use = LOCAL + '/data_extra/vehicle_use.csv'
file_veh_clase = LOCAL + '/data_extra/vehicle_clase.csv'
file_veh_marca = LOCAL + '/data_extra/vehicle_marca.csv'
file_bonus = LOCAL + '/data_extra/bonus_list.csv'


cluster_risk = path_db_extra + 'clusters_zip.csv'
cluster_intm = path_db_extra + 'clusters_intermediary.csv'
cluster_veh = path_db_extra + 'clusters_vehicle.csv'
cluster_customer = path_db_extra + 'clusters_customer.csv'

processed_historical = path_db_extra + 'historical_data.csv'
processed_offer_before = path_db_extra + 'oferta_processed_before.csv'
processed_offer = path_db_extra + 'oferta_processed.csv'
processed_target = path_db_extra + 'oferta_processed_target.csv'

summary_statistics_offers = path_db_extra + 'summary_statistics_offers.csv'

normal_file = path_training + 'normal.csv'
anormal_file = path_training + 'anormal.csv'
normal_describe = path_db_extra + 'normal_avg.csv'
anormal_describe = path_db_extra + 'anormal_avg.csv'

train = path_training + 'train.csv'
valid = path_training + 'valid.csv'
test = path_training + 'test.csv'

test_sample_1 = path_training + 'test_sample1.csv'
test_sample_2 = path_training + 'test_sample2.csv'

metric_save = path_db_extra + 'metric_save.csv'

img_path = os.path.dirname(os.path.dirname(LOCAL)) + '/3-latex/images/'
tensorboard_path = os.path.dirname(LOCAL) + '/tensorboard/'
print(tensorboard_path)

# PARAMETERS
configure_names = {'cliente_numero_siniestros_auto': 'Sum of vehicle claims per year',
                   'cliente_numero_polizas_auto': 'Sum of vehicle policies per year',
                   'mediador_numero_siniestros_AUTO': 'Sum of intermediary claims per year',
                   'mediador_numero_polizas_AUTO': 'Sum of intermediary policies per year',
                   'vehiculo_valor_range': 'Vehicle Value', 'vehiculo_marca_desc': 'Vehicle Brand',
                   'veh_uso': 'Vehicle Usage', 'veh_tipo': 'Vehicle Type', 'antiguedad_vehiculo': 'Vehicle Years',
                   'cliente_edad_range': 'Customer Age', 'antiguedad_permiso_range': 'License Years',
                   'antiguedad_permiso_riesgo': 'License Years < 1',
                   'edad_segundo_conductor_riesgo': 'Second Driver Age < 22',
                   'cliente_extranjero': 'Spanish / Foreigner', 'edad_conductor_riesgo': 'Customer Age < 22',
                   'cliente_sexo': 'Gender', 'antiguedad_permiso_segundo_riesgo': 'Second Driver License Years < 1'}
