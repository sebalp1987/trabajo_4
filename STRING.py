import os

LOCAL = os.path.dirname(os.path.abspath(__file__))

# PATH NAMES
path_db = LOCAL + '\\data_training'
path_db_extra = LOCAL + '\\data_extra'
file_claim = LOCAL + '\\data_extra\\clientesiniestro_od.csv'
file_offer = LOCAL + '\\data_extra\\sincoofertas_od (3).csv'
file_country = LOCAL + '\\data_extra\\country_list.csv'

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
