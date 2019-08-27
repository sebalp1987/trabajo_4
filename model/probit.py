import statsmodels.api as sm
import pandas as pd
import STRING

df = pd.read_csv(STRING.processed_target, sep=';')

y = df[['target']]
x_ols = df[['oferta_tomador_sexo','cliente_edad_18_30', 'antiguedad_permiso', 'antiguedad_vehiculo',
            'clusters_zip code_risk', 'clusters_intermediary_risk',
            'clusters_vehicle_risk', 'clusters_customer_risk'
]]

reg1 = sm.Probit(endog=y, exog=x_ols, missing='none')
results = reg1.fit()
print(results.summary())
meffect = results.get_margeff()
print(meffect.summary())
