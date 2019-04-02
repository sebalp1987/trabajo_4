import STRING
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, fbeta_score

sns.set()

anormal = pd.read_csv(STRING.anormal_file, sep=';')
normal = pd.read_csv(STRING.normal_file, sep=';')

plot.hist(normal['oferta_sim_anios_asegurado'] / 10)
plot.xlabel('Years as Insured')
plot.savefig(STRING.img_path + 'years_insured_normal.png')
plot.show()
plot.close()
plot.hist(anormal['oferta_sim_anios_asegurado'] / 10)
plot.xlabel('Years as Insured')
plot.savefig(STRING.img_path + 'years_insured_anormal.png')
plot.show()
plot.hist(normal['oferta_sim_antiguedad_cia_actual'] / 10)
plot.xlabel('Years Insured Last Company')
plot.savefig(STRING.img_path + 'years_insured_last_normal.png')
plot.show()
plot.close()
plot.hist(anormal['oferta_sim_antiguedad_cia_actual'] / 10)
plot.xlabel('Years Insured Last Company')
plot.savefig(STRING.img_path + 'years_insured_last_anormal.png')
plot.show()
