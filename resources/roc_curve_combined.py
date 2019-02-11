import STRING
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from ast import literal_eval

sns.set()

df = pd.read_csv(STRING.metric_save, sep=';')

print(df)

colors = ['lightcoral','lightseagreen', 'lightgreen']

df_1 = df[df['sample'] == 1]
df_2 = df[df['sample'] == 2]

for model in df_1['model'].values.tolist():

    fpr = df_1.loc[df['model'] == model, 'fpr'].iloc[0]
    fpr = literal_eval(fpr)
    tpr = df_1.loc[df['model'] == model, 'tpr'].iloc[0]
    tpr = literal_eval(tpr)
    roc_auc = df_1.loc[df['model'] == model, 'auc'].iloc[0]
    print(roc_auc)
    if model in ['nn', 'ic', 'rc']:
        linestyle = ':'
        color = colors[2]
    elif model == 'ert':
        linestyle = '-.'
        color = colors[1]
    else:
        linestyle = '-'
        color = colors[0]
    plot.plot(fpr, tpr, color=color, lw=2,
              label='ROC curve of class {0} (area = {1:0.2f})'''.format(model, roc_auc), linestyle=linestyle)

plot.plot([0, 1], [0, 1], 'k--', lw=2)
plot.xlim([0.0, 1.0])
plot.ylim([0.0, 1.05])
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.legend(loc="lower right")
plot.savefig(STRING.img_path + 'roc_curve_1.png')
plot.show()


for model in df_2['model'].values.tolist():

    fpr = df_2.loc[df['model'] == model, 'fpr'].iloc[0]
    fpr = literal_eval(fpr)
    tpr = df_2.loc[df['model'] == model, 'tpr'].iloc[0]
    tpr = literal_eval(tpr)
    roc_auc = df_2.loc[df['model'] == model, 'auc'].iloc[0]
    print(roc_auc)
    if model in ['nn', 'ic', 'rc']:
        linestyle = ':'
        color = colors[2]
    elif model == 'ert':
        linestyle = '-.'
        color = colors[1]
    else:
        linestyle = '-'
        color = colors[0]
    plot.plot(fpr, tpr, color=color, lw=2,
              label='ROC curve of class {0} (area = {1:0.2f})'''.format(model, roc_auc), linestyle=linestyle)

plot.plot([0, 1], [0, 1], 'k--', lw=2)
plot.xlim([0.0, 1.0])
plot.ylim([0.0, 1.05])
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.legend(loc="lower right")
plot.savefig(STRING.img_path + 'roc_curve_2.png')
plot.show()

