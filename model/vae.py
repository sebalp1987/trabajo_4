import pandas as pd
from keras import Input, layers, backend as K, objectives
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from model.ae import DeepAutoencoder
import STRING
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_curve, recall_score, precision_score, fbeta_score)
import seaborn as sns
import numpy as np

sns.set()
latent_dim = 2


def sampling(args):
    """
    This is going to create the latent space, generating a normal probability distribution from the input data
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)

    return z_mean + K.exp(z_log_var / 2) * epsilon


class VariationalAutoencoder(object):
    def __init__(self, batch_size, latent_dim):
        self.batch_size = batch_size
        self.latent_dim = latent_dim

    def vae_loss(self, x, x_decoded_mean, args):
        """
        We train with two loss functions. Reconstruction loss force decoded samples to match to X (just like
        autoencoder). KL loss (latent loss) calculates the divergence between the learned latent distribution
        derived from z_mean and z_logsigma and the original distribution of X.
        """
        z_mean, z_logsigma = args
        print('z_mean', z_mean)
        print('logsig', z_logsigma)
        reconstruction_loss = objectives.cosine_proximity(x, x_decoded_mean)
        latent_loss = -0.50 * K.mean(1 + z_logsigma - K.square(z_mean) - K.exp(z_logsigma), axis=-1)
        print('RL ', reconstruction_loss)
        print('LL ', latent_loss)
        return K.mean(reconstruction_loss + latent_loss)


if __name__ == '__main__':

    import scikitplot as skplt

    seed = 42
    np.random.seed(seed)

    # LOAD FILE
    normal = pd.read_csv(STRING.normal_file, sep=';', encoding='latin1')
    anormal = pd.read_csv(STRING.anormal_file, sep=';', encoding='latin1')

    # NORMALIZE
    normal['CONTROL'] = pd.Series(0, index=normal.index)
    anormal['CONTROL'] = pd.Series(1, index=anormal.index)

    normalize = pd.concat([normal, anormal], axis=0)
    for i in normalize.drop(['oferta_id', 'target', 'CONTROL'], axis=1).columns.values.tolist():
        normalize[i] = normalize[i].map(float)
        normalize[i] = StandardScaler().fit_transform(normalize[i].values.reshape(-1, 1))

    normal = normalize[normalize['CONTROL'] == 0]
    anormal = normalize[normalize['CONTROL'] == 1]

    del normal['CONTROL']
    del anormal['CONTROL']

    # VARIANCE REDUCTION
    selection = VarianceThreshold(threshold=0.0)
    selection.fit(normal.drop(['oferta_id', 'target'], axis=1))
    features = selection.get_support(indices=True)
    features = list(normal.columns[features]) + ['target']
    normal = normal[features]
    test_anormal = anormal[features]

    y_pred_score = np.empty(shape=[0, 2])
    predicted_index = np.empty(shape=[0, ])
    # We divide the normal and the abnormal dataset
    train, valid, _, _ = train_test_split(normal, normal, test_size=0.30, random_state=42)
    valid, test_normal, _, _ = train_test_split(valid, valid, test_size=len(anormal.index), random_state=42)
    valid = valid.drop(['oferta_id', 'target'], axis=1)
    print(train.shape)
    print(valid.shape)
    print(test_normal.shape)
    # INPUT COLS
    cols = train.drop(['oferta_id', 'target'], axis=1).shape[1]
    input = Input(shape=(cols,))

    x = layers.Dense(cols, activation='tanh')(input)
    branch_b = layers.Dense(cols, activation='tanh')(input)
    branch_b = layers.Dense(int(cols * 2 / 4), activation='tanh')(branch_b)
    branch_b = layers.Dense(int(cols * 3 / 4), activation='tanh')(branch_b)

    ae = DeepAutoencoder(n_cols=cols, activation='tanh', prob_dropout=0.2, dimension_node=6, encoding_dim=6,
                         final_activation='relu')
    vae = VariationalAutoencoder(batch_size=100, latent_dim=latent_dim)
    encoded, intermediate_dim = ae.encoded(input, change_encode_name='Encoder')
    print('First_Layer', encoded)

    # We generate z_mean and sigma from the encoded distribution
    z_mean = layers.Dense(latent_dim, name='z_mean')(encoded)
    z_log_var = layers.Dense(latent_dim, name='z_var')(encoded)

    # We generate the Distribution of Z (Lambda wraps in a layer an arbitrary function)
    Z = layers.Lambda(sampling, output_shape=(latent_dim,))(([z_mean, z_log_var]))
    print('layer Z: ', Z)

    # We decode Z
    decode_z = layers.Dense(intermediate_dim, activation='tanh', name='decode_Z')(Z)
    print(decode_z)

    # If you want to add more (OPTIONAL)
    decode_z = layers.Dense(int(cols * 1 / 4), activation='tanh', name='decode_Z_1')(decode_z)
    decode_z = layers.Dense(int(cols * 2 / 4), activation='tanh', name='decode_Z_2')(decode_z)
    decode_z = layers.Dense(int(cols * 3 / 4), activation='tanh', name='decode_Z_3')(decode_z)

    # We decode X from Z
    decode_x = layers.Dense(cols, activation='sigmoid', name='decode_xZ')(decode_z)
    print(decode_x)

    # VAE Model

    vae_model = Model(input, decode_x)
    loss = vae.vae_loss(input, decode_x, [z_mean, z_log_var])
    vae_model.add_loss(loss)
    optimizer = Adam(lr=0.001)
    vae_model.compile(optimizer=optimizer)
    print(vae_model.summary())
    early_stopping_monitor = EarlyStopping(patience=2)
    history = vae_model.fit(train.drop(['oferta_id', 'target'], axis=1), epochs=1000, batch_size=100, verbose=True,
                            callbacks=[early_stopping_monitor],
                            shuffle=True, validation_data=[valid, None]).history

    # Plot Loss
    plot.plot(history['loss'])
    plot.plot(history['val_loss'])
    plot.title('model loss')
    plot.ylabel('loss')
    plot.xlabel('epoch')
    plot.legend(['train', 'valid'], loc='upper right')
    plot.savefig(STRING.img_path + 'error_convergence.png')
    plot.show()

    # VAE Loss
    prediction_true = vae_model.predict(valid)
    prediction_test = vae_model.predict(test_normal.drop(['oferta_id', 'target'], axis=1))
    prediction_anormal = vae_model.predict(test_anormal.drop(['oferta_id', 'target'], axis=1))

    mse_true = np.mean(np.power(valid - prediction_true, 2), axis=1)
    mse_test = np.mean(np.power(test_normal.drop(['oferta_id', 'target'], axis=1) - prediction_test, 2), axis=1)
    mse_anormal = np.mean(np.power(test_anormal.drop(['oferta_id', 'target'], axis=1) - prediction_anormal, 2), axis=1)

    mse_true = pd.DataFrame(mse_true, columns=['reconstruction_error'], index=valid.index)
    mse_test = pd.DataFrame(mse_test, columns=['reconstruction_error'], index=test_normal.index)
    mse_test['oferta_id'] = pd.Series(test_normal['oferta_id'].values.tolist(), index=mse_test.index)
    mse_anormal = pd.DataFrame(mse_anormal, columns=['reconstruction_error'], index=test_anormal.index)
    mse_anormal['oferta_id'] = pd.Series(test_anormal['oferta_id'].values.tolist(), index=mse_anormal.index)

    mse_true['target'] = pd.Series(0, index=mse_true.index)
    mse_test['target'] = pd.Series(0, index=mse_test.index)
    mse_anormal['target'] = pd.Series(1, index=mse_anormal.index)
    error_df = pd.concat([mse_test, mse_anormal], axis=0)

    # We separate the values
    mse_test_valid, mse_test = train_test_split(mse_test, test_size=0.5, random_state=42)
    mse_anormal_valid, mse_anormal = train_test_split(mse_anormal, test_size=0.5, random_state=42)
    error_df_valid = pd.concat([mse_test_valid, mse_anormal_valid], axis=0).reset_index(drop=True)
    error_df_test = pd.concat([mse_test, mse_anormal], axis=0).reset_index(drop=True)
    error_df_test['target'] = error_df_test['target'].map(int)
    error_df_valid['target'] = error_df_valid['target'].map(int)

    # PLOT ERROR WITHOUT ANOMALIES
    fig = plot.figure()
    ax = fig.add_subplot(111)
    normal_error_df = error_df[(error_df['target'] == 0) & (error_df['reconstruction_error'] < 10)]
    _ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
    plot.savefig(STRING.img_path + 'rc_error_normal.png')
    plot.show()
    plot.close()

    # PLOT ERROR WITH ANOMALIES
    fig = plot.figure()
    ax = fig.add_subplot(111)
    fraud_error_df = error_df[(error_df['target'] == 1) & (error_df['reconstruction_error'])]
    _ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
    plot.savefig(STRING.img_path + 'rc_error_anormal.png')
    plot.show()
    plot.close()

    # RECALL-PRECISION
    precision, recall, th = precision_recall_curve(error_df_valid.target, error_df_valid.reconstruction_error)
    plot.plot(recall, precision, 'b', label='Precision-Recall curve')
    plot.title('Recall vs Precision')
    plot.xlabel('Recall')
    plot.ylabel('Precision')
    plot.savefig(STRING.img_path + 'recall_precision.png')
    plot.show()

    plot.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
    plot.plot(th, recall[1:], 'g', label='Threshold-Recall curve')
    plot.title('Precision-Recall for different threshold values')
    plot.xlabel('Threshold')
    plot.ylabel('Precision-Recall')
    plot.xlim(left=0, right=2)
    plot.legend(['precision', 'recall'], loc='upper right')
    plot.savefig(STRING.img_path + 'figure_1.png')
    plot.show()

    # OUTLIER DETECTION
    # We define a threshold for the reconstruction error. It will be based on the error plot
    thresholds = np.linspace(0.001, 100.0, 10000)

    scores = []

    for threshold in thresholds:
        y_hat = [1 if e > threshold else 0 for e in error_df_valid.reconstruction_error.values]
        scores.append([
            recall_score(y_pred=y_hat, y_true=error_df_valid.target.values),
            precision_score(y_pred=y_hat, y_true=error_df_valid.target.values),
            fbeta_score(y_pred=y_hat, y_true=error_df_valid.target.values,
                        beta=1, average='binary')
        ])

    scores = np.array(scores)
    threshold = thresholds[scores[:, 2].argmax()]
    print('final Threshold ', threshold)
    predicted = [1 if e > threshold else 0 for e in error_df_test.reconstruction_error.values]

    precision = precision_score(error_df_test.target.values, predicted)
    recall = recall_score(error_df_test.target.values, predicted)
    fbeta = fbeta_score(error_df_test.target.values, predicted, beta=1)
    print('PRECISION ', precision)
    print('RECALL ', recall)
    print('FBSCORE ', fbeta)

    # Reconstruction Error plot
    groups = error_df.groupby('target')
    fig, ax = plot.subplots()
    for name, group in groups:
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                label="Anomaly" if name == 1 else "Normal")
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plot.title("Reconstruction error for different classes")
    plot.ylabel("Reconstruction error")
    plot.xlabel("Data point index")
    plot.savefig(STRING.img_path + 'final_result.png')
    plot.show()

    # Confussion Matrix
    conf_matrix = confusion_matrix(error_df_test.target, predicted)
    plot.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot=True,
                fmt="d", cmap="Blues", cbar=False)
    plot.title("Confusion matrix")
    plot.ylabel('True class')
    plot.xlabel('Predicted class')
    plot.savefig(STRING.img_path + 'confussion_matrix.png')
    plot.show()

    # Lift Curve
    y_hat_df = pd.DataFrame(None, index=error_df_test.index, columns=['y_hat_1'])
    y_hat_df['y_hat_1'] = error_df_test['reconstruction_error']
    y_hat_df['y_hat_0'] = error_df_test['reconstruction_error']
    error_df_test['predicted'] = pd.Series(predicted, index=error_df_test.index)

    skplt.metrics.plot_cumulative_gain(y_true=error_df_test.target.values, y_probas=y_hat_df[['y_hat_0', 'y_hat_1']])
    plot.savefig(STRING.img_path + 'figure_1-1.png')
    plot.show()
    plot.close()
    skplt.metrics.plot_lift_curve(y_true=error_df_test.target.values, y_probas=y_hat_df[['y_hat_0', 'y_hat_1']])
    plot.savefig(STRING.img_path + 'figure_1-2.png')
    plot.show()

    error_df_test = error_df_test.sort_values(by=['reconstruction_error'], ascending=False).reset_index(
        drop=True).reset_index(drop=False)
    error_df_test['percentage_sample'] = (error_df_test['index'] + 1) / error_df_test['index'].max()
    error_df_test['tptn'] = pd.Series(0, index=error_df_test.index)
    error_df_test.loc[error_df_test['predicted'] == error_df_test['target'], 'tptn'] = 1
    error_df_test['tptn_sum'] = error_df_test['tptn'].cumsum() / (error_df_test['index'] + 1)

    error_df_test.to_csv(STRING.lift_curve, index=False, sep=';')
