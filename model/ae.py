import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from itertools import permutations

from keras import Input, layers, regularizers, losses
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import plot_model

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (confusion_matrix, precision_recall_curve, recall_score, precision_score, fbeta_score,
                             roc_curve, roc_auc_score)

import STRING

sns.set()


class DeepAutoencoder(object):
    def __init__(self, n_cols, activation, prob_dropout, dimension_node, encoding_dim=None, final_activation='relu'):
        """
        :param n_cols: Number of cols of datastet 
        :param activation: Activation function
        :param prob_dropout: proportion to dropout
        :param dimension_node: Number of depth of encoder/decoder
        """
        self.n_cols = n_cols
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.dimension_node = dimension_node
        self.encoding_dim = encoding_dim
        self.final_activation = final_activation

    def encoded(self, input_layer, sparsity_const=0.0001, change_encode_name=None):
        """
        Generate the encode layers
        :param input_layer: The input layer
        :param sparsity_const: Restrict some nodes and not all (as PCA), using regularization strategy
        :param change_encode_name: Define a different layer name
        :return: encode layer
        """
        dim = self.dimension_node
        cols = self.n_cols
        node_size = int(cols / dim)
        nodes_range = range(cols - node_size, node_size - 1, -node_size)

        for nodes in nodes_range:
            if (nodes == nodes_range[-1]) & (self.encoding_dim is not None):
                last_node = self.encoding_dim
            else:
                last_node = nodes

            if nodes == nodes_range[-1]:
                last_activation = self.final_activation
            else:
                last_activation = self.activation

            layer_name = 'Encoded_model_' + str(last_node)
            if change_encode_name is not None:
                layer_name = 'Encoded_model_' + change_encode_name + '_' + str(last_node)

            if sparsity_const is not None:
                input_layer = layers.Dense(last_node, activation=last_activation, name=layer_name, activity_regularizer=
                                           regularizers.l1_l2(sparsity_const, sparsity_const))(input_layer)
            else:
                input_layer = layers.Dense(last_node, activation=last_activation, name=layer_name)(input_layer)
            if self.prob_dropout is not None:
                input_layer = layers.Dropout(self.prob_dropout)(input_layer)

        encode_layer = input_layer

        return encode_layer, last_node

    def decoded(self, encode_layer, change_decode_name=None):
        """
        Generate the decoded model
        :param encode_layer: The input layer to the decode
        :param change_decode_name: Define a different layer name
        :return: decoded layer
        """
        dim = self.dimension_node
        cols = self.n_cols
        node_size = int(cols / dim)
        nodes_range = range(node_size * 2, cols, node_size)
        name = 'Decoded_Model'
        for nodes in nodes_range:
            layer_name = 'Decoded_model_' + str(nodes)
            if change_decode_name is not None:
                layer_name = 'Encoded_model_' + change_decode_name + '_' + str(nodes)
                name = 'Decoded_Model_' + change_decode_name
            encode_layer = layers.Dense(nodes, activation=self.activation, name=layer_name)(encode_layer)

        encode_layer = layers.Dense(self.n_cols, activation=self.final_activation, name=name)(encode_layer)

        decode_layer = encode_layer
        return decode_layer

    def autoencoder(self, input_tensor):
        """
        Generate the autoencoder model
        :param input_tensor: the original input tensor
        :return: autoencoder model
        """
        encode_layer, _ = DeepAutoencoder.encoded(self, input_tensor)
        decode_layer = DeepAutoencoder.decoded(self, encode_layer)
        autoencoder = Model(input_tensor, decode_layer)
        plot_model(autoencoder, to_file=STRING.img_path + 'ae_architecture.png', show_shapes=True)

        print(autoencoder.summary())

        return autoencoder

    def fit(self, x, x_valid, learning_rate=0.001, loss_function=losses.mean_squared_error, epochs=500,
            batch_size=500, verbose=True, callback_list=[], steps_per_epoch=None, validation_steps=None):
        input_tensor = Input(shape=(self.n_cols,), name='Input')
        autoencoder = DeepAutoencoder.autoencoder(self, input_tensor)
        optimizer = Adam(lr=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss=loss_function)
        history = autoencoder.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callback_list,
                                  shuffle=True, validation_data=x_valid, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps).history


if __name__ == '__main__':

    import os

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
        normalize[i] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(normalize[i].values.reshape(-1, 1))

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

    train, valid, _, _ = train_test_split(normal, normal, test_size=0.30, random_state=42)
    valid, test_normal, _, _ = train_test_split(valid, valid, test_size=len(anormal.index), random_state=10)
    valid = valid.drop(['oferta_id', 'target'], axis=1)

    # INPUT COLS
    cols = train.drop(['oferta_id', 'target'], axis=1).shape[1]

    ae = DeepAutoencoder(n_cols=cols, activation='tanh', prob_dropout=0.3, dimension_node=3, encoding_dim=16,
                         final_activation='sigmoid')
    early_stopping_monitor = EarlyStopping(patience=2)
    # tensorboard = TensorBoard(log_dir=STRING.tensorboard_path, histogram_freq=1)
    ae.fit(train.drop(['oferta_id', 'target'], axis=1), x_valid=[valid, valid], callback_list=[early_stopping_monitor],
           batch_size=None, epochs=1000, steps_per_epoch=15, validation_steps=10,
           learning_rate=0.001, loss_function=losses.mean_squared_error)

    # After watching the plot where train and valid have to converge (the reconstruction error)
    # we look if it is enough low
    input_tensor = Input(shape=(cols,))
    autoencoder = ae.autoencoder(input_tensor)

    prediction_true = autoencoder.predict(valid)
    prediction_test = autoencoder.predict(test_normal.drop(['oferta_id', 'target'], axis=1))
    prediction_anormal = autoencoder.predict(test_anormal.drop(['oferta_id', 'target'], axis=1))

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

    # PLOT ERROR WITHOUT ANOMALIES
    fig = plot.figure()
    ax = fig.add_subplot(111)
    normal_error_df = error_df[(error_df['target'] == 0) & (error_df['reconstruction_error'] < 10)]
    _ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
    plot.savefig(STRING.img_path + 'ae_rc_error_normal.png')
    plot.show()
    plot.close()

    # PLOT ERROR WITH ANOMALIES
    fig = plot.figure()
    ax = fig.add_subplot(111)
    fraud_error_df = error_df[(error_df['target'] == 1) & (error_df['reconstruction_error'])]
    _ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
    plot.savefig(STRING.img_path + 'ae_rc_error_anormal.png')
    plot.show()
    plot.close()

    # RECALL-PRECISION
    precision, recall, th = precision_recall_curve(error_df.target, error_df.reconstruction_error)
    plot.plot(recall, precision, 'b', label='Precision-Recall curve')
    plot.title('Recall vs Precision')
    plot.xlabel('Recall')
    plot.ylabel('Precision')
    plot.savefig(STRING.img_path + 'ae_recall_precision.png')
    plot.show()

    plot.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
    plot.plot(th, recall[1:], 'g', label='Threshold-Recall curve')
    plot.title('Precision-Recall for different threshold values')
    plot.xlabel('Threshold')
    plot.ylabel('Precision-Recall')
    plot.legend(['precision', 'recall'], loc='upper right')
    plot.savefig(STRING.img_path + 'ae_figure_1.png')
    plot.show()

    # OUTLIER DETECTION
    # We define a threshold for the reconstruction error. It will be based on the error plot

    # WE SEPARATE THE SAMPLES 50/50
    mse_test_valid, mse_test = train_test_split(mse_test, test_size=0.5, random_state=10)
    mse_anormal_valid, mse_anormal = train_test_split(mse_anormal, test_size=0.5, random_state=42)
    error_df_valid = pd.concat([mse_test_valid, mse_anormal_valid], axis=0).reset_index(drop=True)
    error_df_test = pd.concat([mse_test, mse_anormal], axis=0).reset_index(drop=True)
    error_df_test['target'] = error_df_test['target'].map(int)
    error_df_valid['target'] = error_df_valid['target'].map(int)

    # ITERATE THROUGH THE SAMPLES
    thresholds = np.linspace(0.001, 2.0, 10000)
    i = 0
    metric_save=[]
    for error in permutations([error_df_valid, error_df_test], 2):
        scores = []
        i += 1
        error_test = error[0]
        error_valid = error[1]

        for threshold in thresholds:
            y_hat = [1 if e > threshold else 0 for e in error_valid.reconstruction_error.values]
            scores.append([
                recall_score(y_pred=y_hat, y_true=error_valid.target.values),
                precision_score(y_pred=y_hat, y_true=error_valid.target.values),
                fbeta_score(y_pred=y_hat, y_true=error_valid.target.values,
                            beta=1, average='binary')
            ])

        scores = np.array(scores)
        threshold = thresholds[scores[:, 2].argmax()]
        print('final Threshold ', threshold)
        predicted = [1 if e > threshold else 0 for e in error_test.reconstruction_error.values]

        precision = precision_score(error_test.target.values, predicted)
        recall = recall_score(error_test.target.values, predicted)
        fbeta = fbeta_score(error_test.target.values, predicted, beta=1)
        print('PRECISION ', precision)
        print('RECALL ', recall)
        print('FBSCORE ', fbeta)

        # Reconstruction Error plot
        error_df = error_df.reset_index(drop=True)
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
        plot.savefig(STRING.img_path + 'ae_final_result_sample_' + str(i) + '.png')
        plot.show()

        # Confussion Matrix
        conf_matrix = confusion_matrix(error_test.target, predicted)
        plot.figure(figsize=(12, 12))
        sns.heatmap(conf_matrix, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot=True,
                    fmt="d", cmap="Blues", cbar=False)
        plot.title("Confusion matrix")
        plot.ylabel('True class')
        plot.xlabel('Predicted class')
        plot.savefig(STRING.img_path + 'ae_confussion_matrix_sample_' + str(i) + '.png')
        plot.show()

        metric_save_i = ['ae', i, threshold, precision, recall, fbeta]
        metric_save.append(metric_save_i)

        # PROBABILITY FILE
        try:
            df_prob = pd.read_csv(STRING.path_db_extra + 'probability_save_' + str(i) + '.csv', sep=';')
            del df_prob['ae']
            df_prob['oferta_id'] = df_prob['oferta_id'].map(int)
            error_test['oferta_id'] = error_test['oferta_id'].map(int)
            df_prob = pd.merge(df_prob, error_test[['oferta_id', 'reconstruction_error']], how='left', on='oferta_id')
            df_prob = df_prob.rename(columns={'reconstruction_error': 'ae'})
            df_prob.to_csv(STRING.path_db_extra + 'probability_save_' + str(i) + '.csv', sep=';', index=False)
            del df_prob
        except FileNotFoundError:
            df_prob = pd.DataFrame(columns=['oferta_id', 'vae', 'ae', 'ert', 'nn', 'ic', 'rc'])
            error_test = error_test.rename(columns={'reconstruction_error': 'vae'})
            df_prob = pd.concat([df_prob, error_test[['oferta_id', 'ae']]], axis=0)
            df_prob.to_csv(STRING.path_db_extra + 'probability_save_' + str(i) + '.csv', sep=';', index=False)


    metric_save = pd.DataFrame(metric_save,
                               columns=['model', 'sample', 'threshold', 'precision', 'recall', 'fbscore'])
    try:
        df = pd.read_csv(STRING.metric_save, sep=';')
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['model', 'sample', 'threshold', 'precision', 'recall', 'fbscore'])

    df = pd.concat([df, metric_save], axis=0)
    df = df.drop_duplicates(subset=['model', 'sample'], keep='last')
    df.to_csv(STRING.metric_save, sep=';', index=False)