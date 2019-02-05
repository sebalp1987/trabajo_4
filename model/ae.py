import pandas as pd
import numpy as np
from keras import Input, layers, regularizers, losses
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import STRING
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_curve, recall_score, precision_score, fbeta_score)
import seaborn as sns

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

    def encoded(self, input_layer, sparsity_const=10e-5, change_encode_name=None):
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
                                           regularizers.l1(sparsity_const))(input_layer)
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
        print(autoencoder.summary())

        return autoencoder

    def fit(self, x, x_valid, learning_rate=0.001, loss_function=losses.mean_squared_error, epochs=500,
            batch_size=500, verbose=True, callback_list=[]):
        input_tensor = Input(shape=(self.n_cols,), name='Input')
        autoencoder = DeepAutoencoder.autoencoder(self, input_tensor)
        optimizer = Adam(lr=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss=loss_function)
        history = autoencoder.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callback_list,
                                  shuffle=True, validation_data=x_valid).history

        plot.plot(history['loss'])
        plot.plot(history['val_loss'])
        plot.title('model loss')
        plot.ylabel('loss')
        plot.xlabel('epoch')
        plot.legend(['train', 'valid'], loc='upper right')
        plot.show()


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
        normalize[i] = StandardScaler().fit_transform(normalize[i].values.reshape(-1, 1))

    normal = normalize[normalize['CONTROL'] == 0]
    anormal = normalize[normalize['CONTROL'] == 1]

    del normal['CONTROL']
    del anormal['CONTROL']

    # VARIANCE REDUCTION
    selection = VarianceThreshold(threshold=0.0)
    selection.fit(normal.drop(['oferta_id', 'target'], axis=1))
    features = selection.get_support(indices=True)
    features = list(normal.columns[features]) + ['oferta_id', 'target']

    normal = normal[features]
    test_anormal = anormal[features]

    train, valid, _, _ = train_test_split(normal, normal, test_size=0.30, random_state=42)
    valid, test_normal, _, _ = train_test_split(valid, valid, test_size=len(anormal.index), random_state=42)
    valid = valid.drop(['oferta_id', 'target'], axis=1)

    # INPUT COLS
    cols = train.drop(['oferta_id', 'target'], axis=1).shape[1]

    ae = DeepAutoencoder(n_cols=cols, activation='tanh', prob_dropout=0.2, dimension_node=4, encoding_dim=14)
    early_stopping_monitor = EarlyStopping(patience=2)
    ae.fit(train.drop(['oferta_id', 'target'], axis=1), x_valid=[valid, valid], callback_list=[early_stopping_monitor],
           batch_size=200, epochs=1000,
           learning_rate=0.001)

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

    mse_true = pd.DataFrame(mse_true, columns=['reconstruction_error'])
    mse_test = pd.DataFrame(mse_test, columns=['reconstruction_error'])
    mse_anormal = pd.DataFrame(mse_anormal, columns=['reconstruction_error'])

    mse_true['target'] = pd.Series(0, index=mse_true.index)
    mse_test['target'] = pd.Series(0, index=mse_test.index)
    mse_anormal['target'] = pd.Series(1, index=mse_anormal.index)
    error_df = pd.concat([mse_test, mse_anormal], axis=0)
    print(error_df.describe())

    # We separate the values
    mse_test_valid, mse_test = train_test_split(mse_test, test_size=0.3, random_state=42)
    mse_anormal_valid, mse_anormal = train_test_split(mse_anormal, test_size=0.3, random_state=42)
    error_df_valid = pd.concat([mse_test_valid, mse_anormal_valid], axis=0).reset_index(drop=True)
    error_df_test= pd.concat([mse_test, mse_anormal], axis=0).reset_index(drop=True)
    error_df_test['target'] = error_df_test['target'].map(int)
    error_df_valid['target'] = error_df_valid['target'].map(int)

    # PLOT ERROR WITHOUT ANOMALIES
    fig = plot.figure()
    ax = fig.add_subplot(111)
    normal_error_df = error_df[(error_df['target'] == 0) & (error_df['reconstruction_error'] < 10)]
    _ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
    plot.show()
    plot.close()

    # PLOT ERROR WITH ANOMALIES
    fig = plot.figure()
    ax = fig.add_subplot(111)
    fraud_error_df = error_df[error_df['target'] == 1]
    _ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
    plot.show()
    plot.close()

    # RECALL-PRECISION
    precision, recall, th = precision_recall_curve(error_df_valid.target, error_df_valid.reconstruction_error)
    plot.plot(recall, precision, 'b', label='Precision-Recall curve')
    plot.title('Recall vs Precision')
    plot.xlabel('Recall')
    plot.ylabel('Precision')
    plot.show()

    plot.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
    plot.plot(th, recall[1:], 'g', label='Threshold-Recall curve')
    plot.title('Precision-Recall for different threshold values')
    plot.xlabel('Threshold')
    plot.ylabel('Precision-Recall')
    plot.legend(['precision', 'recall'], loc='upper right')
    plot.show()

    # OUTLIER DETECTION
    # We define a threshold for the reconstruction error. It will be based on the error plot
    thresholds = np.linspace(0.1, 10.0, 200)

    scores = []

    for threshold in thresholds:
        y_hat = [1 if e > threshold else 0 for e in error_df_valid.reconstruction_error.values]

        scores.append([
            recall_score(y_pred=y_hat, y_true=error_df_valid.target.values),
            precision_score(y_pred=y_hat, y_true=error_df_valid.target.values),
            fbeta_score(y_pred=y_hat, y_true=error_df_valid.target.values,
                        beta=1)])

    scores = np.array(scores)
    threshold = thresholds[scores[:, 2].argmax()]
    print('final Threshold ', threshold)
    predicted = [1 if e > threshold else 0 for e in error_df_test.reconstruction_error.values]

    print('PRECISION ', precision_score(error_df_test.target.values, predicted))
    print('RECALL ', recall_score(error_df_test.target.values, predicted))
    print('FBSCORE ', fbeta_score(error_df_test.target.values, predicted, beta=1))

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
    plot.show()

    conf_matrix = confusion_matrix(error_df_test.target, predicted)
    plot.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot=True, fmt="d")
    plot.title("Confusion matrix")
    plot.ylabel('True class')
    plot.xlabel('Predicted class')
    plot.show()
