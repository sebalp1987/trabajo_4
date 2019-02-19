import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras import layers, Input, regularizers
from keras.models import Model
from sklearn import metrics
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import STRING
from resources.sampling import under_sampling, over_sampling
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class InceptionModel(object):
    def __init__(self, n_cols, node_size=100, branch_number=4,
                 prob_dropout=0.1, sparsity_const=10e-3, activation='relu', sampling=None, beta=1):

        self.n_cols = n_cols
        self.node_size = node_size
        self.branch_number = branch_number
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.sparsity_const = sparsity_const
        self.sampling = sampling
        self.beta = beta

        input_layer = Input(shape=(n_cols,))

        x = input_layer
        branches = []
        for branch in range(0, branch_number + 1, 1):
            if sparsity_const is not None:
                branch_i = layers.Dense(node_size, activation=self.activation, activity_regularizer=
                regularizers.l1(sparsity_const))(x)
            else:
                branch_i = layers.Dense(node_size, activation=self.activation, activity_regularizer=
                regularizers.l1(l1=sparsity_const))(x)
            if self.prob_dropout is not None:
                branch_i = layers.Dropout(self.prob_dropout)(branch_i)

            branches.append(branch_i)

        branches = layers.concatenate(branches, axis=-1)
        output_tensor = layers.Dense(2, activation='softmax')(branches)

        self.model = Model(input_layer, output_tensor)
        print(self.model.summary())

    def fit_model(self, predictors, target, learning_rate=0.001, loss_function='categorical_crossentropy', epochs=500,
                  batch_size=500, verbose=True, callback_list=[], validation_data=None):

        if self.sampling is None:
            pass
        elif self.sampling == 'ALLKNN':
            predictors, target = under_sampling(predictors, target)
        else:
            predictors, target = over_sampling(predictors, target, model=self.sampling)

        target = to_categorical(target.target)
        if validation_data is not None:
            validation_data[1] = to_categorical(validation_data[1].target)

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        history = self.model.fit(x=predictors, y=target, epochs=epochs, batch_size=batch_size,
                                 validation_data=validation_data,
                                 callbacks=callback_list, verbose=verbose)

    def predict_model(self, x_test, y_test):
        y_test = to_categorical(y_test.target)
        predicted = self.model.predict(x_test)
        predicted = predicted[:, 1]

        thresholds = np.linspace(0.001, 1.0, 1000)

        scores = []

        for threshold in thresholds:
            y_hat = (predicted > threshold).astype(float)

            scores.append([
                metrics.recall_score(y_pred=y_hat, y_true=y_test[:, 1].astype(float)),
                metrics.precision_score(y_pred=y_hat, y_true=y_test[:, 1].astype(float)),
                metrics.fbeta_score(y_pred=y_hat, y_true=y_test[:, 1].astype(float),
                                    beta=self.beta)])

        scores = np.array(scores)
        threshold = thresholds[scores[:, 2].argmax()]
        print('final Threshold ', threshold)
        predicted = (predicted > threshold).astype(float)

        print('PRECISION ', metrics.precision_score(y_test[:, 1].astype(float), predicted))
        print('RECALL ', metrics.recall_score(y_test[:, 1].astype(float), predicted))
        print('FBSCORE ', metrics.fbeta_score(y_test[:, 1].astype(float), predicted, beta=self.beta))


if __name__ == '__main__':

    seed = 42
    np.random.seed(seed)

    # LOAD FILE
    train = pd.read_csv(STRING.train, sep=';', encoding='latin1')
    test = pd.read_csv(STRING.test, sep=';', encoding='latin1')
    valid = pd.read_csv(STRING.valid, sep=';', encoding='latin1')

    # VARIANCE REDUCTION
    selection = VarianceThreshold(threshold=0.0)
    selection.fit(train)
    features = selection.get_support(indices=True)
    features = train.columns[features]

    train = train[features]
    test = test[features]
    valid = valid[features]

    # NORMALIZE
    train['CONTROL'] = pd.Series(0, index=train.index)
    test['CONTROL'] = pd.Series(1, index=test.index)
    valid['CONTROL'] = pd.Series(2, index=valid.index)

    normalize = pd.concat([train, test, valid], axis=0)
    for i in normalize.drop(['oferta_id', 'target', 'CONTROL'], axis=1).columns.values.tolist():
        normalize[i] = normalize[i].map(float)
        normalize[i] = StandardScaler().fit_transform(normalize[i].values.reshape(-1, 1))

    train = normalize[normalize['CONTROL'] == 0]
    test = normalize[normalize['CONTROL'] == 1]
    valid = normalize[normalize['CONTROL'] == 2]

    del train['CONTROL']
    del test['CONTROL']
    del valid['CONTROL']

    # X, Y VARIABLES
    train['target'] = train['target'].map(float)
    y_train = train[['target']]
    train = train.drop(['oferta_id', 'target'], axis=1)

    valid = valid.fillna(0)
    y_valid = valid[['target']]
    y_valid['target'] = y_valid['target'].map(float)
    valid = valid.drop(['oferta_id', 'target'], axis=1)

    test = test.fillna(0)
    y_test = test[['target']]
    y_test['target'] = y_test['target'].map(float)
    test = test.drop(['oferta_id', 'target'], axis=1)

    # INPUT COLS
    cols = train.shape[1]

    im = InceptionModel(n_cols=cols, activation='relu', node_size=100, branch_number=6,
                        prob_dropout=0.2, sampling=None)
    early_stopping_monitor = EarlyStopping(patience=2)
    im.fit_model(train, y_train, learning_rate=0.001, loss_function='cosine_proximity', epochs=1000,
                 batch_size=100, verbose=True, callback_list=[early_stopping_monitor], validation_data=[valid, y_valid])

    im.predict_model(x_test=test, y_test=y_test)