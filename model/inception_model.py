from keras.utils import to_categorical
from keras import layers, Input, regularizers
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

class InceptionModel(object):
    def __init__(self, n_cols, node_size=100, branch_number=4,
                 prob_dropout=0.1, sparsity_const=10e-5, activation='relu', beta=1):

        self.n_cols = n_cols
        self.node_size = node_size
        self.branch_number = branch_number
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.sparsity_const = sparsity_const
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
                  batch_size=500, verbose=True, validation_data=None, validation_split=None, class_weight=None):

        target = to_categorical(target)
        if validation_data is not None:
            validation_data[1] = to_categorical(validation_data[1].target)

        callback_list = EarlyStopping(patience=2)
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        self.model.fit(x=predictors, y=target, epochs=epochs, batch_size=batch_size,
                       validation_data=validation_data,
                       callbacks=[callback_list], verbose=verbose, validation_split=validation_split,
                       class_weight=class_weight)

    def predict_model(self, x_test):
        return self.model.predict(x_test)