from keras.utils import to_categorical, plot_model
from keras import layers, Input, regularizers
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard
import STRING


class NeuralNetwork(object):

    def __init__(self, n_cols, node_size=[100],
                 prob_dropout=None, sparsity_const=10e-4, activation='relu'):
        """
        :param n_cols: Number of Predictors
        :param node_size: Node Size (but also the len(node_size) determines the number of layers)
        :param prob_dropout: proportion to dropout
        :param sparsity_const: Restrict some nodes and not all (as PCA), using regularization strategy
        :param activation: Activation function
        """

        self.n_cols = n_cols
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.node_size = node_size
        self.sparsity_const = sparsity_const

        input_layer = Input(shape=(n_cols,))

        x = input_layer

        for node in node_size:
            if sparsity_const is not None:
                x = layers.Dense(node, activation=self.activation,
                                 activity_regularizer=regularizers.l1_l2(sparsity_const, sparsity_const))(x)
            else:
                x = layers.Dense(node, activation=self.activation)(x)

            if self.prob_dropout is not None:
                x = layers.Dropout(self.prob_dropout)(x)

        output_tensor = layers.Dense(2, activation='sigmoid')(x)

        self.model = Model(input_layer, output_tensor)
        plot_model(self.model, to_file=STRING.img_path + 'nn_architecture.png', show_shapes=True)
        print(self.model.summary())

    def fit_model(self, predictors, target, learning_rate=0.001, loss_function='categorical_crossentropy', epochs=500,
                  batch_size=500, verbose=True, validation_data=None, validation_split=None, class_weight=None,
                  steps_per_epoch=None, validation_steps=None):
        target = to_categorical(target)
        if validation_data is not None:
            validation_data[1] = to_categorical(validation_data[1].target)
        callback_list = EarlyStopping(patience=2)
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        # tensorboard = TensorBoard(log_dir=STRING.tensorboard_path, histogram_freq=1)
        self.model.fit(x=predictors, y=target, epochs=epochs, batch_size=batch_size,
                       validation_data=validation_data,
                       callbacks=[callback_list], verbose=verbose, validation_split=validation_split,
                       class_weight=class_weight, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    def predict_model(self, x_test):
        return self.model.predict(x_test)

