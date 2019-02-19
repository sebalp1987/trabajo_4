from keras.utils import to_categorical, plot_model
from keras import layers, Input, regularizers
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard
import STRING


class ResidualConnection(object):
    def __init__(self, n_cols, number_layers=6, node_size=100,
                 prob_dropout=0.1, sparsity_const=10e-2, activation='relu', different_size=None,
                 beta=1, nodes_range='auto'):
        """
        :param n_cols: Number of columns of the dataset
        :param number_layers: Number of total layers in the network (without considering the output node)
        :param node_size: Number of nodes per layer
        :param prob_dropout: proportion to dropout
        :param sparsity_const: Restrict some nodes and not all (as PCA), using regularization strategy
        :param activation: Activation function
        :param different_size: Different sizes in the nodes between root and auxiliars
        """
        self.n_cols = n_cols
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.number_layers = number_layers
        self.node_size = node_size
        self.sparsity_const = sparsity_const
        self.beta = beta
        self.nodes_range = nodes_range

        input_layer = Input(shape=(n_cols,))
        if nodes_range == 'auto':
            nodes_range = range(n_cols - node_size * 2, node_size - 1, -node_size)
        else:
            nodes_range = self.nodes_range
        print(nodes_range)
        # RESIDUAL LAYER
        if sparsity_const is not None:
            residual = layers.Dense(node_size, activation=self.activation, name='residual_layer_' + str(node_size),
                                    activity_regularizer=
                                    regularizers.l1_l2(sparsity_const, sparsity_const))(input_layer)
        else:
            residual = layers.Dense(node_size, activation=self.activation, name='root_layer_' + str(node_size))(
                input_layer)

        y = residual
        print('residual', y)

        # ROOT LAYERS
        if different_size is None:
            for nodes in nodes_range:
                print(nodes)
                if sparsity_const is not None:
                    y = layers.Dense(node_size, activation=self.activation, activity_regularizer=
                    regularizers.l1_l2(sparsity_const, sparsity_const))(y)
                else:
                    y = layers.Dense(node_size, activation=self.activation)(y)
                if self.prob_dropout is not None:
                    y = layers.Dropout(self.prob_dropout)(y)
                print(y)
        else:
            for nodes in nodes_range:
                if sparsity_const is not None:
                    y = residual
                    y = layers.Dense(nodes, activation=self.activation, name='root_layer_' + str(nodes),
                                     activity_regularizer=
                                     regularizers.l1_l2(sparsity_const, sparsity_const))(y)
                else:
                    y = layers.Dense(nodes + different_size, activation=self.activation,
                                     name='root_layer_' + str(nodes))(y)
                if self.prob_dropout is not None:
                    y = layers.Dropout(self.prob_dropout)(y)

            residual = layers.Dense(node_size + different_size)(residual)

        y = layers.add([y, residual])
        output_tensor = layers.Dense(2, activation='softmax')(y)

        self.model = Model(input_layer, output_tensor)
        plot_model(self.model, to_file=STRING.img_path + 'rc_architecture.png', show_shapes=True)
        print(self.model.summary())

    def fit_model(self, predictors, target, learning_rate=0.001, loss_function='categorical_crossentropy', epochs=500,
                  batch_size=500, verbose=True, validation_data=None, validation_split=None,
                  class_weight=None, steps_per_epoch=None, validation_steps=None):
        target = to_categorical(target)
        if validation_data is not None:
            validation_data[1] = to_categorical(validation_data[1].target)
        # tensorboard = TensorBoard(log_dir=STRING.tensorboard_path, histogram_freq=1)
        callback_list = EarlyStopping(patience=2)
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        self.model.fit(x=predictors, y=target, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                       callbacks=[callback_list], verbose=verbose, class_weight=class_weight,
                       validation_split=validation_split, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                       shuffle=False)

    def predict_model(self, x_test):
        return self.model.predict(x_test)
