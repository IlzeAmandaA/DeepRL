from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def neural_net_prey(num_sensors, params, load=''):
    model = Sequential()

    # First layer.
    model.add(Dense(
        params[0], init='lecun_uniform', input_shape=(num_sensors,)
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer.
    # model.add(Dense(params[1], init='lecun_uniform'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(3, init='lecun_uniform')) #actions 3: right, left, forward
    model.add(Activation('linear'))

    rms = RMSprop(lr=0.01)
    adam = Adam(lr=0.01)
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)

    return model