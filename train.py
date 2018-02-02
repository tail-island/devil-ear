import pickle

from data_set           import load_data
from funcy              import identity, juxt, partial, rcompose, repeatedly
from keras.callbacks    import ReduceLROnPlateau
from keras.layers       import Activation, Add, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D, Input
from keras.models       import Model, save_model
from keras.optimizers   import Adam
from keras.regularizers import l2
from utility            import ZeroPadding


def computational_graph(class_size):
    # Utilities.

    def ljuxt(*fs):
        return rcompose(juxt(*fs), list)

    def add():
        return Add()

    def average_pooling():
        return AveragePooling2D()

    def batch_normalization():
        return BatchNormalization()

    def concatenate():
        return Concatenate()

    def conv(filters, kernel_size):
        return Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001), use_bias=False)

    def dense(units):
        return Dense(units, kernel_regularizer=l2(0.0001))

    def global_average_pooling():
        return GlobalAveragePooling2D()

    def max_pooling():
        return MaxPooling2D()

    def relu():
        return Activation('relu')

    def softmax():
        return Activation('softmax')

    def zero_padding(filter_size):
        return ZeroPadding(filter_size)

    # Computational graph.

    def squeeze_net():
        def fire_module(filters_squeeze, filters_expand):
            return rcompose(batch_normalization(),
                            conv(filters_squeeze, 1),
                            batch_normalization(),
                            relu(),
                            ljuxt(conv(filters_expand // 2, 1),
                                  conv(filters_expand // 2, 3)),
                            concatenate())

        def fire_module_with_shortcut(filters_squeeze, filters_expand):
            return rcompose(ljuxt(fire_module(filters_squeeze, filters_expand),
                                  identity),
                            add())

        return rcompose(conv(96, 3),
                        # max_pooling(),
                        fire_module(16, 128),
                        fire_module_with_shortcut(16, 128),
                        fire_module(32, 256),
                        max_pooling(),
                        fire_module_with_shortcut(32, 256),
                        fire_module(48, 384),
                        fire_module_with_shortcut(48, 384),
                        fire_module(64, 512),
                        max_pooling(),
                        fire_module_with_shortcut(64, 512),
                        global_average_pooling())

    def wide_residual_net():
        def residual_unit(filter_size):
            return rcompose(ljuxt(rcompose(batch_normalization(),
                                           conv(filter_size, 3),
                                           batch_normalization(),
                                           relu(),
                                           conv(filter_size, 3),
                                           batch_normalization()),
                                  identity),
                            add())

        def residual_block(filter_size, unit_size):
            return rcompose(zero_padding(filter_size),
                            rcompose(*repeatedly(partial(residual_unit, filter_size), unit_size)))

        return rcompose(conv(16, 3),
                        residual_block(160, 4),
                        average_pooling(),
                        residual_block(320, 4),
                        average_pooling(),
                        residual_block(640, 4),
                        global_average_pooling())

    return rcompose(squeeze_net(),  # wide_residual_net(),
                    dense(256),
                    dense(class_size),
                    softmax())


def main():
    (x_train, y_train), (x_validate, y_validate) = load_data()

    x_mean = -14.631151332833856  # x_train.mean()
    x_std  = 92.12358373202312    # x_train.std()

    x_train, x_validate = map(lambda x: ((x - x_mean) / x_std).reshape(x.shape + (1,)), (x_train, x_validate))
    y_train, y_validate = map(lambda y: y[:, 1],                                        (y_train, y_validate))

    model = Model(*juxt(identity, computational_graph(max(y_validate) + 1))(Input(shape=x_validate.shape[1:])))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])  # 学習が安定しなかったので、学習率を下げてみました。正しい？
    model.summary()

    results = model.fit(x_train, y_train, batch_size=100, epochs=400, validation_data=(x_validate, y_validate), callbacks=[ReduceLROnPlateau(factor=0.5, patience=20, verbose=1)])

    with open('./results/history.pickle', 'wb') as f:
        pickle.dump(results.history, f)

    save_model(model, './results/model.h5')

    del model


if __name__ == '__main__':
    main()
