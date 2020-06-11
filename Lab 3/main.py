import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


def network(data, labels, epochs=20, batch_size=256, layers=4):

    x, x_val, x_test = data
    y, y_val, y_test = labels

    n_features = x.shape[1]
    n_phones = 61

    # Neural network model:
    model = Sequential()


    # Setting up x layers - tests for how these should be done could be done more extensively given time...
    # Layer 1
    model.add(Dense(256, input_dim=n_features, kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    for x in range(layers-1):
        # Layer 2
        model.add(Dense(256, input_dim=256, kernel_initializer='he_normal'))
        model.add(Activation('relu'))

    # Output layer
    model.add(Dense(n_phones, input_dim=256, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))


    # Define compiler, could use "sgd" instead of "adam":
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "categorical_accuracy"])


    # Setting up training checkpoint, for the best model:
    checkpoint = keras.callbacks.ModelCheckpoint('best_model_test1', verbose=1, monitor='val_loss', save_best_only=True, mode='min')
    validation = (x_val, y_val)

    # Tensorboard, to be able to view training:
    callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.summary()
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation, callbacks=[callback, checkpoint])

    """
    accuracy = model.evaluate(x_test, y_test)

    print("Final accuracy: " + str(accuracy))
    for i in range(len(accuracy)):
        print(model.metrics_names[i], ": ", accuracy[i])

    """

    # Evaluation using the test set:
    saved_model = keras.models.load_model('best_model_test1')

    accuracy = saved_model.evaluate(x_test, y_test)
    print("Final score for the best saved model: " + str(accuracy))
    for i in range(len(accuracy)):
        print(model.metrics_names[i], ": ", accuracy[i])


    return model

if __name__ == "__main__":
    # Load data, then use dynamic features + normalization, then call NN

    network(data[0],data[1], epochs = 5)