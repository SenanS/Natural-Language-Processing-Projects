import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


def network(data, labels, epochs=20, batch_size=256, layers=4, name="test"):

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
    checkpoint = keras.callbacks.ModelCheckpoint("best_model_" + name + ".h5", verbose=1, monitor='val_loss', save_best_only=True, mode='min')
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
    saved_model = keras.models.load_model("best_model_" + name + ".h5")

    accuracy = saved_model.evaluate(x_test, y_test)
    print("Final score for the best saved model: " + str(accuracy))
    for i in range(len(accuracy)):
        print(model.metrics_names[i], ": ", accuracy[i])


    return model, saved_model

if __name__ == "__main__":
    # Load data, then use dynamic features + normalization, then call NN

    """
    Specification of tests: (we can do only the 1 layer and 4 layers)

        1. input: liftered MFCCs, one to four hidden layers of size 256, rectified linear units
        2. input: filterbank features, one to four hidden layers of size 256, rectified linear units
        3. same as 1. but with dynamic features as explained in Section 4.5
        4. same as 2. but with dynamic features as explained in Section 4.5
        Note the evolution of the loss function and the accuracy of the model f

    """
    x_all = []
    y_all = []

    # LMFCC
    x_all.append([x_lmfcc, x_lmfxx_val, x_lmfcc_test])
    y_all.append([y_lmfcc, y_lmfxx_val, y_lmfcc_test])

    # Filterbank
    x_all.append([x_filter, x_filter_val, x_filter_test])
    y_all.append([y_filter, y_filter_val, y_filter_test])

    # Dynamic features LMFCC:
    x_all.append([x_lmfcc_dyn, x_lmfxx_val_dyn, x_lmfcc_test_dyn])
    y_all.append([y_lmfcc_dyn, y_lmfxx_val_dyn, y_lmfcc_test_dyn])

    # Filterbank
    x_all.append([x_filter_dyn, x_filter_val_dyn, x_filter_test_dyn])
    y_all.append([y_filter_dyn, y_filter_val_dyn, y_filter_test_dyn])

    # idk; maybe save the best models? might be worth something....
    for i in range(len(x_all)):
        name = ""
        if i == 0:
            name = "lmfcc"
        elif i == 1:
            name = "filterbank"
        elif i == 2:
            name = "lmfcc_dyn"
        else:
            name = "filterbank_dyn"
        
        model1, best_model1 = network(x_all[i], y_all[i], epochs = 10, layers = 1, name=name + "_1layer")
        model4, best_model4 = network(x_all[i], y_all[i], epochs = 10, name=name + "_4layer")
