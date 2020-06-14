import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

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

    for i in range(layers-1):
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

    # LMFCC
    x = np.load('data/normalised features/lmfcc_train_x.npz', allow_pickle=True)['lmfcc_train_x']
    x_val = np.load('data/normalised features/lmfcc_val_x.npz', allow_pickle=True)['lmfcc_val_x']
    x_test = np.load('data/normalised features/lmfcc_test_x.npz', allow_pickle=True)['lmfcc_test_x']

    y = np.load('data/normalised features/train_y.npz', allow_pickle=True)['train_y']
    y_val = np.load('data/normalised features/val_y.npz', allow_pickle=True)['val_y']
    y_test = np.load('data/normalised features/test_y.npz', allow_pickle=True)['test_y']

    x = [x, x_val, x_test]
    y = [y, y_val, y_test]

    model1, best_model1 = network(x, y, epochs = 10, layers = 1, name="lmfcc" + "_1layer")
    model4, best_model4 = network(x, y, epochs = 10, name="lmfcc" + "_4layer")


    # Filterbank
    x = np.load('data/normalised features/mspec_train_x.npz', allow_pickle=True)['mspec_train_x']
    x_val = np.load('data/normalised features/mspec_val_x.npz', allow_pickle=True)['mspec_val_x']
    x_test = np.load('data/normalised features/mspec_test_x.npz', allow_pickle=True)['mspec_test_x']

    x = [x, x_val, x_test]

    model1, best_model1 = network(x, y, epochs = 10, layers = 1, name="mspec" + "_1layer")
    model4, best_model4 = network(x, y, epochs = 10, name="mspec" + "_4layer")


    # Dynamic features LMFCC:
    x = np.load('data/normalised features/dlmfcc_train_x.npz', allow_pickle=True)['dlmfcc_train_x']
    x_val = np.load('data/normalised features/dlmfcc_val_x.npz', allow_pickle=True)['dlmfcc_val_x']
    x_test = np.load('data/normalised features/dlmfcc_test_x.npz', allow_pickle=True)['dlmfcc_test_x']

    x = [x, x_val, x_test]

    model1, best_model1 = network(x, y, epochs = 10, layers = 1, name="dlmfcc" + "_1layer")
    model4, best_model4 = network(x, y, epochs = 10, name="dlmfcc" + "_4layer")

    # Filterbank
    x = np.load('data/normalised features/dmspec_train_x.npz', allow_pickle=True)['dmspec_train_x']
    x_val = np.load('data/normalised features/dmspec_val_x.npz', allow_pickle=True)['dmspec_val_x']
    x_test = np.load('data/normalised features/dmspec_test_x.npz', allow_pickle=True)['dmspec_test_x']

    x = [x, x_val, x_test]

    model1, best_model1 = network(x, y, epochs = 10, layers = 1, name="dmspec" + "_1layer")
    model4, best_model4 = network(x, y, epochs = 10, name="dmspec" + "_4layer")


    # Detailed evaluation:

    """
    Detailed evaluation description:

        1. frame-by-frame at the state level: 
            count the number of frames (time steps) that were correctly classified over the total

        2. frame-by-frame at the phoneme level: 
            same as 1., but merge all states that correspond to the same phoneme, for example ox_0, ox_1 and ox_2 are merged to ox

        3. edit distance at the state level: 
        convert the frame-by-frame sequence of classifications into a transcription by merging all the consequent identical states, 
        for example:  
            ox_0 ox_0 ox_0 ox_1 ox_1 ox_2 ox_2 ox_2 ox_2... becomes ox_0 ox_1 ox_2 .... 
        
        Then measure the Phone Error Rate (PER), that is the length normalised edit distance between the sequence
        of states from the DNN and the correct transcription (that has also been converted this way).

        4. edit distance at the phoneme level: 
            same as 3. but merging the states into phonemes as in 2.
    """

    # load the two models we want:

    # then perform the detailed evaluation:
    # isn't 1. just our normal accuracy? I think so? Agreed.
