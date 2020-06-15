import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix
from lab3_proto import run_preprocessing
import matplotlib.pyplot as plt
import numpy as np
from edit_distance import SequenceMatcher

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
    callback = keras.callbacks.TensorBoard(log_dir="./Graph/" + name, histogram_freq=0, write_graph=True, write_images=True)
    model.summary()

    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation, callbacks=[callback, checkpoint])
    # lr = 0.01, momentum = 0

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

def train():
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

def transcribe(sequence):
    ## I didn't think the transcribe function was working as it was meant to. Made a new one, which I think works..?
# def transcribe(target, output):
    # ret = output[0, :]
    # prev_state = target[0, :]
    #
    # for i in range(1, target.shape[0]):
    #     curr_state = target[i, :]
    #
    #     if (curr_state == prev_state).all():
    #         ret[-1, :] = (ret[-1, :] + output[i, :])/2
    #     else:
    #         ret = np.vstack((ret, output[i, :]))
    #
    #     prev_state = curr_state
    #
    # return ret

    merged_seq = sequence[0, :]
    prev_state = sequence[0, :]

    for i in range(1, sequence.shape[0]):
        curr_state = sequence[i, :]
        if not np.array_equal(curr_state, prev_state) :
            merged_seq = np.vstack((merged_seq, sequence[i, :]))
        prev_state = curr_state

    return merged_seq


if __name__ == "__main__":
    # train()

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

    # final_test is the utterance we're testing our models on
    final_test = run_preprocessing()
    y_sample = np.zeros((len(final_test['target']), 61))
    i = 0
    for x in final_test['target']:
        y_sample[i, x] = 1
        i += 1

    feature_names = ["lmfcc", "mspec", "dlmfcc", "dmspec"]
    state_list = np.load('state_list.npz', allow_pickle=True)['state_list']
    state_list_LUT =    [0,0,0,1,1,1,2,2,2,3,3,3,4,4,
                        4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,
                        11,11,11,12,12,12,13,13,13,14,15,15,15,16,
                        16,16,17,17,17,18,18,18,19,19,19,20,20,20]

    for name in feature_names:
        # TODO: work out the full test % accuracies by this metric using argmax
        model_one_layer = keras.models.load_model("best_model_" + name + "_1layer.h5")
        model_four_layer = keras.models.load_model("best_model_" + name + "_4layer.h5")

        # Part 1: State level FbF
        print("State Level, frame-by-Frame evaluation of " + name)
        print("One Layer Model")
        prediction_one_layer = model_one_layer.predict(final_test[name])
        print("Four Layer Model")
        prediction_four_layer = model_four_layer.predict(final_test[name])

        compressed_y = np.argmax(y_sample, axis=1)
        compressed_pred_four = np.argmax(prediction_four_layer, axis=1)
        compressed_pred_one = np.argmax(prediction_one_layer, axis=1)
        stated_y = [state_list[int(i)] for i in compressed_y]
        stated_pred_four = [state_list[int(i)] for i in compressed_pred_four]
        stated_pred_one = [state_list[int(i)] for i in compressed_pred_one]

        if False:
            four_count = 0
            one_count = 0
            for i in range(324):
                if compressed_pred_four[i] == compressed_y[i]:
                    four_count += 1
                if compressed_pred_one[i] == compressed_y[i]:
                    one_count += 1

            print("Part 1: {:.2f}% correct from 1 layer.".format(one_count/324))
            print("Part 1: {:.2f}% correct from 4 layer.".format(four_count/324))
            full_four = np.argmax(model_four_layer.predict(final_test[name]), axis=1)


        fig, axs = plt.subplots(3)
        axs[0].set_title("Correct output, state level")
        axs[0].pcolormesh(y_sample.T)
        axs[1].set_title(name + " 1 layer")
        axs[1].pcolormesh(prediction_one_layer.T)
        axs[2].set_title(name + " 4 layers")
        axs[2].pcolormesh(prediction_four_layer.T)
        plt.show()

        confusion = confusion_matrix(compressed_y, compressed_pred_four, normalize='true')
        plt.imshow(confusion)
        plt.title("Part 1: Confusion Matrix of " + name + " Predictions vs. Ground Truths")
        plt.xlabel('Posteriors')
        plt.ylabel('Target Values')
        # plt.xticks(range(compressed_pred_four.shape[1]), state_list[:compressed_pred_four.shape[0]])
        # plt.yticks(range(compressed_y.shape[0]), state_list[:compressed_y.shape[0]])
        plt.show()


        # Part 2 Phenome Level FbF
        y_sample_merged = np.zeros((324, 21))
        prediction_one_layer_merged = np.zeros((324, 21))
        prediction_four_layer_merged = np.zeros((324, 21))

        for i in range(324):
            # for j in range(61):
            # if np.round(compressed_y[i], 0) == 1:
            y_sample_merged[i, state_list_LUT[compressed_y[i]]] = 1
            # if np.round(compressed_pred_one[i], 0) == 1:
            prediction_one_layer_merged[i, state_list_LUT[compressed_pred_one[i]]] = 1
            # if np.round(compressed_pred_four[i], 0) == 1:
            prediction_four_layer_merged[i, state_list_LUT[compressed_pred_four[i]]] = 1
        
        fig, axs = plt.subplots(3)
        axs[0].set_title("Correct output, phoneme level")
        axs[0].pcolormesh(y_sample_merged.T)
        axs[1].set_title(name + " 1 layer")
        axs[1].pcolormesh(prediction_one_layer_merged.T)
        axs[2].set_title(name + " 4 layers")
        axs[2].pcolormesh(prediction_four_layer_merged.T)
        plt.show()

        confusion = confusion_matrix(np.argmax(y_sample_merged, axis=1), np.argmax(prediction_four_layer_merged, axis=1),
                                     normalize='true')
        plt.imshow(confusion)
        plt.title("Part 2: Confusion Matrix of " + name + " Merged Predictions vs. Merged Ground Truths")
        plt.xlabel('Posteriors')
        plt.ylabel('Target Values')
        plt.show()

        # Part 3 State Level edit dist
        # y_transcribed = transcribe(y_sample, y_sample)
        # pred_one_layer_trans = transcribe(y_sample, prediction_one_layer)
        # pred_four_layer_trans = transcribe(y_sample, prediction_four_layer)
        y_transcribed = transcribe(y_sample)
        pred_one_layer_trans = transcribe(prediction_one_layer)
        pred_four_layer_trans = transcribe(prediction_four_layer)

        fig, axs = plt.subplots(3)
        axs[0].set_title("Correct output, state level merged")
        axs[0].pcolormesh(y_transcribed.T)
        axs[1].set_title(name + " 1 layer")
        axs[1].pcolormesh(pred_one_layer_trans.T)
        axs[2].set_title(name + " 4 layers")
        axs[2].pcolormesh(pred_four_layer_trans.T)
        plt.show()

        seq1 = SequenceMatcher(y_transcribed, prediction_four_layer_merged)
        distance = seq1.distance() / 324*100

        # TODO: Then measure the Phone Error Rate (PER),
        #  that is the length normalised edit distance between the sequence
        #  of states from the DNN and the correct transcription

        # TODO: Use SequenceMatcher from edit distance to quickly calculate PER


        # Part 4 Phenome Level edit dist
        # y_transcribed_merged = transcribe(y_sample_merged, y_sample_merged)
        # pred_one_layer_trans_merged = transcribe(y_sample_merged, prediction_one_layer_merged)
        # pred_four_layer_trans_merged = transcribe(y_sample_merged, prediction_four_layer_merged)
        y_transcribed_merged = transcribe(y_sample_merged)
        pred_one_layer_trans_merged = transcribe(prediction_one_layer_merged)
        pred_four_layer_trans_merged = transcribe(prediction_four_layer_merged)

        fig, axs = plt.subplots(3)
        axs[0].set_title("Correct output, state level merged")
        axs[0].pcolormesh(y_transcribed_merged.T)
        axs[1].set_title(name + " 1 layer")
        axs[1].pcolormesh(pred_one_layer_trans_merged.T)
        axs[2].set_title(name + " 4 layers")
        axs[2].pcolormesh(pred_four_layer_trans_merged.T)
        plt.show()

        seq2 = SequenceMatcher(y_transcribed_merged, pred_four_layer_trans_merged)
        # distance = seq2.distance()


        # TODO: Label all axes. Write meaningful notes about graphics.
        # TODO: Answer Questions
        print(0)