import os

import numpy as np
from prondict import prondict
from lab1_proto import *
from lab2_proto import *
from lab2_tools import *
from lab3_tools import *
import pysndfile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils

# np.seterr(divide='ignore')
np.seterr(divide='ignore', invalid='ignore')


def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
        wordList: list of word symbols
        pronDict: pronunciation dictionary. The keys correspond to words in wordList
        addSilence: if True, add initial and final silence
        addShortPause: if True, add short pause model "sp" at end of each word
    Output:
        list of phone symbols
    """
    
    phone_list = []
    for digit in wordList:
        if addShortPause:
            phone_list = phone_list + pronDict[digit] + ["sp"]
        else:
            phone_list = phone_list + pronDict[digit]

    if addSilence:
        phone_list = ["sil"] + phone_list + ["sil"]

    return phone_list

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

   Args:
      lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
            computed the same way as for the training of phoneHMMs
      phoneHMMs: set of phonetic Gaussian HMM models
      phoneTrans: list of phonetic symbols to be aligned including initial and
                  final silence

   Returns:
      list of strings in the form phoneme_index specifying, for each time step
      the state from phoneHMMs corresponding to the viterbi path.
   """
    HMM = concatHMMs(phoneHMMs, phoneTrans)

    # given by assignement:
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans \
                  for stateid in range(phoneHMMs[phone]['means'].shape[0])]

    data_log_lik = log_multivariate_normal_density_diag(
        lmfcc, HMM["means"], HMM["covars"])
    viterbi_loglik, viterbi_path = viterbi(data_log_lik,
                                           np.log(HMM["startprob"]),
                                           np.log(HMM["transmat"]))

    result = []
    # for i in viterbi_path:
    #     result = result + np.int64(stateTrans[i])
    result = [stateTrans[i] for i in viterbi_path.astype(np.int64)]

    return result, viterbi_path

def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

   Args:
      hmmmodels: list of dictionaries with the following keys:
         name: phonetic or word symbol corresponding to the model
         startprob: M+1 array with priori probability of state
         transmat: (M+1)x(M+1) transition matrix
         means: MxD array of mean vectors
         covars: MxD array of variances
      namelist: list of model names that we want to combine, if None,
               all the models in hmmmodels are used

   D is the dimension of the feature vectors
   M is the number of emitting states in each HMM model (could be
   different in each model)

   Output
      combinedhmm: dictionary with the same keys as the input but
                  combined models
      stateMap: map between states in combinedhmm and states in the
               input models.

   Examples:
      phoneLoop = hmmLoop(phoneHMMs)
      wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
   """

    # didn't we already do this in lab 2?
    # We'll have to double check


def extractFeatures():
    phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
    
    traindata = []
    for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                # Feature extraction
                lmfcc = mfcc(samples)
                mspec_val = mspec(samples, samplingrate=samplingrate)
                
                # Forced alignement:
                wordTrans = list(path2info(filename)[2])
                phoneTrans = words2phones(wordTrans, prondict)
                targets, path = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
                targets = [stateList.index(t) for t in targets]

                traindata.append({'filename': filename, 'lmfcc': lmfcc,
                                  'mspec': mspec_val, 'targets': targets})

    testdata = []
    for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                # your code for feature extraction and forced alignment
                lmfcc = mfcc(samples)
                mspec_val = mspec(samples, samplingrate=samplingrate)

                # Forced alignement:
                wordTrans = list(path2info(filename)[2])
                phoneTrans = words2phones(wordTrans, prondict)
                targets, path = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
                targets = [stateList.index(t) for t in targets]

                testdata.append({'filename': filename, 'lmfcc': lmfcc,
                                  'mspec': mspec_val, 'targets': targets})

    print("Feature extraction performed, saving...")
    np.savez('testdata.npz', testdata=testdata)
    np.savez('traindata.npz', traindata=traindata)
    print("Feature extraction saved.")

def train_val_split(train_data):
    speakerIDs = ['ae', 'aj', 'al', 'aw', 'bd', 'ac', 'ag', 'ai', 'an', 'bh', 'bi']
    val_data = []
    train_data_split = []
    for i in tqdm(range(len(train_data))):
        gender, speakerID, digits, repetition = path2info(train_data[i]['filename'])
        if speakerID in speakerIDs:
            val_data.append(train_data[i])
        else:
            train_data_split.append(train_data[i])
    np.savez('data/train_data_split.npz', train_data_split=train_data_split)
    np.savez('data/val_data.npz', val_data=val_data)


def regular_features(dataset):
    # LMFCC dimension = 13 wide
    dim_LMFCC = train_data[0]['lmfcc'].shape[1]
    # MSPEC dimension = 40 wide
    dim_MSPEC = train_data[0]['mspec'].shape[1]
    # N = the sum of the sizes of each LMFCC, MSPEC or Targets
    N = sum((x['lmfcc']).shape[0] for x in dataset)

    lmfcc_acoustic_context = np.zeros((N, dim_LMFCC))
    mspec_acoustic_context = np.zeros((N, dim_MSPEC))

    i = 0
    for utterance in tqdm(dataset):
        iterations = len(utterance['targets'])
        for n in range(iterations):
            lmfcc_acoustic_context[i, :] = utterance['lmfcc'][n, :]
            mspec_acoustic_context[i, :] = utterance['mspec'][n, :]
            i += 1
    return lmfcc_acoustic_context, mspec_acoustic_context

def dynamic_features(dataset):
    # (LMFCC dimension = 13) * 7, to get a stack of 7
    dim_LMFCC = train_data[0]['lmfcc'].shape[1] * 7
    # (MSPEC dimension = 40) * 7, to get a stack of 7
    dim_MSPEC = train_data[0]['mspec'].shape[1] * 7
    # N = the sum of the sizes of each LMFCC, MSPEC or Targets
    N = sum((x['lmfcc']).shape[0] for x in dataset)

    lmfcc_acoustic_context = np.zeros((N, dim_LMFCC))
    mspec_acoustic_context = np.zeros((N, dim_MSPEC))
    targets = []

    order = np.array([
        [3, 2, 1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3, 4, 5],
        [3, 2, 1, 0, 1, 2, 3],
        [4, 3, 2, 1, 0, 1, 2],
        [5, 4, 3, 2, 1, 0, 1]
    ])

    i = 0
    for utterance in tqdm(dataset):
        iterations = len(utterance['targets'])
        for n in range(iterations):
            if n < 3:
                lmfcc_acoustic_context[i, :] = np.hstack(utterance['lmfcc'][order[n], :])
                mspec_acoustic_context[i, :] = np.hstack(utterance['mspec'][order[n], :])

            elif n > iterations - 4:
                lmfcc_acoustic_context[i, :] = np.hstack(utterance['lmfcc'][n - order[(iterations - n) + 2], :])
                mspec_acoustic_context[i, :] = np.hstack(utterance['mspec'][n - order[(iterations - n) + 2], :])

            else:
                # Else just look at the 3 values either side of the current
                lmfcc_acoustic_context[i, :] = np.hstack(utterance['lmfcc'][np.arange(n - 3, n + 4), :])
                mspec_acoustic_context[i, :] = np.hstack(utterance['mspec'][np.arange(n - 3, n + 4), :])
            i += 1
        targets += utterance['targets']
    return lmfcc_acoustic_context, mspec_acoustic_context, targets

def create_features(train, val, test):
    # Saving all features.

    lmfcc_train_x, mspec_train_x, = regular_features(train)
    lmfcc_val_x, mspec_val_x = regular_features(val)
    lmfcc_test_x, mspec_test_x = regular_features(test)

    np.savez('data/features/lmfcc_train_x.npz', lmfcc_train_x=lmfcc_train_x)
    np.savez('data/features/mspec_train_x.npz', mspec_train_x=mspec_train_x)
    np.savez('data/features/lmfcc_val_x.npz', lmfcc_val_x=lmfcc_val_x)
    np.savez('data/features/mspec_val_x.npz', mspec_val_x=mspec_val_x)
    np.savez('data/features/lmfcc_test_x.npz', lmfcc_test_x=lmfcc_test_x)
    np.savez('data/features/mspec_test_x.npz', mspec_test_x=mspec_test_x)

    dlmfcc_train_x, dmspec_train_x, train_y = dynamic_features(train)
    dlmfcc_val_x, dmspec_val_x, val_y = dynamic_features(val)
    dlmfcc_test_x, dmspec_test_x, test_y = dynamic_features(test)

    np.savez('data/features/dlmfcc_train_x.npz', dlmfcc_train_x=dlmfcc_train_x)
    np.savez('data/features/dmspec_train_x.npz', dmspec_train_x=dmspec_train_x)
    np.savez('data/features/train_y.npz', train_y=train_y)
    np.savez('data/features/dlmfcc_val_x.npz', dlmfcc_val_x=dlmfcc_val_x)
    np.savez('data/features/dmspec_val_x.npz', dmspec_val_x=dmspec_val_x)
    np.savez('data/features/val_y.npz', val_y=val_y)
    np.savez('data/features/dlmfcc_test_x.npz', dlmfcc_test_x=dlmfcc_test_x)
    np.savez('data/features/dmspec_test_x.npz', dmspec_test_x=dmspec_test_x)
    np.savez('data/features/test_y.npz', test_y=test_y)


def standardise_features(train_x, test_x, val_x):
    #Normalise each set of feature vectors
    scalar = StandardScaler().fit(train_x)

    normal_train_x = scalar.transform(train_x)
    normal_val_x = scalar.transform(val_x)
    normal_test_x = scalar.transform(test_x)

    return normal_train_x.astype('float32'), normal_test_x.astype('float32'), normal_val_x.astype('float32')

def standardise_targets(train_y, test_y, val_y, state_list):

    for data in [train_y, test_y, val_y]:
        for idx, state in enumerate(data):
            data[idx] = state_list.index(state)

    return np_utils.to_categorical(train_y, len(state_list)), np_utils.to_categorical(test_y, len(state_list)), np_utils.to_categorical(val_y, len(state_list))

def load_and_standardise(state_list):
    #Load in, standardise and save each feature and target

    # lmfcc_train_x = np.load('data/features/lmfcc_train_x.npz', allow_pickle=True)['lmfcc_train_x']
    # lmfcc_test_x = np.load('data/features/lmfcc_test_x.npz', allow_pickle=True)['lmfcc_test_x']
    # lmfcc_val_x = np.load('data/features/lmfcc_val_x.npz', allow_pickle=True)['lmfcc_val_x']
    # lmfcc_train_x, lmfcc_test_x, lmfcc_val_x = standardise_features(lmfcc_train_x, lmfcc_test_x, lmfcc_val_x)
    # np.savez('data/normalised features/lmfcc_train_x.npz', lmfcc_train_x=lmfcc_train_x)
    # np.savez('data/normalised features/lmfcc_test_x.npz', lmfcc_test_x=lmfcc_test_x)
    # np.savez('data/normalised features/lmfcc_val_x.npz', lmfcc_val_x=lmfcc_val_x)
    #
    # dlmfcc_train_x = np.load('data/features/dlmfcc_train_x.npz', allow_pickle=True)['dlmfcc_train_x']
    # dlmfcc_test_x = np.load('data/features/dlmfcc_test_x.npz', allow_pickle=True)['dlmfcc_test_x']
    # dlmfcc_val_x = np.load('data/features/dlmfcc_val_x.npz', allow_pickle=True)['dlmfcc_val_x']
    # dlmfcc_train_x, dlmfcc_test_x, dlmfcc_val_x = standardise_features(dlmfcc_train_x, dlmfcc_test_x, dlmfcc_val_x)
    # np.savez('data/normalised features/dlmfcc_train_x.npz', dlmfcc_train_x=dlmfcc_train_x)
    # np.savez('data/normalised features/dlmfcc_test_x.npz', dlmfcc_test_x=dlmfcc_test_x)
    # np.savez('data/normalised features/dlmfcc_val_x.npz', dlmfcc_val_x=dlmfcc_val_x)
    #
    # mspec_train_x = np.load('data/features/mspec_train_x.npz', allow_pickle=True)['mspec_train_x']
    # mspec_test_x = np.load('data/features/mspec_test_x.npz', allow_pickle=True)['mspec_test_x']
    # mspec_val_x = np.load('data/features/mspec_val_x.npz', allow_pickle=True)['mspec_val_x']
    # mspec_train_x, mspec_test_x, mspec_val_x = standardise_features(mspec_train_x, mspec_test_x, mspec_val_x)
    # np.savez('data/normalised features/mspec_train_x.npz', mspec_train_x=mspec_train_x)
    # np.savez('data/normalised features/mspec_test_x.npz', mspec_test_x=mspec_test_x)
    # np.savez('data/normalised features/mspec_val_x.npz', mspec_val_x=mspec_val_x)

    # dmspec_train_x, dmspec_test_x, dmspec_val_x = standardise_features(dmspec_train_x, dmspec_test_x, dmspec_val_x)
    # np.savez('data/normalised features/dmspec_train_x.npz', dmspec_train_x=dmspec_train_x)
    # np.savez('data/normalised features/dmspec_test_x.npz', dmspec_test_x=dmspec_test_x)
    # np.savez('data/normalised features/dmspec_val_x.npz', dmspec_val_x=dmspec_val_x)

    # dmspec_train_x = np.load('data/features/dmspec_train_x.npz', allow_pickle=True)['dmspec_train_x']
    # scalar = StandardScaler().fit(dmspec_train_x)
    # normal_train_x = scalar.transform(dmspec_train_x)
    # np.savez('data/normalised features/dmspec_train_x.npz', dmspec_train_x=normal_train_x.astype('float32'))
    # dmspec_val_x = np.load('data/features/dmspec_val_x.npz', allow_pickle=True)['dmspec_val_x']
    # normal_val_x = scalar.transform(dmspec_val_x)
    # np.savez('data/normalised features/dmspec_val_x.npz', dmspec_val_x=normal_val_x.astype('float32'))
    # dmspec_test_x = np.load('data/features/dmspec_test_x.npz', allow_pickle=True)['dmspec_test_x']
    # normal_test_x = scalar.transform(dmspec_test_x)
    # np.savez('data/normalised features/dmspec_test_x.npz', dmspec_test_x=normal_test_x.astype('float32'))

    train_y = np.load('data/features/train_y.npz', allow_pickle=True)['train_y']
    test_y = np.load('data/features/test_y.npz', allow_pickle=True)['test_y']
    val_y = np.load('data/features/val_y.npz', allow_pickle=True)['val_y']
    train_y, test_y, val_y = standardise_targets(train_y, test_y, val_y, state_list)
    np.savez('data/normalised features/train_y.npz', train_y=train_y)
    np.savez('data/normalised features/test_y.npz', test_y=test_y)
    np.savez('data/normalised features/val_y.npz', val_y=val_y)


if __name__ == "__main__":
    ##                                      4.1 Load all possible Phones & their states.                                        ##

    phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

    ## Maybe save this stateList to a file, to preserve stability.

    ##Loading examples
    example = np.load('lab3_example.npz', allow_pickle=True)['example'].item()

    ##                                      4.2 Forcefully aligning transcripts of data                                     ##

    # This is done by concatting HMM of utterance & using viterbi to find best path
    # filename = '../tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    print("Running tests for each function against example data:\n")
    filename = 'z43a.wav'
    samples, samplingrate = loadAudio(filename)
    np.testing.assert_almost_equal(samples, example['samples'], 6)
    print("\tSample assert passed.")

    # LMFCC test:
    lmfcc = mfcc(samples)
    np.testing.assert_almost_equal(lmfcc, example['lmfcc'], 6)
    print("\tlmfcc assert passed.")

    # phoneme transcription
    wordTrans = list(path2info(filename)[2])
    phoneTrans = words2phones(wordTrans, prondict, addSilence=True, addShortPause=True)

    test = True
    for i, val in enumerate(phoneTrans):
        test = test and (val == example["phoneTrans"][i])

    if test:
        print("\tphoneTrans is correct.")
    else:
        print("\tphoneTrans is NOT correct!")

    # utteranceHMM test:
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    for i, val in example['utteranceHMM'].items():
        np.testing.assert_almost_equal(utteranceHMM[i], example['utteranceHMM'][i], 6)
    print("\tUtteranceHMM assert passed.")


    #stateTrans test:
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]

    test = True
    for i, val in enumerate(stateTrans):
        test = test and (val == example["stateTrans"][i])

    if test:
        print("\tstateTrans is correct.")
    else:
        print("\tstateTrans is NOT correct!")

    # forcedAlignement test:
    force_aligned = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

    test = True
    for i, val in enumerate(force_aligned[0]):
        test = test and (val == example["viterbiStateTrans"][i])

    if test:
        print("\tforcedAlignment is correct.")
    else:
        print("\tforcedAlignment is NOT correct!")

    # Check success in wavesurfer
    frames2trans(force_aligned[0], outfilename='z43a.lab')

    ##                                      4.3 Feature Extraction                                      ##

    if False:
        print("\n\nStarting feature extraction...")
        extractFeatures()

    ##                                      4.4 Split Data                                      ##
    if False:
        train_data = np.load('data/traindata.npz', allow_pickle=True)['traindata']
        # print(train_data.shape)
        train_val_split(train_data)

    train_data = np.load('data/train_data_split.npz', allow_pickle=True)['train_data_split']
    val_data = np.load('data/val_data.npz', allow_pickle=True)['val_data']
    test_data = np.load('data/testdata.npz', allow_pickle=True)['testdata']
    print("Shape of train data after split: " + str(train_data.shape))
    print("Shape of val data after split: " + str(val_data.shape))
    print("Ratio val/train: " + str(val_data.shape[0] / train_data.shape[0]))

    ##                                      4.5 Dynamic Features                                      ##
    if False:
        print("Individual dynamic and regular features created.")
        create_features(train_data, val_data, test_data)

    ##                                      4.6 Feature Standardisation                                     ##

    load_and_standardise(stateList)
