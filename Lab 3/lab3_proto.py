import numpy as np
from prondict import prondict
from lab1_proto import *
from lab2_proto import *
from lab3_tools import *
import pysndfile

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
            phone = phone + pronDict[digit] + ["sp"]
        else:
            phone = phone + pronDict[digit]

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
    for i in viterbi_path:
        result = result + [stateTrans[i]]

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


if __name__ == "__main__":
    ## 4.1 Load all possible Phones & their states.
    phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

    ## Maybe save this stateList to a file, to preserve stability.

    ## setting up isolated pronounciations:
    # isolated = {}
    # for digit in prondict.keys():
    #     isolated[digit] = ['sil'] + prondict[digit] + ['sil']
    #
    # wordHMMs = {}
    # for digit in isolated.keys():
    #     wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])

    ##Loading examples
    example = np.load('lab3_example.npz', allow_pickle=True)['example'].item()

    ## 4.2 Forcefully aligning transcripts of data,
    # This is done by concatting HMM of utterance & using viterbi to find best path
    #TODO: test each function below against example data
    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    samples, samplingrate = loadAudio(filename)
    np.testing.assert_almost_equal(samples, example['samples'], 6)

    lmfcc = mfcc(samples)
    np.testing.assert_almost_equal(lmfcc, example['lmfcc'], 6)

    wordTrans = list(path2info(filename)[2])
    np.testing.assert_almost_equal(wordTrans, example['wordTrans'], 6)

    phoneTrans = words2phones(wordTrans, prondict)
    np.testing.assert_almost_equal(phoneTrans, example['phoneTrans'], 6)

    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    np.testing.assert_almost_equal(utteranceHMM, example['utteranceHMM'], 6)

    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]
    np.testing.assert_almost_equal(stateTrans, example['stateTrans'], 6)

    viterbiStateTrans, viterbiPath = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
    np.testing.assert_almost_equal(viterbiStateTrans, example['viterbiStateTrans'], 6)
    np.testing.assert_almost_equal(viterbiPath, example['viterbiPath'], 6)

    #Check success in wavesurfer
    frames2trans(viterbiStateTrans, outfilename='z43a.lab')