import numpy as np
import lab2_tools
from prondict import prondict
import matplotlib.pyplot as plt


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    size1 = np.size(hmm1['startprob']) - 1
    size2 = np.size(hmm2['startprob']) - 1

    concat_hmm = {}
    concat_hmm['startprob'] = hmm2['startprob'] * hmm1['startprob'][size1]
    concat_hmm['startprob'] = np.concatenate((hmm1['startprob'][0:size1], concat_hmm['startprob']))

    mul = np.reshape(hmm1['transmat'][0:-1, -1], (size1, 1)) @ np.reshape(hmm2['startprob'], (1, size2+1))
    concat_hmm['transmat'] =  np.concatenate((hmm1['transmat'][0:-1, 0:-1], mul), axis=1)

    tmp = np.concatenate((np.zeros([size2+1, size1]), hmm2['transmat']), axis=1)
    concat_hmm['transmat'] = np.concatenate((concat_hmm['transmat'], tmp), axis=0)

    concat_hmm['means'] = np.concatenate((hmm1['means'], hmm2['means']), axis=0)
    concat_hmm['covars'] = np.concatenate((hmm1['covars'], hmm2['covars']), axis=0)

    return concat_hmm


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat



def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    #TODO: GMMLOGLIK still in testing
    loglik_gmm = 0
    loglik_gmm += lab2_tools.logsumexp(log_emlik[:, :] + np.log(weights))

    gmm = 0
    for i in range(log_emlik.shape[0]):
        gmm += lab2_tools.logsumexp(log_emlik[i, :] + np.log(weights))

    print(gmm - loglik_gmm)


    return loglik_gmm



def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

    N, M = log_emlik.shape

    # Create alpha return matrix, populate with n=0 formula result.
    forward_prob = np.zeros((N, M))

    forward_prob[0, :] = log_startprob[:-1] + log_emlik[0, :]

    for n in range(1, N):
        for j in range(M):
            forward_prob[n, j] = lab2_tools.logsumexp(forward_prob[n-1, :] + log_transmat[:-1, j]) + log_emlik[n, j]

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

    N, M = log_emlik.shape

    # Create zeroed beta return matrix.
    log_beta = np.zeros((N, M))

    #For all other n, populate beta with regular formula result.
    #Start at N-2 &, in increments of -1, finish at 0.
    for n in range(N - 2, -1, -1):
        for j in range(M):
            log_beta[n][j] = lab2_tools.logsumexp(log_beta[n + 1, :] + log_emlik[n + 1, :] + log_transmat[j, :-1])

    return log_beta


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

    N, M = log_emlik.shape

    log_viterbi = np.zeros((N, M))
    viterbi_path = np.zeros(N)
    backtrack_matrix = np.zeros((N, M))

    # Populate viterbi matrix with with n=0 formula result.
    log_viterbi[0][:] = np.add(log_startprob[:-1], log_emlik[0][:])

    # For all other n, populate viterbi with regular recursive formula result.
    for n in range(1, N):
        for j in range(M):
            # Store the highest likelihood and it's index.
            B_n = log_viterbi[n - 1, :] + log_transmat[:-1, j]
            log_viterbi[n, j] = log_emlik[n, j] + np.max(B_n)
            backtrack_matrix[n, j] = np.argmax(B_n)

    # Setup path variable, depending on forceFinalState.
    if forceFinalState:
        viterbi_path[N - 1] = M - 1
    else:
        viterbi_path[N - 1] = np.argmax(log_viterbi[N - 1, :])
    # Go through each column of the matrix backwards to find the route of the highest likelihood.
    for i in range(N - 2, -1, -1):
        viterbi_path[i] = backtrack_matrix[i + 1, int(viterbi_path[i + 1])]

    #Get best score
    viterbi_loglik = np.max(log_viterbi[-1, :])

    return viterbi_loglik, viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    log_gamma = log_alpha + log_beta - lab2_tools.logsumexp(log_alpha[-1,:])
    return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """

    N, D = X.shape
    M = log_gamma.shape[1]

    means  = np.zeros((M, D))
    covars = np.zeros((M, D))

    for i in range(M):
        dot_product = np.dot(X.T, np.exp(log_gamma[:, i]))
        means[i, :] = dot_product / np.sum(np.exp(log_gamma[:, i]))

        C = X.T - means[i,:].reshape((D, 1))

        res = 0
        for j in range(N):
            res = res + np.exp(log_gamma[j, i]) * np.outer(C[:, j], C[:, j])

        covars[i, :] = np.diag(res) / np.sum(np.exp(log_gamma[:, i]))

    covars[covars < varianceFloor] = varianceFloor

    return means, covars


if __name__ == "__main__":

    data = np.load('lab2_data.npz', allow_pickle=True)['data']
    example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()
    # trained on only one single female speaker:
    phoneHMMs_one = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
    # trained on the entire dataset:
    phoneHMMs_all = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()


    # setting up isolated pronounciations:
    isolated = {}
    for digit in prondict.keys():
        isolated[digit] = ['sil'] + prondict[digit] + ['sil']

    wordHMMs = {}
    for digit in isolated.keys():
        wordHMMs[digit] = concatHMMs(phoneHMMs_one, isolated[digit])
    print(list(wordHMMs['o'].keys()))


    #Testing log likelihood function
    o_obsloglik = lab2_tools.log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'],
                                                                  wordHMMs['o']['covars'])

    print("Testing if likelihood is correct: ")
    np.testing.assert_almost_equal(o_obsloglik, example['obsloglik'], 6)
    print("Likelihood is correct.")

    # plotting likelihood functions
    fig, axs = plt.subplots(2)
    axs[0].set_title("Computed \"o\" obsloglik")
    axs[0].pcolormesh(o_obsloglik.T)
    axs[1].set_title("Example \"o\" obsloglik")
    axs[1].pcolormesh(example['obsloglik'].T)
    plt.show()
    # The dark bars in the middle refer to 'ow', while the higher prob light bars
    # refer the 'sil' on either side of the dark.

    # Testing Forward function
    forward_probability = forward(o_obsloglik,
            np.log(wordHMMs['o']["startprob"]),
            np.log(wordHMMs['o']["transmat"]))

    print("Testing if forward probability is ≃ to example: ")
    np.testing.assert_almost_equal(forward_probability, example['logalpha'], 6)
    print("Likelihood is correct.")

    #  plotting forward functions:
    fig, axs = plt.subplots(2)
    axs[0].set_title("Computed \"o\" forward probability")
    axs[0].pcolormesh(forward_probability.T)
    axs[1].set_title("Example \"o\" forward probability")
    axs[1].pcolormesh(example['logalpha'].T)
    plt.show()


    # Testing Backward function
    backward_probability = backward(o_obsloglik,
                                  np.log(wordHMMs['o']["startprob"]),
                                  np.log(wordHMMs['o']["transmat"]))

    print("Testing if backward probability is ≃ to example: ")
    np.testing.assert_almost_equal(backward_probability, example['logbeta'], 6)
    print("Likelihood is correct.")

    # plotting backward functions:
    fig, axs = plt.subplots(2)
    axs[0].set_title("Computed \"o\" backward probability")
    axs[0].pcolormesh(backward_probability.T)
    axs[1].set_title("Example \"o\" backward probability")
    axs[1].pcolormesh(example['logbeta'].T)
    plt.show()


    # Testing Viterbi function
    viterbi_score, viterbi_path = viterbi(o_obsloglik,
                                    np.log(wordHMMs['o']["startprob"]),
                                    np.log(wordHMMs['o']["transmat"]),
                                    False)

    print("Testing if viterbi likelihood is ≃ to example: ")
    np.testing.assert_almost_equal(viterbi_score, example['vloglik'], 6)
    print("Likelihood is correct.")
    print("Testing if viterbi path is ≃ to example: ")
    np.testing.assert_almost_equal(viterbi_path, example['vpath'], 6)
    print("Path is correct.")


    # Testing State Posteriors function
    gamma = statePosteriors(forward_probability, backward_probability)

    print("Testing if State Posteriors is ≃ to example: ")
    np.testing.assert_almost_equal(gamma, example['loggamma'], 6)
    print("Likelihood is correct.")

    # plotting backward functions:
    fig, axs = plt.subplots(2)
    axs[0].set_title("Computed \"o\" State Posteriors")
    axs[0].pcolormesh(gamma.T)
    axs[1].set_title("Example \"o\" State Posteriors")
    axs[1].pcolormesh(example['loggamma'].T)
    plt.show()

    # Testing log likeliood GMM
    #TODO: Test gmmloglik, untested because I don't know where to get the weights

    # gmmloglik(o_obsloglik, )
