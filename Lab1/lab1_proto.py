import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
import scipy.signal
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
from lab1_tools import *


# DT2119, Lab1 Feature Extraction

# Function given by the exercise ----------------------------------
# idk why, but this needs to be commented out if I want to run the code, the import from above:
# from lab1_tools import *, does this for me I guess?
# from Lab1.lab1_tools import *


def mspec(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """

    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    # first frame:
    frames = samples[0:winlen].reshape((1, winlen))
    # As we created the first frame, we start at winlen-winshift, go to final sample -1 winlen, and increase our i with winlen - winshift
    for i in range(winlen-winshift, samples.shape[0] - winlen, winlen-winshift):
        # create new frame:
        frames = np.vstack((frames, samples[i:i+winlen].reshape((1, winlen))))
    return frames


def compare(frames1, frames2):
    """
    Short bit of code to compare two 2d arrays with each other,

    If one of the arrays differs, the function prints "Not the same!"
    else prints "Same!"
    """
    for i in range(frames1.shape[0]):
        for j in range(frames1.shape[1]):
            if frames1[i][j] != frames2[i][j]:
                print("Not the same!")
                return
    
    print("Same!")


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    # Digitally filter the input sequence (smoothes a noisy signal)
    # use a = [1] for default a value & b = [1, 0.97] to use for a preemphasis filter coeff of 0.97.

    return scipy.signal.lfilter([1, -p], [1], input)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    #Applying hamming window to each frame for spectral analysis

    hamming_window = scipy.signal.hamming(input.shape[1], sym=0)

    #PLOTS THE HAMMING WINDOW
    # plt.title("Hamming Window plot")
    # plt.plot(hamming_window)
    # plt.show()
    # Sacrifices differences between comparable strength components with similar frequencies
    # to highlight disparate strength components with dissimilar frequencies.
    #(Reduces noise of each speech sample by averaging the signal by frequency)

    return input * hamming_window


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    # I liked wolfram alpha's description of what a power spectrum is as a short explanation (https://mathworld.wolfram.com/PowerSpectrum.html):
    # "For a given signal, the power spectrum gives a plot of the portion of a signal's power (energy per unit time) falling within given frequency bins"

    # discrete fourier transform using scipy:
    fft = scipy.fftpack.fft(input, nfft)
    # square:
    return np.square(np.abs(fft))

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    #tranposed filter bank
    filter_bank = trfbank(samplingrate, input.shape[1]).T

    #PLOTS FILTERS
    # plt.title("Filters")
    # plt.plot(filter_bank)
    # plt.show()
    #Inverse exponential trend to filters

    #Getting the log of the dot product of the input and filter bank
    return np.log(np.dot(input, filter_bank))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    # Explanation from wikipedia: (https://en.wikipedia.org/wiki/Cepstrum)
    # "It serves as a tool to investigate periodic structures within frequency spectra. 
    # Such effects are related i.e. to noticeable echos/reflections in the signal or to the 
    # occurence of harmonic frequencies (partials, overtones)."
    # +
    # "Mathematically it deals with the problem of deconvolution of signals in the frequency space"

    # "Return the Discrete Cosine Transform of arbitrary type sequence x." :
    dct = scipy.fftpack.realtransforms.dct(input)
    # Get the correct number of output cepstra coefficients:
    return dct[:, 0:nceps]


def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lengths of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path through AD

    Note that you only need to define the first output for this exercise.
    """
    N = x.shape[0]
    M = y.shape[0]
    LD = np.zeros((N, M))
    AD = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            LD[i, j] = dist(x[i] - y[j])

    for i in range(N):
        for j in range(M):
            AD[i, j] = LD[i, j] + min(AD[i - 1, j - 1], AD[i - 1, j], AD[i, j - 1])

    d = (AD[N - 1, M - 1]) / (N + M)

    # return d, LD, AD
    return d

if __name__ == "__main__":
    example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
    data = np.load('lab1_data.npz', allow_pickle=True)['data']

    # Frames:
    print("Sampling rate: " + str(example['samplingrate']))
    # 20000 * 0.02 = 400
    # 20000 * 0.01 = 200
    frames_test = enframe(example['samples'], 400, 200)

    # Testing if correct:
    print("enframe")
    compare(frames_test, example['frames'])
    plt.title("Enframe Function")
    plt.pcolormesh(frames_test.T)
    plt.show()
    plt.title("Enframe Example")
    plt.pcolormesh(example['frames'].T)
    plt.show()

    # Preemph:
    preemp_test = preemp(frames_test)

    # Testing if correct:
    print("Preemph")
    compare(preemp_test, example['preemph'])
    plt.title("Pre-emphasis Function")
    plt.pcolormesh(preemp_test.T)
    plt.show()
    plt.title("Pre-emphasis Example")
    plt.pcolormesh(example['preemph'].T)
    plt.show()

    # Hamming Window:
    hamming_test = windowing(preemp_test)

    # Testing if correct:
    print("Hamming")
    plt.title("Hamming Function")
    compare(hamming_test, example['windowed'])
    plt.pcolormesh(hamming_test.T)
    plt.show()
    plt.title("Hamming Example")
    plt.pcolormesh(example['windowed'].T)
    plt.show()

    # Power spectrum:
    spec_test = powerSpectrum(hamming_test, 512)
    
    # Testing if correct:
    print("Power Spectrum")
    compare(spec_test, example['spec'])
    plt.title("Power Spectrum Function")
    plt.pcolormesh(spec_test.T)
    plt.show()
    plt.title("Power Spectrum Example")
    plt.pcolormesh(example['spec'].T)
    plt.show()

    # Mel-filterbank:
    mel_test = logMelSpectrum(spec_test, example['samplingrate'])

    # Testing if correct:
    print("Log Mel Spectrum")
    plt.title("Log Mel Spectrum Function")
    compare(mel_test, example['mspec'])
    plt.pcolormesh(mel_test.T)
    plt.show()
    plt.title("Log Mel Spectrum Example")
    plt.pcolormesh(example['mspec'].T)
    plt.show()

    # Cepstrals:
    cepstrals_test = cepstrum(mel_test, 13)

    # Testing if correct:
    print("Cepstrals")
    plt.title("Cepstrals Function")
    compare(cepstrals_test, example['mfcc'])
    plt.pcolormesh(cepstrals_test.T)
    plt.show()
    plt.title("Cepstrals Example")
    plt.pcolormesh(example['mfcc'].T)
    plt.show()

    # LMFCC:
    lifter_test = lifter(cepstrals_test, 22)

    # Testing if correct:
    print("Lifter")
    plt.title("Lifter Function")
    compare(lifter_test, example['lmfcc'])
    plt.pcolormesh(lifter_test.T)
    plt.show()
    plt.title("Lifter Example")
    plt.pcolormesh(example['lmfcc'].T)
    plt.show()

    #Gets the LMFCC for the 0th entry in data
    MFCC_concat_array = mfcc(data[0]['samples'])
    MSPEC_concat_array = mspec(data[0]['samples'])
    plt.title("Data - Lifter")
    plt.pcolormesh(MFCC_concat_array)
    plt.show()


    # Section 5 - Concatenation & Correlation

    # Contains the 44 processed features
    data_features = []
    for j in range(44):
        data_features.append(mfcc(data[j]['samples']))

    for i in range(1, 44):
        MFCC_concat_array = np.concatenate((MFCC_concat_array, data_features[i]), axis=0)

        MSPEC_computed_data = mspec(data[i]['samples'])
        MSPEC_concat_array = np.concatenate((MSPEC_concat_array, MSPEC_computed_data), axis=0)

    MFCC_correlation_coeff = np.corrcoef(MFCC_concat_array, rowvar=False)
    MSPEC_correlation_coeff = np.corrcoef(MSPEC_concat_array, rowvar=False)

    plt.pcolormesh(MFCC_correlation_coeff)
    plt.title("MFCC correlation")
    plt.show()
    plt.pcolormesh(MSPEC_correlation_coeff)
    plt.title("MSPEC correlation")
    plt.show()



    # Section 6 - Gaussian mixture model: 
    components = [4, 8, 16, 32]

    for comp in components:
        # use sk learns GMM to create gaussian mixture model:
        gmm = GaussianMixture(comp, verbose=1)


        # #I think all of this (the following 11 lines) is done in section 5
        # # Get all training mfcc's:
        # all_mfccs = []
        # for i in range(data.size):
        #     sample = data[i]['samples']
        #     single_mfcc = mfcc(sample)
        #     all_mfccs.append(single_mfcc)
        #
        # # Get first mfcc and make it an array:
        # features = np.matrix(all_mfccs[0])
        # # Concatenate all other mfccs
        # for i in range(1, len(all_mfccs)):
        #     features = np.concatenate((features, all_mfccs[i]))


        # Get first mfcc and make it an array:
        features = np.matrix(data_features[0])
        # Concatenate all other mfccs
        for i in range(1, len(data_features)):
            features = np.concatenate((features, data_features[i]))

        # fit using all data:
        gmm.fit(features)

        # These are utterances of the same word; "seven"
        sevens = [data_features[16], data_features[17], data_features[38], data_features[39]]

        post = []
        for seven in sevens:
            post.append(gmm.predict_proba(seven))

        for i in range(4):
            plt.pcolormesh(post[i].T)
            plt.title("GMM posterior for utterance seven, comp = " + str(comp) + " post nr = " + str(i))
            plt.show()


    # Section 7 - Utterance Comparison

    distances = np.zeros((44, 44))

    for i in range(44):
        print(i)
        for j in range(44):
            # or np.linalg.norm, distance.Euclidean
            distances[i, j] = dtw(data_features[i], data_features[j], np.linalg.norm)

    plt.title("Distances between utterances")
    plt.pcolormesh(distances)
    plt.show()

    labels = tidigit2labels(data)
    link = linkage(distances, method="complete")
    dendrogram(link, labels=labels)
    plt.show()


