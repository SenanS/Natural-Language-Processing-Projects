# Script to collect examples from the TIDIGITS database, compatible with python3
# To be able to run you need:
# - libsndfile development package (libsndfile1-dev in Ubuntu)
# - pysndfile (https://pypi.python.org/pypi/pysndfile/0.2.1)
# - access to KTH afs cell kth.se
# - access rights to the TIDIGITS database
#
# Usage:
# python3 lab1_data.py
#
# (C) 2015-2018 Giampiero Salvi <giampi@kth.se>
# DT2119 Speech and Speaker Recognition
import numpy as np
import os
import sys
from pysndfile import sndio

for tidigitsroot in ['/home/giampi/corpora/tidigits/disc_4.2.1/tidigits/test/',
                     '/afs/kth.se/misc/csc/dept/tmh/corpora/tidigits/disc_4.2.1/tidigits/test/']:
    if os.path.exists(tidigitsroot):
        break
    else:
        continue
    raise NameError('TIDIGITS root directory not found on system')

genders = ["man", "woman"]
speakers = ["bm", "ew"]
#speakers = ["ae", "ac"]

digits = ["o", "z", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
repetitions = ["a", "b"]

data = []
for idx in range(len(speakers)):
    for digit in digits:
        for repetition in repetitions:
            filename = os.path.join(tidigitsroot, genders[idx], speakers[idx], digit+repetition+'.wav')
            # giving the dtype in sndio.read prevents it from converting the samples to floats
            # and normalizing them from -1.0 to 1.0. Reading the data this way keeps compatibility
            # with other software packages such as HTK and Kaldi.
            sndobj = sndio.read(filename, dtype=np.int16)
            samplingrate = sndobj[1]
            data.append({"filename": filename,
                         "samplingrate": samplingrate,
                         "gender": genders[idx],
                         "speaker": speakers[idx],
                         "digit": digit,
                         "repetition": repetition,
                         "samples": sndobj[0]})

np.savez('lab1_data.npz', data=data)

