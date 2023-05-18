# Artficial neural network for detecting sleep episodes in ECoG of rats
This is software that was developed to produce the results published in the article [https://arxiv.org/abs/2302.00933] and submited to the journal "Chaos, Solitons and Fractals". It was written in the Python programming language using libraries Tensorflow, Numpy, Matplotlib and Pyedflib.

Folder "tained_ANNs" contains already ANNs with averaged weights (see Table 1 in arxiv:2302.00933) working with different ECoG channel combinations: 1&2, 1&3, 2&3 and all three channels 1&2&3.  
1 - frontal left channel
2 - frontal right channel
3 - occipital channel

The main file "main.py" applying one of four existing ANN models (variable _ANN_file_) to edf-file (variable _edf_file_) with three channels. If there is a markup based on wavelet transform, it can be reffereed via variable _file_markup_
