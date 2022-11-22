# Fastdenoiser
Speech denoiser autoencoder in time domain based on 
https://doi.org/10.48550/arXiv.1806.03185
Skip connection going through LSTM block.
Added antialiasing filter to downsampling module.
For model training https://datashare.ed.ac.uk/handle/10283/2791 dataset was used.

example.py contain a use case of the denoiser class.
Initialization of DenoiseFile class has two arguments.
Model take string 'w' of 'w2' as first for models trained with L2 and L1 loss function on third epoch.
Second argument take recursion depth of the denoiser. More depth decrease noise, but add some artifacts, therefore
it is compromise for every particular case.
