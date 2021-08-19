# PtychoNN: Deep learning of ptychographic imaging

PtychoNN is a two-headed encoder-decoder network that simultaneously predicts sample amplitude and phase from input diffraction data alone. PtychoNN is 100s of times faster than iterative phase retrieval and can work with as little as **25X less data.** 

Companion repository to the paper at: https://aip.scitation.org/doi/full/10.1063/5.0013065

The strucuture of the network is shown below:
![alt text](./fig1.png)

## Requires:
git lfs

-- Data to train the network is hosted at the git lfs server due to its size

Tensorflow 1.14

Keras 2.2.4

### Tensorflow 2.x version:
Tf2 folder contains notebooks compatible with TF 2.x

-- NOTE: notebooks were run in Google Colab, modify as required for local runtimes

### Mixed precision training

Newest version also contains notebooks that use PyTorch and TF2 mixed precision frameworks for faster training. The original PyTorch version is likely using float64, and the mixed precision mode therefore provides substantial acceleration. It looks like the original TF2 code might be using float32, so the mixed precision code only offers slight improvement in runtime. The test results remain unchanged.
