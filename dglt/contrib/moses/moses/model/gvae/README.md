# Introduction

Deep generative models have been wildly successful at learning coherent latent representations for continuous data such as natural images, artwork, and audio. However, generative modeling of discrete data such as arithmetic expressions and molecular structures still poses significant challenges. Crucially, state-of-the-art methods often produce outputs that are not valid. We make the key observation that frequently, discrete data can be represented as a parse tree from a context-free grammar. We propose a variational autoencoder which directly encodes from and decodes to these parse trees, ensuring the generated outputs are always syntactically valid. Surprisingly, we show that not only does our model more often generate valid outputs, it also learns a more coherent latent space in which nearby points decode to similar discrete outputs. We demonstrate the effectiveness of our learned models by showing their improved performance in Bayesian optimization for symbolic regression and molecule generation.[1]

## Reference

[1] [Kusner et al., Grammar Variational Autoencoder. ICML 2017.](http://proceedings.mlr.press/v70/kusner17a.html)
