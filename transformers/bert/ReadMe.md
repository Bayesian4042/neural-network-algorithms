# Introduction

This is a PyTorch implementation of the Masked Language Model (MLM) used to pre-train the BERT model introduced in the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/pdf/1810.04805.pdf

# Masked Language Model
It masks the percentage of tokens at random and trains the model to predict the masked tokens. They mask 15% of tokens by replacing with the token [MASK] token.

The loss is computed on predicting the masked tokens only. 