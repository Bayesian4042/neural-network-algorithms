# What is the problem without attention mechanisms ?
Suppose we want to develop a language translation model that translate text from one language to another. We can't simple translate text word by word due to the grammatical structures in the source and the target language.

To address this issue, it is common to use encoder-decoder framework. The job of encoder is to read the entire text first, and then decode then produces the translated text. 

An RNN is a type of neural network where output form previous steps are fed as inputs to the current step, making them well suited for sequential data like text. In an encoder-decoder RNN, the input text is fed into the encoder, which processes it sequentially. The encoder updates its hidden state at each step, trying to capture the entire meaning of the input sentence in the final hidden state. The decoder then takes this final hidden state to start generating the translated sentence, on word at a time. It also updates its hidden state at each step, which carry the context necessary for the next-word prediction.

[encoder-decoder]("encoder-decoder.png)

The decoder can not access earlier hidden states from the encoder during the decoding phase, it just has last hidden state. The last hidden state encapsulated all relevant information which can lead to a loss of context, especially in complex sentences where dependencies span long distances.

RNN work fine for translating short sentences but don;t work well for longer texts as they don't have direct access to the previous words in the input. Hence, bahdanau attention mechanism for RNNs in 2014 developed which modifies the encoder-decoder RNN such that it selectively access different parts of the input sequence at each decoding step.

In 2017, transformer architecture was proposed with a self attention mechanism inspired by Bhadanau attention mechanism.

Self-attention is a mechanism that allows each position in the input sequence to attend to all the positions in the same sequence when computing the representation of a sequence. 
The purpose to create enriched representations of each element in an input sequence by incorporating infirmation from all other element in an input sequence. This is essential as it needs to understand the relationship and relevance of words in a sentence to each othe. 

