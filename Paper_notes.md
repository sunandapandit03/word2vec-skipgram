PAPER_NOTES.md

Paper: Word2Vec — Mikolov et al., 2013


1. What is the paper claiming?

Old models for learning word meaning were slow and needed too much compute.

This paper says: remove the complex parts, keep it simple, train on more data.
Result → better word vectors, trained much faster.

They propose two models: CBOW and Skip-gram.
Skip-gram is the better one (especially for meaning-based tasks) so that's what we implement.

What Skip-gram does in one line:


Give it one word → it learns to predict the words around it.



Do this for the entire corpus millions of times → words with similar meaning end up with similar vectors.

The famous result:


vector(king) - vector(man) + vector(woman) = vector(queen)



This works because the model accidentally learns that "gender" is a direction in vector space.


2. What do we implement?

Skip-gram with Negative Sampling:


Every word is stored as a list of 100 numbers (a vector)
For each word in text, look at nearby words (window = 5)
For each (center word, nearby word) pair → this is a positive example
Pick 5 random unrelated words → these are negative examples
Train the model: score positive pairs high, negative pairs low
Repeat. Vectors slowly improve.


Hyperparameters (ours vs paper):

SettingPaperOursEmbedding size300100Window size105Negative samples55OptimizerSGD, lr=0.025 decayingAdam, lr=0.001Epochs33Training data783M words5M words

We used Adam instead of SGD because SGD with linear decay did not converge on our small dataset.


3. Dataset, metric, baseline

Dataset: Paper used Google News (6 billion words) — too big for us.
We use text8 (~17M words, we use first 5M) — standard smaller substitute.

How we measure quality:
Analogy questions like:


"Paris is to France as Berlin is to ___?"



Compute: vector(Paris) - vector(France) + vector(Berlin) → find nearest word → should be Germany.
Count how many we get right = accuracy %.

Paper's results (Table 3, 640-dim, same training data):

ModelSemantic %Syntactic %RNNLM936NNLM2353CBOW2464Skip-gram5559

Our actual result:

SettingSemantic %Syntactic %Total %Ours (5M words, 100d, 3 epochs)2.20.20.7

Lower than paper due to 157x less training data and smaller embeddings.
Qualitative results confirm learning happened:


king → queen ✓
paris → zurich, munich, vienna ✓
dog → cat, tiger ✓