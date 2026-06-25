Word2Vec — Skip-gram Implementation

Paper: Efficient Estimation of Word Representations in Vector Space — Mikolov et al., 2013


1. What is the Paper Claiming?

Older word embedding models were slow and computationally heavy. This paper argues: simplify the architecture, train on more data, get better results.

The authors propose two models: CBOW and Skip-gram.

We implement Skip-gram — it performs better on semantic (meaning-based) tasks.

What Skip-gram does (in one line):

Given one word → predict the surrounding words.



Repeat this over millions of word pairs across the entire corpus → words with similar meanings end up with similar vectors.

The famous result:

vector(king) - vector(man) + vector(woman) ≈ vector(queen)

This works because the model learns that "gender" is a direction in vector space — not explicitly taught, just an emergent pattern.



2. What Do We Implement?

Skip-gram with Negative Sampling (SGNS)

Representation->
Each word = a list of 100 numbers (a vector)

Context window=For each word, look at 5 nearby words

Positive pair=(center word, nearby word) → score should be HIGH

Negative pair=(center word, 5 random unrelated words) → score should be LOW

Training=Repeat across all pairs; vectors slowly improve



Hyperparameters(Ours vs Paper)->

Embedding size of paper=300
Embedding size of ours=100
Window size of paper=10
Window size of ours=5
Negative samples of paper=55O
Negative samples of ours=5
Epochs=same(3)
Training data of paper=783M words
Training data of ours=5M words



Why Adam instead of SGD?

SGD with linear decay didn't converge on our smaller dataset. Adam worked more stably.



3. Dataset, Evaluation & Results

Dataset

Paper used: Google News — 6 billion words (too large for us)
We use: text8 — ~17M words total, first 5M used (standard smaller benchmark)



4. How We Measure Quality — Word Analogy Task

"Paris is to France as Berlin is to ___?"

Compute: vector(Paris) - vector(France) + vector(Berlin) → find nearest word → should be Germany.

We count how many analogies we get right = accuracy %.

Paper's Results (Table 3, 640-dim, same training data)

ModelSemantic %Syntactic %RNNLM936NNLM2353CBOW2464Skip-gram5559



5. Our Results

Semantic=2.2 %

Syntactic=0.7%

Total=0.7%

Why lower? We used 157x less training data and smaller embeddings (100 vs 640 dims). Expected gap.



6. Qualitative Results — Learning Did Happen 

Despite low accuracy scores, the model learned meaningful relationships:

The vectors capture real-world similarity — just not at paper-scale accuracy due to data constraints.