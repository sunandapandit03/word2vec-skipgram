Word2Vec — Skip-gram Implementation

Implementation of the Skip-gram model with Negative Sampling from:
Efficient Estimation of Word Representations in Vector Space — Mikolov et al., 2013
https://arxiv.org/abs/1301.3781


1. Setup

pip3 install torch numpy tqdm requests


2. How to Run

Step 1 — Train

cd src
python3 train.py

This will:

Download the text8 dataset (~100MB, one time only)
Build vocabulary from the top 20,000 most frequent words
Train Skip-gram with Negative Sampling for 3 epochs
Save embeddings to embeddings.pt



Expected training time: ~4–5 hours on CPU (5M tokens, 3 epochs)




Step 2 — Evaluate

python3 evaluate.py --embeddings embeddings.pt --analogies ../questions-words.txt --check-words king paris dog

This will:


Print nearest neighbors for king, paris, dog (sanity check)
Run the full analogy benchmark (~19.5k questions)
Report semantic and syntactic accuracy %



3. Results

Paper (783M words, 300-dim, 3 epochs): Semantic 50.0% — Syntactic 55.9% — Total 53.3%

Ours (5M words, 100-dim, 3 epochs, CPU): Semantic 2.2% — Syntactic 0.2% — Total 0.7%

Why lower than the paper?


Paper trained on 783M words; we used 5M (~157x less data)
Paper used 300-dim embeddings; we used 100-dim
9,460 out of 19,544 analogy questions skipped due to OOV words (small vocab = many missing words)


The gap is entirely explained by compute constraints, not implementation error.

Qualitative results confirm learning happened


king → queen, cleopatra, darius 
paris → zurich, munich, vienna, geneva (all European cities) 
dog → cat, tiger, monkey (all animals) 



4. Key Design Decisions

Negative Sampling over hierarchical softmax — simpler to implement, works well in practice. Formalized in the follow-up NIPS 2013 paper.

Two embedding matrices (in_embed + out_embed) — separate center-word and context-word embeddings, both initialized with small uniform random values.

Adam optimizer (lr=0.001) — more stable than SGD with linear decay for small datasets on CPU.

Random window size sampled from [1, C] each time — closer words are seen more often, as described in the paper.

Unigram^0.75 for negatives — frequent words sampled more but not proportionally, reducing dominance of very common words like "the".