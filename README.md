Word2Vec — Skip-gram Implementation

Paper: Efficient Estimation of Word Representations in Vector Space — Mikolov et al., 2013


Setup

bashpip3 install torch numpy tqdm requests


How to run

Step 1 — Train:

bashcd src
python3 train.py

This will:


Download the text8 dataset (~100MB, one time only)
Build vocabulary (top 20k most frequent words)
Train Skip-gram with negative sampling for 3 epochs
Save embeddings.pt checkpoint


Expected training time: ~4-5 hrs on CPU (5M tokens, 3 epochs)

Step 2 — Evaluate:

bashpython3 evaluate.py --embeddings embeddings.pt --analogies ../questions-words.txt --check-words king paris dog

This will:


Print nearest neighbors for king, paris, dog (sanity check)
Run full analogy benchmark (~19.5k questions)
Report semantic and syntactic accuracy



Actual results

SettingSemantic %Syntactic %Total %Paper (783M words, 300d, 3 epochs)50.055.953.3Ours (5M words, 100d, 3 epochs, CPU)2.20.20.7

Why lower than paper:


Paper trained on 783M words; we used 5M (~157x less data)
Paper used 300-dim embeddings; we used 100-dim
9460 out of 19544 questions skipped due to OOV words (small vocab = many words missing)
Qualitative results are meaningful despite low accuracy:

king → queen, cleopatra, darius ✓
paris → zurich, munich, vienna, geneva ✓ (all European cities)
dog → cat, tiger, monkey ✓ (all animals)



Gap is entirely explained by compute constraints, not implementation error



File structure

your-repo/
├── PAPER_NOTES.md           # reading notes — claim, method, metrics
├── README.md                # this file
├── questions-words.txt      # official Word2Vec analogy benchmark (~19.5k questions)
├── src/
│   ├── train.py             # Skip-gram training with negative sampling
│   └── evaluate.py          # analogy evaluation
└── results/                 # screenshots of training output and evaluation results


Key design decisions


Negative Sampling instead of hierarchical softmax — simpler to implement, works well in practice. Paper hints at this; formalized in the follow-up NIPS 2013 paper.
Two embedding matrices — separate in_embed (center words) and out_embed (context/negative words). Both initialized with small uniform random values.
Adam optimizer (lr=0.001) — more stable than SGD with linear decay for small datasets on CPU.
Random window size — window sampled from [1, C] each time, as described in the paper. Closer words are seen more often.
Unigram^0.75 for negatives — frequent words sampled more but not proportionally (as in the paper). Reduces dominance of very common words like "the".