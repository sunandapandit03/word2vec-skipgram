import argparse
import sys
import torch
import torch.nn.functional as F

def load_embeddings(path):
    checkpoint = torch.load(path, map_location='cpu')
    embeddings = checkpoint['embeddings']
    word2idx = checkpoint['word2idx']

    embeddings = F.normalize(embeddings, dim=1)

    idx2word = {idx: w for w, idx in word2idx.items()}
    return embeddings, word2idx, idx2word


def nearest_word(query_vec, embeddings, idx2word, exclude_indices):
    query_vec = F.normalize(query_vec.unsqueeze(0), dim=1)  
    sims = torch.matmul(embeddings, query_vec.squeeze(0))   

    for idx in exclude_indices:
        sims[idx] = float('-inf')

    best_idx = torch.argmax(sims).item()
    return idx2word[best_idx], sims[best_idx].item()


def answer_analogy(a, b, c, embeddings, word2idx, idx2word):

    for w in (a, b, c):
        if w not in word2idx:
            return None, None

    vec_a = embeddings[word2idx[a]]
    vec_b = embeddings[word2idx[b]]
    vec_c = embeddings[word2idx[c]]

    target = vec_a - vec_b + vec_c

    exclude = {word2idx[a], word2idx[b], word2idx[c]}
    predicted, score = nearest_word(target, embeddings, idx2word, exclude)
    return predicted, score


def load_analogies(path):
    questions = []
    current_category = "unknown"

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith(':'):
                raw_category = line[1:].strip().lower()
                current_category = (
                    "syntactic-" + raw_category if raw_category.startswith("gram")
                    else "semantic-" + raw_category
                )
                continue

            parts = line.lower().split()

            if len(parts) == 4:
                a, b, c, expected = parts
                questions.append((current_category, a, b, c, expected))
            elif len(parts) == 5:
                category, a, b, c, expected = parts
                questions.append((category, a, b, c, expected))
            else:
                print(f"Skipping malformed line: {line}", file=sys.stderr)
                continue

    return questions


def builtin_sample_analogies():
    return [
        ("semantic-capital", "athens", "greece", "oslo", "norway"),
        ("semantic-capital", "paris", "france", "berlin", "germany"),
        ("semantic-gender", "king", "man", "queen", "woman"),
        ("semantic-gender", "brother", "sister", "grandson", "granddaughter"),
        ("syntactic-comparative", "great", "greater", "tough", "tougher"),
        ("syntactic-superlative", "easy", "easiest", "lucky", "luckiest"),
        ("syntactic-plural", "mouse", "mice", "dollar", "dollars"),
        ("syntactic-pasttense", "walking", "walked", "swimming", "swam"),
    ]


def run_evaluation(embeddings, word2idx, idx2word, questions, verbose=False):
    totals = {'semantic': [0, 0], 'syntactic': [0, 0]}  
    overall_correct = 0
    overall_attempted = 0
    skipped_oov = 0

    for category, a, b, c, expected in questions:
        predicted, score = answer_analogy(a, b, c, embeddings, word2idx, idx2word)

        if predicted is None:
            skipped_oov += 1
            continue

        is_correct = (predicted == expected)
        bucket = 'semantic' if category.startswith('semantic') else 'syntactic'

        totals[bucket][1] += 1
        overall_attempted += 1
        if is_correct:
            totals[bucket][0] += 1
            overall_correct += 1

        if verbose:
            mark = 'CORRECT' if is_correct else 'WRONG'
            print(f"[{mark}] {a} - {b} + {c} = {predicted} "
                  f"(expected {expected}, sim={score:.3f})")

    def pct(correct, attempted):
        return 100.0 * correct / attempted if attempted > 0 else 0.0

    print("\n=== Evaluation Results ===")
    print(f"Semantic accuracy:  {pct(*totals['semantic']):.1f}% "
          f"({totals['semantic'][0]}/{totals['semantic'][1]})")
    print(f"Syntactic accuracy: {pct(*totals['syntactic']):.1f}% "
          f"({totals['syntactic'][0]}/{totals['syntactic'][1]})")
    print(f"Total accuracy:     {pct(overall_correct, overall_attempted):.1f}% "
          f"({overall_correct}/{overall_attempted})")
    if skipped_oov > 0:
        print(f"Skipped (out-of-vocabulary words involved): {skipped_oov}")

    return {
        'semantic_accuracy': pct(*totals['semantic']),
        'syntactic_accuracy': pct(*totals['syntactic']),
        'total_accuracy': pct(overall_correct, overall_attempted),
        'attempted': overall_attempted,
        'skipped_oov': skipped_oov,
    }


def nearest_neighbors(word, embeddings, word2idx, idx2word, k=8):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary.")
        return

    query_vec = embeddings[word2idx[word]]
    sims = torch.matmul(embeddings, F.normalize(query_vec.unsqueeze(0), dim=1).squeeze(0))
    top = torch.topk(sims, k + 1) 

    print(f"\nNearest neighbors of '{word}':")
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        neighbor = idx2word[idx]
        if neighbor == word:
            continue
        print(f"  {neighbor:<15} {score:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Skip-gram embeddings on the word analogy task.")
    parser.add_argument('--embeddings', type=str, default='embeddings.pt',
                         help="Path to the saved embeddings checkpoint from train.py")
    parser.add_argument('--analogies', type=str, default=None,
                         help="Path to an analogy question file. If omitted, a small built-in sample set is used.")
    parser.add_argument('--verbose', action='store_true',
                         help="Print every question's prediction, not just the summary.")
    parser.add_argument('--check-words', type=str, nargs='*', default=[],
                         help="Optional: print nearest neighbors for these words as a sanity check.")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings} ...")
    embeddings, word2idx, idx2word = load_embeddings(args.embeddings)
    print(f"Vocabulary size: {len(word2idx)}, embedding dim: {embeddings.shape[1]}")

    for word in args.check_words:
        nearest_neighbors(word, embeddings, word2idx, idx2word)

    if args.analogies:
        questions = load_analogies(args.analogies)
        print(f"\nLoaded {len(questions)} analogy questions from {args.analogies}")
    else:
        questions = builtin_sample_analogies()
        print(f"\nNo --analogies file given. Using built-in sample set "
              f"({len(questions)} questions). This is NOT the official test "
              f"set -- results here are a sanity check, not a paper comparison.")

    run_evaluation(embeddings, word2idx, idx2word, questions, verbose=args.verbose)

if __name__ == '__main__':
    main()
