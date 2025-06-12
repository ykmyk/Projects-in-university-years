# Install dependencies
!pip install -q sacremoses

import os
import math
import random
import urllib.request
import re
from collections import defaultdict, Counter
from sacremoses import MosesTokenizer, MosesPunctNormalizer
import matplotlib.pyplot as plt
import numpy as np

# 1. Data Gathering: Download example texts for each language
data_sources = {
    'eng': 'https://www.gutenberg.org/files/1342/1342-0.txt',  # Pride and Prejudice
    'ces': 'https://www.gutenberg.org/cache/epub/34225/pg34225.txt',  # Czech book
    'spa': 'https://www.gutenberg.org/files/2000/2000-0.txt'   # Don Quijote
}
os.makedirs("texts", exist_ok=True)

for lang, url in data_sources.items():
    file_path = f"texts/{lang}.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

# 2. Tokenization and Preprocessing
normalizer = MosesPunctNormalizer()
tokenizer = MosesTokenizer()

# Preprocessing and word-level tokenization
def preprocess_text(text):
    text = normalizer.normalize(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# Converts word list into character tokens without special <w> symbols
def words_to_char_tokens(words):
    return [char for word in words for char in word]

# Preprocess and split full text into train/heldout/test character tokens and word tokens
def process_and_split(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    cleaned = preprocess_text(raw)
    n = len(cleaned)
    train_text = cleaned[:int(0.8 * n)]
    heldout_text = cleaned[int(0.8 * n):int(0.9 * n)]
    test_text = cleaned[int(0.9 * n):]

    # Tokenize into words using Moses
    train_words = tokenizer.tokenize(train_text, return_str=False)
    heldout_words = tokenizer.tokenize(heldout_text, return_str=False)
    test_words = tokenizer.tokenize(test_text, return_str=False)

    # Convert to character-level tokens (no <w> special tokens)
    train_tokens = words_to_char_tokens(train_words)
    heldout_tokens = words_to_char_tokens(heldout_words)
    test_tokens = words_to_char_tokens(test_words)

    return {
        'train': train_tokens,
        'heldout': heldout_tokens,
        'test': test_tokens,
        'train_words': train_words,
        'heldout_words': heldout_words,
        'test_words': test_words,
        'raw': raw
    }

language_tokens = {}
language_sizes_bytes = {}

for lang in data_sources:
    file_path = f"texts/{lang}.txt"
    data = process_and_split(file_path)
    language_tokens[lang] = data
    language_sizes_bytes[lang] = os.path.getsize(file_path)

# 3. N-gram counts and probabilities
def count_ngrams(tokens, n):
    counts = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        counts[ngram] += 1
    return counts

def estimate_ngram_probs(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

models = {}

for lang in language_tokens:
    tokens = language_tokens[lang]['train']
    models[lang] = {
        'unigram': count_ngrams(tokens, 1),
        'bigram': count_ngrams(tokens, 2),
        'trigram': count_ngrams(tokens, 3)
    }

# 4. Top 5 trigrams
top_trigrams = {}

for lang in models:
    tri_counts = models[lang]['trigram']
    total = sum(tri_counts.values())
    common = tri_counts.most_common(5)
    top_trigrams[lang] = [(t, c, c/total) for t, c in common]

# 5. Add-alpha smoothing and cross-entropy
def trigram_prob(model, trigram, alpha, vocab_size):
    bigram = trigram[:2]
    trigram_count = model['trigram'].get(trigram, 0)
    bigram_count = model['bigram'].get(bigram, 0)
    return (trigram_count + alpha) / (bigram_count + alpha * vocab_size)

def cross_entropy(model, data, alpha):
    vocab = set(model['unigram'].keys())
    vocab_size = len(vocab)
    trigrams = [tuple(data[i:i+3]) for i in range(len(data)-2)]
    entropy = 0
    for trigram in trigrams:
        p = trigram_prob(model, trigram, alpha, vocab_size)
        entropy += -math.log2(p)
    return entropy / len(trigrams)

initial_alpha = 0.01
initial_entropies = {}

for lang in models:
    test = language_tokens[lang]['test']
    ce = cross_entropy(models[lang], test, initial_alpha)
    initial_entropies[lang] = ce

# 6. Tune alpha using heldout
def tune_alpha(model, heldout):
    best_alpha = None
    best_entropy = float('inf')
    for alpha in np.linspace(0.001, 1, 20):
        ce = cross_entropy(model, heldout, alpha)
        if ce < best_entropy:
            best_entropy = ce
            best_alpha = alpha
    return best_alpha, best_entropy

best_alphas = {}
best_entropies = {}

for lang in models:
    best_alpha, best_entropy = tune_alpha(models[lang], language_tokens[lang]['heldout'])
    best_alphas[lang] = best_alpha
    best_entropies[lang] = cross_entropy(models[lang], language_tokens[lang]['test'], best_alpha)

# 7. Language Identification Function
def identify_language(text):
    text = preprocess_text(text)
    words = tokenizer.tokenize(text, return_str=False)
    tokens = words_to_char_tokens(words)

    scores = []
    for lang, model in models.items():
        alpha = best_alphas[lang]
        ce = cross_entropy(model, tokens, alpha)
        prob = 2 ** (-ce)
        scores.append((prob, lang))
    return sorted(scores, reverse=True)

# 8. Random Sentence Output
random_sentence = "me gusta mucho la comida española."
result = identify_language(random_sentence)

print("Language Identification Result for:", random_sentence)
for prob, lang in result:
    print(f"{lang}: {prob:.6f}")

print(f"Data sizes in bytes: {language_sizes_bytes}")
print("Data sizes in tokens:", {lang: len(language_tokens[lang]['train']) + len(language_tokens[lang]['heldout']) + len(language_tokens[lang]['test']) for lang in language_tokens})

# 9. Word-level OOV Calculation (more realistic)
def word_level_oov_rate(train_words, eval_words):
    train_vocab = set(train_words)
    oov = [w for w in eval_words if w not in train_vocab]
    return 100 * len(oov) / len(eval_words)

print("\nWord-level OOV percentage:")
for lang in language_tokens:
    heldout_oov = word_level_oov_rate(language_tokens[lang]['train_words'], language_tokens[lang]['heldout_words'])
    test_oov = word_level_oov_rate(language_tokens[lang]['train_words'], language_tokens[lang]['test_words'])
    print(f"{lang}: heldout OOV = {heldout_oov:.2f}%, test OOV = {test_oov:.2f}%")

# 10. Print Top Trigrams
print("\nThree frequent trigrams")
for lang, trigrams in top_trigrams.items():
    print(f"{lang}:")
    for t, count, prob in trigrams:
        print(f"{''.join(t)} - {count} - {prob:.6f}")
    print()

print("Initial entropies")
print(initial_entropies)

print("best smoothing parameter")
print(best_alphas)

print("best entropies")
print(best_entropies)
