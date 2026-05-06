# ============================================
# KNOWLEDGE_BASE.PY
# TF-IDF similarity search over Q&A pairs
# Finds the best matching answer for any question
# ============================================

import json
import math
import re


def tokenize(text):
    """
    Clean and split text into words.
    Lowercases, removes punctuation, splits on spaces.

    "What is a P/E ratio?" → ["what", "is", "a", "p", "e", "ratio"]
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def build_tfidf(documents):
    """
    Build a TF-IDF index from a list of text documents.

    TF-IDF stands for Term Frequency - Inverse Document Frequency.
    It measures how important a word is to a document in a collection.

    TF  = how often a word appears in THIS document
    IDF = how rare a word is across ALL documents
          (common words like "the", "is" get low scores)
          (rare specific words like "dividend" get high scores)

    TF-IDF score = TF × IDF
    """
    n_docs = len(documents)
    tokenized = [tokenize(doc) for doc in documents]

    # Count how many documents each word appears in
    doc_freq = {}
    for tokens in tokenized:
        for word in set(tokens):
            doc_freq[word] = doc_freq.get(word, 0) + 1

    # Build TF-IDF vector for each document
    vectors = []
    for tokens in tokenized:
        tf = {}
        for word in tokens:
            tf[word] = tf.get(word, 0) + 1

        # Normalize TF by document length
        doc_len = len(tokens) if tokens else 1
        vector = {}
        for word, count in tf.items():
            tf_score  = count / doc_len
            idf_score = math.log(n_docs / (doc_freq.get(word, 1)))
            vector[word] = tf_score * idf_score

        vectors.append(vector)

    return vectors, doc_freq, n_docs


def cosine_similarity(vec_a, vec_b):
    """
    Measures similarity between two TF-IDF vectors.
    Returns a score between 0 (completely different) and 1 (identical).

    Cosine similarity = dot product / (magnitude_a × magnitude_b)
    """
    # Dot product — sum of products of matching dimensions
    dot = sum(vec_a.get(w, 0) * vec_b.get(w, 0) for w in vec_b)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


class KnowledgeBase:
    def __init__(self, qa_path="data/qa_pairs.json"):
        """
        Load Q&A pairs and build TF-IDF index on startup.
        """
        with open(qa_path, "r") as f:
            self.qa_pairs = json.load(f)

        # Extract all questions
        self.questions = [pair["question"] for pair in self.qa_pairs]
        self.answers   = [pair["answer"]   for pair in self.qa_pairs]

        # Build TF-IDF index over all questions
        self.vectors, self.doc_freq, self.n_docs = build_tfidf(self.questions)

        print(f"Knowledge base loaded: {len(self.qa_pairs)} Q&A pairs")

    def search(self, query, top_k=1, threshold=0.15):
        """
        Find the best matching answer for a query.

        query     = user's question
        top_k     = number of results to return
        threshold = minimum similarity score to return a result
                    below this → no match found

        Returns list of (score, question, answer) tuples.
        """
        # Build TF-IDF vector for the query
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Calculate TF for query
        tf = {}
        for word in query_tokens:
            tf[word] = tf.get(word, 0) + 1

        query_len = len(query_tokens)
        query_vec = {}
        for word, count in tf.items():
            tf_score  = count / query_len
            idf_score = math.log(self.n_docs / (self.doc_freq.get(word, 1) + 1))
            query_vec[word] = tf_score * idf_score

        # Score every question against the query
        scores = []
        for i, q_vec in enumerate(self.vectors):
            score = cosine_similarity(query_vec, q_vec)
            scores.append((score, i))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        # Return top results above threshold
        results = []
        for score, idx in scores[:top_k]:
            if score >= threshold:
                results.append({
                    "score":    round(score, 4),
                    "question": self.questions[idx],
                    "answer":   self.answers[idx]
                })

        return results

    def get_answer(self, query, threshold=0.15):
        """
        Simple wrapper — returns just the best answer string,
        or None if no good match is found.
        """
        results = self.search(query, top_k=1, threshold=threshold)
        if results:
            return results[0]["answer"], results[0]["score"]
        return None, 0.0


# ============================================
# TEST — run this file directly
# ============================================
if __name__ == "__main__":
    kb = KnowledgeBase()

    test_queries = [
        "What is a stock?",
        "explain dividends to me",
        "how does inflation affect the market",
        "what is P/E ratio",
        "tell me about ETFs",
        "how do I start investing",
        "what is a bear market",
        "cryptocurrency prices",           # not in knowledge base
        "what should I eat for lunch",     # completely unrelated
    ]

    print("\n=== KNOWLEDGE BASE SEARCH TEST ===\n")

    for query in test_queries:
        answer, score = kb.get_answer(query)
        print(f"Query: '{query}'")
        print(f"Score: {score:.4f}")
        if answer:
            print(f"Answer: {answer[:80]}...")
        else:
            print("Answer: No match found — will fall back to LLM")
        print()