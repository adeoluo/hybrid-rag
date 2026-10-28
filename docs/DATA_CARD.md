# Data Card - Hybrid RAG

The project bundles one small, synthetic data asset: the corpus and labeled query set used
for the ablation. There is no training data beyond the corpus the dense retriever fits its
idf on, and nothing is downloaded.

## bundled corpus (`src/hybrid_rag/data/sample.py`)

- **Provenance:** hand-authored for this project. Fourteen short, encyclopedia-style
  passages across distinct topics (biology, physics, history, earth science), including a
  few deliberate lexical distractors that share surface words across topics.
- **Labeled queries:** fifteen natural-language queries, each mapped to the id of the
  passage that answers it, used to compute retrieval metrics.
- **Licensing:** authored by this repository; MIT.
- **Known biases and gaps:** small, English, general-knowledge; a demonstration set, not a
  representative or competitive benchmark.

## Ethical considerations

The corpus is synthetic and contains no personal, sensitive, or copyrighted content. Users
indexing their own documents take on the data governance of that content.
