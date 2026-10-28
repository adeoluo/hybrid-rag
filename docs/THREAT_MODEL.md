# Threat Model - Hybrid RAG

## Assets

- Integrity of the retrieved results and the reported metrics.

## Trust boundaries

- **Documents and queries are untrusted text.** They are tokenized and scored, never
  executed; there is no code path from text to execution.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Index poisoning | Malicious documents added to the corpus | Skewed retrieval | Retrieval is read-only over indexed text; callers control what is indexed |
| Resource exhaustion | Very large corpus or query | Memory / time blowup | In-memory index documented as demonstration-scale; k bounds result size |
| Misleading metrics | Mislabeled query set | Overstated quality | Labeled data is bundled and inspectable; metrics are standard and reproducible |
| Supply-chain compromise | Malicious dependency | Code execution | Zero runtime dependencies; Dependabot and CodeQL in continuous integration |

## Residual risk

The dense retriever cannot detect semantically-phrased adversarial queries that avoid all
corpus vocabulary; this is a capability limit of lexical retrieval, documented in the
README, not a security control.
