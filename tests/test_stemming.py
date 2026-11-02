"""The Porter stemmer against reference stems from Porter's 1980 paper and NLTK."""

from __future__ import annotations

import pytest

from hybrid_rag.core import Chunk
from hybrid_rag.core.stemming import stem
from hybrid_rag.core.tokenization import tokenize
from hybrid_rag.retrievers import BM25Retriever, TfidfRetriever

# Expected stems produced by NLTK's PorterStemmer(mode=ORIGINAL_ALGORITHM).
CASES = [
    ("caresses", "caress"),
    ("ponies", "poni"),
    ("ties", "ti"),
    ("caress", "caress"),
    ("cats", "cat"),
    ("feed", "feed"),
    ("agreed", "agre"),
    ("plastered", "plaster"),
    ("bled", "bled"),
    ("motoring", "motor"),
    ("sing", "sing"),
    ("conflated", "conflat"),
    ("troubled", "troubl"),
    ("sized", "size"),
    ("hopping", "hop"),
    ("tanned", "tan"),
    ("falling", "fall"),
    ("hissing", "hiss"),
    ("fizzed", "fizz"),
    ("failing", "fail"),
    ("filing", "file"),
    ("happy", "happi"),
    ("sky", "sky"),
    ("relational", "relat"),
    ("conditional", "condit"),
    ("rational", "ration"),
    ("valenci", "valenc"),
    ("hesitanci", "hesit"),
    ("digitizer", "digit"),
    ("conformabli", "conform"),
    ("radicalli", "radic"),
    ("differentli", "differ"),
    ("vileli", "vile"),
    ("analogousli", "analog"),
    ("vietnamization", "vietnam"),
    ("predication", "predic"),
    ("operator", "oper"),
    ("feudalism", "feudal"),
    ("decisiveness", "decis"),
    ("hopefulness", "hope"),
    ("callousness", "callous"),
    ("formaliti", "formal"),
    ("sensitiviti", "sensit"),
    ("sensibiliti", "sensibl"),
    ("triplicate", "triplic"),
    ("formative", "form"),
    ("formalize", "formal"),
    ("electriciti", "electr"),
    ("electrical", "electr"),
    ("hopeful", "hope"),
    ("goodness", "good"),
    ("revival", "reviv"),
    ("allowance", "allow"),
    ("inference", "infer"),
    ("airliner", "airlin"),
    ("gyroscopic", "gyroscop"),
    ("adjustable", "adjust"),
    ("defensible", "defens"),
    ("irritant", "irrit"),
    ("replacement", "replac"),
    ("adjustment", "adjust"),
    ("dependent", "depend"),
    ("adoption", "adopt"),
    ("homologou", "homolog"),
    ("communism", "commun"),
    ("activate", "activ"),
    ("angulariti", "angular"),
    ("homologous", "homolog"),
    ("effective", "effect"),
    ("bowdlerize", "bowdler"),
    ("probate", "probat"),
    ("rate", "rate"),
    ("cease", "ceas"),
    ("controll", "control"),
    ("roll", "roll"),
    ("generalizations", "gener"),
    ("treatments", "treatment"),
]


@pytest.mark.parametrize(("word", "expected"), CASES)
def test_matches_reference(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_short_words_are_unchanged() -> None:
    assert [stem(w) for w in ("a", "is", "os", "9s")] == ["a", "is", "os", "9s"]


def test_tokenize_stems_only_when_asked() -> None:
    assert tokenize("Treatments treated") == ["treatments", "treated"]
    assert tokenize("Treatments treated", stem=True) == ["treatment", "treat"]


@pytest.mark.parametrize("retriever_type", [BM25Retriever, TfidfRetriever])
def test_stemming_matches_other_word_forms(retriever_type: type) -> None:
    chunks = [
        Chunk(id="a", document_id="d", text="The treatment reduced mortality in patients."),
        Chunk(id="b", document_id="d", text="Weather forecasts predict rain tomorrow."),
    ]
    stemmed = retriever_type()
    stemmed.index(chunks)
    assert [r.id for r in stemmed.retrieve("treatments", k=1)] == ["a"]
    plain = retriever_type(stem=False)
    plain.index(chunks)
    assert plain.retrieve("treatments", k=1) == []
