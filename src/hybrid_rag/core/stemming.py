"""The Porter stemmer (M.F. Porter, "An algorithm for suffix stripping", 1980).

Stemming maps inflected forms to a shared stem ("treatments", "treated", "treating" all
become "treat"), so a query and a document that use different forms of a word still match.
This is the original 1980 algorithm. It gives the same stem as NLTK's
``PorterStemmer(mode=ORIGINAL_ALGORITHM)`` for all 34,955 words longer than two letters in the
BEIR SciFact corpus; like Porter's reference C implementation, and unlike that NLTK mode, it
leaves words of one or two letters unchanged. ``tests/test_stemming.py`` pins reference stems.

A word is read as ``[C](VC){m}[V]``: alternating runs of consonants and vowels, where ``m``
(the *measure*) counts vowel-consonant pairs. Each step strips the longest matching suffix
when the remaining stem satisfies that rule's condition, usually a minimum measure.
"""

from __future__ import annotations

from collections.abc import Callable

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """The number of vowel-to-consonant transitions, Porter's ``m``."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if prev_vowel and not vowel:
            m += 1
        prev_vowel = vowel
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """``*o``: consonant, vowel, consonant, where the last is not w, x or y."""
    return (
        len(word) >= 3
        and _is_consonant(word, len(word) - 3)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


Rule = tuple[str, str, Callable[[str], bool]]


def _m_gt(n: int) -> Callable[[str], bool]:
    return lambda stem: _measure(stem) > n


def _apply(word: str, rules: list[Rule]) -> str:
    """Apply the rule with the longest matching suffix, if its condition holds."""
    for suffix, replacement, condition in rules:  # rules are ordered longest suffix first
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            return stem + replacement if condition(stem) else word
    return word


def _sorted(rules: list[Rule]) -> list[Rule]:
    return sorted(rules, key=lambda rule: -len(rule[0]))


_STEP2 = _sorted(
    [
        (suffix, replacement, _m_gt(0))
        for suffix, replacement in (
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        )
    ]
)

_STEP3 = _sorted(
    [
        (suffix, replacement, _m_gt(0))
        for suffix, replacement in (
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        )
    ]
)

_STEP4 = _sorted(
    [
        (suffix, "", _m_gt(1))
        for suffix in (
            "al",
            "ance",
            "ence",
            "er",
            "ic",
            "able",
            "ible",
            "ant",
            "ement",
            "ment",
            "ent",
            "ou",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
        )
    ]
    + [("ion", "", lambda stem: _measure(stem) > 1 and stem[-1:] in ("s", "t"))]
)


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        stem = word[:-3]
        return stem + "ee" if _measure(stem) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if not _has_vowel(stem):
                return word
            if stem.endswith(("at", "bl", "iz")):
                return stem + "e"
            if _ends_double_consonant(stem) and stem[-1] not in "lsz":
                return stem[:-1]
            if _measure(stem) == 1 and _ends_cvc(stem):
                return stem + "e"
            return stem
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


# Stems memoized in a plain dict: a corpus repeats the same words constantly, and a dict
# lookup is cheaper than an lru_cache call. Bounded so a stream of unique tokens cannot grow it
# without limit.
_CACHE: dict[str, str] = {}
_CACHE_LIMIT = 1 << 20


def stem(word: str) -> str:
    """Return the Porter stem of a lowercase ``word``."""
    cached = _CACHE.get(word)
    if cached is None:
        cached = _stem(word)
        if len(_CACHE) < _CACHE_LIMIT:
            _CACHE[word] = cached
    return cached


def stem_all(words: list[str]) -> list[str]:
    """``[stem(w) for w in words]``, with the cache lookup inlined for speed."""
    get = _CACHE.get
    out = []
    for word in words:
        cached = get(word)
        out.append(cached if cached is not None else stem(word))
    return out


def _stem(word: str) -> str:
    if len(word) <= 2:
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _apply(word, _STEP2)
    word = _apply(word, _STEP3)
    word = _apply(word, _STEP4)
    return _step5(word)
