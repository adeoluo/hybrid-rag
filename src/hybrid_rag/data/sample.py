"""A small bundled corpus and labeled query set for reproducible evaluation.

The corpus is a handful of short, self-authored encyclopedia-style passages across
distinct topics; the labeled queries map a natural-language question to the passage(s)
that answer it. It exists so the retrieval metrics and the ablation produce concrete,
reproducible numbers offline, with no download.
"""

from __future__ import annotations

from hybrid_rag.core.types import Document
from hybrid_rag.evaluation.harness import LabeledQuery

_DOCUMENTS: tuple[tuple[str, str], ...] = (
    (
        "photosynthesis",
        "Photosynthesis is the process by which green plants convert sunlight, water, "
        "and carbon dioxide into glucose and oxygen. It takes place in the chloroplasts, "
        "where the pigment chlorophyll absorbs light energy.",
    ),
    (
        "mitochondria",
        "Mitochondria are membrane-bound organelles that generate most of a cell's "
        "supply of adenosine triphosphate, used as chemical energy. They are often "
        "called the powerhouse of the cell.",
    ),
    (
        "newton-gravity",
        "Isaac Newton formulated the law of universal gravitation, stating that every "
        "mass attracts every other mass with a force proportional to the product of "
        "their masses and inversely proportional to the square of the distance.",
    ),
    (
        "relativity",
        "Albert Einstein's theory of general relativity describes gravity as the "
        "curvature of spacetime caused by mass and energy, replacing Newton's picture of "
        "gravity as a force acting at a distance.",
    ),
    (
        "french-revolution",
        "The French Revolution began in 1789 and led to the end of the monarchy in "
        "France. It was driven by financial crisis, social inequality, and Enlightenment "
        "ideas about liberty and citizenship.",
    ),
    (
        "industrial-revolution",
        "The Industrial Revolution was a period of rapid industrial growth beginning in "
        "Britain in the late eighteenth century, marked by the steam engine, mechanized "
        "textile production, and the rise of factories.",
    ),
    (
        "dna",
        "Deoxyribonucleic acid, or DNA, carries the genetic instructions for the "
        "development and functioning of living organisms. Its double-helix structure was "
        "described by Watson and Crick in 1953.",
    ),
    (
        "evolution",
        "The theory of evolution by natural selection, proposed by Charles Darwin, "
        "explains how species change over generations as individuals with advantageous "
        "traits are more likely to survive and reproduce.",
    ),
    (
        "water-cycle",
        "The water cycle describes the continuous movement of water through evaporation, "
        "condensation, and precipitation, circulating water between the oceans, the "
        "atmosphere, and the land.",
    ),
    (
        "plate-tectonics",
        "Plate tectonics is the theory that Earth's outer shell is divided into large "
        "plates that move over the mantle, causing earthquakes, volcanoes, and the "
        "formation of mountain ranges.",
    ),
    (
        "solar-power",
        "Solar power converts energy from sunlight into electricity using photovoltaic "
        "panels. Unlike plants, these panels do not use water or carbon dioxide; they "
        "rely on semiconductors to move electrons when struck by light.",
    ),
    (
        "cellular-respiration",
        "Cellular respiration releases energy from glucose inside cells. It is distinct "
        "from the organelle that stores energy, and it consumes oxygen to produce carbon "
        "dioxide and water, the reverse of photosynthesis.",
    ),
    (
        "tidal-forces",
        "Tidal forces arise from gravity acting more strongly on the near side of a body "
        "than the far side. They stretch oceans and, over long periods, can reshape the "
        "orbits of moons without curving spacetime noticeably.",
    ),
)

_QUERIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("how do plants make energy from sunlight", ("photosynthesis",)),
    ("what is the powerhouse of the cell", ("mitochondria",)),
    ("who described gravity as curved spacetime", ("relativity",)),
    ("law of universal gravitation force between masses", ("newton-gravity",)),
    ("when did the monarchy end in France", ("french-revolution",)),
    ("steam engine and factories in Britain", ("industrial-revolution",)),
    ("double helix genetic instructions", ("dna",)),
    ("species change by natural selection over generations", ("evolution",)),
    ("evaporation condensation precipitation movement of water", ("water-cycle",)),
    ("plates moving over the mantle causing earthquakes", ("plate-tectonics",)),
    # Harder queries: shared vocabulary across passages forces real ranking.
    ("how does energy move inside a cell", ("mitochondria",)),
    ("Einstein theory of gravity and mass", ("relativity",)),
    ("what causes mountains and volcanoes to form", ("plate-tectonics",)),
    ("Darwin survival of advantageous traits", ("evolution",)),
    ("chlorophyll absorbs light in plants", ("photosynthesis",)),
)


def corpus() -> list[Document]:
    """Return the bundled documents."""
    return [Document(id=doc_id, text=text) for doc_id, text in _DOCUMENTS]


def labeled_queries() -> list[LabeledQuery]:
    """Return the labeled queries.

    Each document becomes a single chunk (the passages are short), so the relevant chunk
    id is ``<document id>#0``.
    """
    return [
        LabeledQuery(query=query, relevant={f"{doc_id}#0" for doc_id in doc_ids})
        for query, doc_ids in _QUERIES
    ]
