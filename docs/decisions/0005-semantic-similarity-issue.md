# 0005: Semantic Similarity Issue

**Status:** Accepted

## Context

---

`compute_semantic_similarity` is a plain cosine over the direct E5 embeddings of each word instance's meaning. 
I suspected that the embedding space is anisotropic, because even seemingly unrelated pairs scores 0.8-0.9.

The analysis in the `0005-semantic-similarity-issue` notebook confirmed that space is indeed badly anisotropic: 50k random (i.e. unrelated) pairs average 0.809 cosine, with p1 = 0.739 and p99 = 0.884.
For instance, 食あたり ("food poisoning") and 支払い ("payment") score 0.907, while a genuinely related pair - 呑気 / 気楽, both "carefree" - scores 0.915. Such a raw score carries almost no information: at a 0.70 cut, 99.99% of all pairs pass.
And even my current 0.90 threshold doesn't provide any meaningful filtering.

## Decisions
---

As a solution, the corpus centroid will subtracted and renormalized to re-centre the random-pair distribution on 0.000 ± 0.119.
As the experiments showed, 呑気/気楽, in this case, will be moved to 0.414 making the similarity score more meaningful.

## Consequences

---

1. The thresholds for similarities as a determining filtering method will be abandoned, and we will switch to top-K across the channels; leaving thresholds only to keep the neighborhood free of noise.
2. The obtained embeddings will be mean-centered to make the scores more meaningful.
3. Perhaps, other measures to ensure similarity score alignment will be taken.