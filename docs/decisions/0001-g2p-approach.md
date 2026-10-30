# 0001: Grapheme-to-phoneme approach for phonological similarity

**Status:** Accepted

## Context

Phonological similarity needs an IPA transcription per lexeme.
Japanese lexemes `data/interim/lexemes.jsonl` carry a `reading` field (kana); where present, kana -> IPA is a
near-deterministic lookup and no G2P is needed. The open question is only what to do where `reading` is missing.

`decisions/0001-g2p-approach` (2026-07-19) reports fraction of lexemes with
non-null `reading`, by `language`:

|          | reading_coverage | reading_count |
| -------- | ---------------- | ------------- |
| language |                  |               |
| ---      | ---              | ---           | --- | --- | --- |
| es       | 0.000000         | 744           |
| it       | 0.000000         | 2429          |
| ja       | 0.683936         | 29551         |
| nl       | 0.000000         | 638           |
| pt       | 0.000000         | 92            |

When `fugashi` with `UniDic` is used to extract reading based on `form` only, we receive full alignment of real and predicted readings (both transformed to hiragana) in 91% of the cases with a minimal distance where differences exist:

```
Mean normalized distance: 0.04514383284299136
Mean Levenshtein distance: 0.24790021626515113
Exact matches: 18208
Compared: 19883
Accuracy: 0.9157571795000754
```

Further `PonPhon`-based phonological comparison of weighted IPA transcriptions obtained with the `Epitran` backend of `G2p+` yield even higher accuracy and high similarity:

```
PanPhon feature comparison
Mean similarity: 0.9905992314873757
Mean normalized distance: 0.009400768512623838
Mean feature edit distance: 0.06287483389285212
Zero-distance pairs: 18191
Compared: 19064
Zero-distance rate: 0.95420688208141
```

> `G2p+` was used because of its unified IPA format. `PonPhon` was used due to its weighted edit distance feature which is important based on Laufer's synformy theory.

Our further calculation of readings for words that miss them result in almost full coverage:

```
|  | reading_coverage | reading_count |
| --- | --- | --- |
| language |  |  |
| --- | --- | --- | --- | --- | --- |
| es | 0.000000 | 744 |
| it | 0.000000 | 2429 |
| ja | 0.998511 | 29551 |
| nl | 0.000000 | 638 |
| pt | 0.000000 | 92 |
```

100% was not achieved becaused of 46 malformed lexemes like なん／なに or サクサク★.

## Decision

1. Use `fugashi` for tokenization, together with hiragana normalization (which can be achieved by shifting unicode);
2. Fill in missing `reading` via `fugashi`;
3. Filter out the 46 malformed lexemes;
4. Use `G2p+` with the Epitran backend for IPA transcription;
5. Use `PonPhon` for weighted edit distance-based phonological comparison.

## Consequences

- Phonological similarity can be reliably implemented with the only issue of having a small chunk of readings which are not aligned with the input text. For this I suspect meaning-based disambiguation should be implemented which is out of scope for this decision.
