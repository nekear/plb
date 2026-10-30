# 0003: Orthographic similarity

**Status:** Accepted

## Context

---

To compute orthographic similarity between kanji characters [Niteru](https://github.com/ktfleming/niteru) (a.k.a., "stroke-edit distance"), ported from Scala to Python, was used.
However, the question of comparing kana characters was not addressed.

Experiments carried out in `decisions/0003-orthographic-similarity` suggested the following kana coverage on the dataset:

```
3493 kanas out of 24433 (14.3%)
```

Kana is a phonetic script - the grapheme-to-phoneme mapping is near-bijective. So for kana strings, orthographic similarity and phonological similarity, which has been already implemented, are close to the same quantity. What's not covered is the residual visual-only confusion: シ/ツ and ソ/ン are visually near-identical but phonologically unrelated. That said, there's a suggestion that they're also beginner-level confusions that A2–C1 learners have mostly outgrown by the time they're drilling vocabulary.

Nevertheless, an experiment with a pixel-level similarity for characters rendered with NotoSansJP-Regular was carried out yielding the following results:

```
シ / ツ: aligned=0.48, forward=0.53 [ツ->シ: angle=-10°, scale=(1.00, 1.00), shift=(1, -1), penalty=0.94], reverse=0.53 [シ->ツ: angle=+10°, scale=(1.00, 1.00), shift=(-1, 0), penalty=0.96], symmetric=0.53, gain=+0.05,
ソ / ン: aligned=0.51, forward=0.68 [ン->ソ: angle=+15°, scale=(1.00, 1.00), shift=(0, 0), penalty=0.96], reverse=0.68 [ソ->ン: angle=-15°, scale=(1.00, 1.00), shift=(0, 0), penalty=0.96], symmetric=0.68, gain=+0.17,
ク / ワ: aligned=0.37, forward=0.37 [ワ->ク: angle=+0°, scale=(1.00, 1.00), shift=(0, -2), penalty=0.92], reverse=0.37 [ク->ワ: angle=+0°, scale=(1.00, 1.00), shift=(0, 2), penalty=0.92], symmetric=0.37, gain=+0.00,
ア / マ: aligned=0.62, forward=0.62 [マ->ア: angle=+0°, scale=(1.00, 1.00), shift=(-2, 0), penalty=0.92], reverse=0.62 [ア->マ: angle=+0°, scale=(1.00, 1.00), shift=(2, 0), penalty=0.92], symmetric=0.62, gain=+0.00,
ス / ヌ: aligned=0.47, forward=0.48 [ヌ->ス: angle=+0°, scale=(1.00, 0.95), shift=(0, -2), penalty=0.85], reverse=0.48 [ス->ヌ: angle=+0°, scale=(1.00, 1.05), shift=(0, 1), penalty=0.90], symmetric=0.48, gain=+0.01,
テ / ラ: aligned=0.59, forward=0.59 [ラ->テ: angle=+0°, scale=(1.00, 1.00), shift=(1, 0), penalty=0.98], reverse=0.59 [テ->ラ: angle=+0°, scale=(1.00, 1.00), shift=(-1, 0), penalty=0.98], symmetric=0.59, gain=+0.00,
ソ / リ: aligned=0.48, forward=0.60 [リ->ソ: angle=-15°, scale=(1.00, 1.00), shift=(-1, -1), penalty=0.92], reverse=0.60 [ソ->リ: angle=+15°, scale=(1.00, 1.00), shift=(0, 1), penalty=0.94], symmetric=0.60, gain=+0.12,
は / ほ: aligned=0.74, forward=0.74 [ほ->は: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], reverse=0.74 [は->ほ: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], symmetric=0.74, gain=+0.00,
ね / れ: aligned=0.66, forward=0.66 [れ->ね: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], reverse=0.66 [ね->れ: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], symmetric=0.66, gain=+0.00,
る / ろ: aligned=0.78, forward=0.78 [ろ->る: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], reverse=0.78 [る->ろ: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], symmetric=0.78, gain=+0.00,
さ / き: aligned=0.54, forward=0.55 [き->さ: angle=-5°, scale=(1.00, 1.00), shift=(0, 1), penalty=0.98], reverse=0.54 [さ->き: angle=+5°, scale=(1.00, 1.00), shift=(0, -1), penalty=0.98], symmetric=0.55, gain=+0.00,
い / り: aligned=0.20, forward=0.28 [り->い: angle=-20°, scale=(1.00, 1.00), shift=(-1, 0), penalty=0.90], reverse=0.29 [い->り: angle=+20°, scale=(0.95, 1.00), shift=(0, 0), penalty=0.85], symmetric=0.28, gain=+0.09,
つ / う: aligned=0.30, forward=0.37 [う->つ: angle=-20°, scale=(1.00, 0.95), shift=(-1, 0), penalty=0.84], reverse=0.36 [つ->う: angle=+20°, scale=(1.00, 1.00), shift=(1, 1), penalty=0.89], symmetric=0.37, gain=+0.06,
土 / 士: aligned=0.93, forward=0.93 [士->土: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], reverse=0.93 [土->士: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], symmetric=0.93, gain=+0.00,
幸 / 爪: aligned=0.44, forward=0.44 [爪->幸: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], reverse=0.44 [幸->爪: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], symmetric=0.44, gain=+0.00,
微 / 入: aligned=0.24, forward=0.25 [入->微: angle=+5°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], reverse=0.25 [微->入: angle=-5°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], symmetric=0.25, gain=+0.00,
千 / 干: aligned=0.77, forward=0.77 [干->千: angle=+0°, scale=(1.00, 1.00), shift=(1, 0), penalty=0.98], reverse=0.77 [千->干: angle=+0°, scale=(1.00, 1.00), shift=(-1, 0), penalty=0.98], symmetric=0.77, gain=+0.00,
未 / 末: aligned=0.94, forward=0.94 [末->未: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], reverse=0.94 [未->末: angle=+0°, scale=(1.00, 1.00), shift=(0, 0), penalty=1.00], symmetric=0.94, gain=+0.00,
Normalized glyph cache: CacheInfo(hits=73, misses=35, maxsize=10000, currsize=35)
Directional affine cache: CacheInfo(hits=0, misses=36, maxsize=100000, currsize=36)
```

> Note: it cannot be considered fully reliable due real higher perceived similarity between the tested pairs.

Further discovery of [Letter visual similarity of Japanese hiragana and katakana based on reaction times](https://link.springer.com/article/10.1007/s12144-021-02664-w) by Higuchi and Kobayashi (2022) hinted that confusability is higher than the similarity above suggests.

Since perceptual-distance studies usually cover the basic kana but often omit dakuten (が, ざ), handakuten (ぱ), small/yōon kana (ゃ, ッ), and the long-vowel mark (ー), to verify that using their manually-crafted confusability matrix would work with our dataset, a quick coverage check was conducted:

```
Coverage of kanas in ESM based on our dataset: 20566 / 24433 (84.2%)
Some examples include:
['10日', '10月', '11日', '11月', '12月', '1つ目', '1分', '1日', '1時', '1月', '2つ目', '2分', '2時', '2月', '3つ目', '3分', '3時', '3月', '4月', '5日', '5月', '6月', '7日', '7月', '8日', '8月', '9日', '9月', 'JR', 'OL', 'verb+ことなく', '々', '〇', 'ああ', 'ああいう', 'あいこ', 'あいさつ', 'あいつ', 'あいつら', 'あいにく']
```

## Decisions

---

Instead of a single orthographic similarity metric, four columns per neighbor pair for the orthographic group are currently used:

1. `ortho_stroke_sim` - kanji–kanji only
2. `ortho_stroke_coverage` - fraction of char substitutions that were kanji–kanji
3. `ortho_kana_confusion` - kana–kana only
4. `ortho_kana_coverage` - fraction that were kana–kana

The affine `ortho_pixel_sim` feature is deferred until a later Model B performance iteration. Its implementation and experiments are retained, but it is not computed or emitted by the current similarity pipeline because cold character-pair scoring makes the full candidate pass infeasible.

One of the main reasons is that we shouldn't confuse orthographic similarity between kanji with the reversed midrank percentile expressed from 0 to 1 calculated using the Higuchi-Kobayashi (ESM) confusability matrix.

Moreover, since similarity can be computed within a single script sub-family, we end up with the following combinations:

1. kanji–kanji → stroke-edit (Niteru);
2. kana–kana → confusion score;
3. kanji–kana → neither is defined.

Similarities are computed between characters, and to combine them into a single score per lexeme, a weighted edit distance will be used: substitution cost = `1 - char_sim(a, b)`, insertion/deletion = 1.

## Consequences

---

We're introducing four active columns across two orthographic similarity channels, making the implementation more complex and less portable to other languages.

In return, we expect to improve the accuracy of the model on CJK lexemes.
