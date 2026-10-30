# 0002: Choosing morphological similarity backend

**Status:** Accepted

# Context
***
Morphological similarity requires a library able to split lexemes into morphemes.
Performance varies from implementation to implementation and having a language-specific version is
preferable since usually it also means higher performance.

For Japanese `Fugashi` with UniDic was tested - it doesn't output morphemes per se,
but meaningfully similar structures can be obtained from UniDic's `feature.lemma`.

For other languages there was an attempt to use `polyglot`, but trying to load any of the available backends
resulted in 403: Forbidden - it seems like that project has been abandoned for a long time.

# Decisions
***
1. The current version will only support Japanese to simplify the implementation;
2. For other languages it's recommended to perform a custom training of morphessor.

# Consequences
***
1. The first version will only support Japanese.
