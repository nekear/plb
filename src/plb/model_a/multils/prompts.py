from __future__ import annotations

from enum import StrEnum
from textwrap import dedent

from plb.model_a.kvl.data import PLACEHOLDER_DIGIT


class MultiLSCondition(StrEnum):
    FULL_L1 = "full_l1"
    FULL_L2 = "full_l2"
    JA_WORD_ONLY = "ja_word_only"
    L1_NAME_ONLY = "l1_name_only"
    FULL_L2_NO_CONTEXT = "full_l2_no_context"
    FULL_L2_DEPLOY = "full_l2_deploy"


def make_multils_prompt(
    condition: MultiLSCondition | str,
    *,
    target: str,
    l1_name: str | None = None,
    l1_gloss: str | None = None,
    l1_context: str | None = None,
    japanese_context: str | None = None,
    l2_clue: str | None = None,
    l1_clue: str | None = None,
) -> str:
    condition = MultiLSCondition(condition)

    if condition is MultiLSCondition.JA_WORD_ONLY:
        return dedent(f"""
            Rate how difficult the Japanese word is for learners on a scale from 1 to 5 (1=very easy, 5=very difficult).
            Japanese word: {target}
            Difficulty: {PLACEHOLDER_DIGIT}
        """).strip()

    if not l1_name:
        raise ValueError(f"l1_name is required for condition {condition.value!r}")

    ### -> HERE: `l1_name` should be present

    if condition is MultiLSCondition.L1_NAME_ONLY:
        return dedent(f"""
        Rate how difficult the Japanese word is for learners whose native language is {l1_name} on a scale from 1 to 5 (1=very easy, 5=very difficult).
        Japanese word: {target}
        Difficulty: {PLACEHOLDER_DIGIT}
        """).strip()

    if not l1_gloss:
        raise ValueError(f"l1_gloss is required for condition {condition.value!r}")

    ### -> HERE: `l1_name` and `l1_gloss` should be present

    if condition is MultiLSCondition.FULL_L2_DEPLOY:
        ### -> HERE: `l1_name` and `l1_gloss` should be present
        return dedent(
            f"""
                Rate how easy it is for learners to guess the {l1_name} word based on the Japanese word on a scale from 1 to 5 (1=very easy, 5=very difficult).
                Japanese word: {target}
                {l1_name} word: {l1_gloss}
                Difficulty: {PLACEHOLDER_DIGIT}
            """
        ).strip()

    if condition is MultiLSCondition.FULL_L1:
        if not l1_context or not l2_clue:
            raise ValueError(
                "l1_context and l2_clue are required for condition 'full_l1'"
            )
        return dedent(f"""
        Rate how easy it is for learners to guess the Japanese word based on the {l1_name} word, context and clue on a scale from 1 to 5 (1=very easy, 5=very difficult).
        {l1_name} word: {l1_gloss}
        {l1_name} context: {l1_context}
        Clue: {l2_clue}
        Japanese word: {target}
        Difficulty: {PLACEHOLDER_DIGIT}
        """).strip()

    if condition is MultiLSCondition.FULL_L2:
        if not japanese_context or not l1_clue:
            raise ValueError(
                "japanese_context and l1_clue are required for condition 'full_l2'"
            )

        ### -> HERE: `l1_name`, `l1_gloss`, `japanese_context`, and `l1_clue` should be present

        return dedent(
            f"""
                Rate how easy it is for learners to guess the {l1_name} word based on the Japanese word, context and clue on a scale from 1 to 5 (1=very easy, 5=very difficult).
                Japanese word: {target}
                Japanese context: {japanese_context}
                Clue: {l1_clue}
                {l1_name} word: {l1_gloss}
                Difficulty: {PLACEHOLDER_DIGIT}
            """
        ).strip()

    if condition is MultiLSCondition.FULL_L2_NO_CONTEXT:
        if not l1_clue:
            raise ValueError(
                "l1_clue is required for condition 'full_l2_no_context'"
            )

        ### -> HERE: `l1_name`, `l1_gloss`, and `l1_clue` should be present

        return dedent(
            f"""
                Rate how easy it is for learners to guess the {l1_name} word based on the Japanese word and clue on a scale from 1 to 5 (1=very easy, 5=very difficult).
                Japanese word: {target}
                Clue: {l1_clue}
                {l1_name} word: {l1_gloss}
                Difficulty: {PLACEHOLDER_DIGIT}
            """
        ).strip()
