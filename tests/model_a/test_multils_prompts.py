import pytest

from plb.model_a.multils.prompts import MultiLSCondition, make_multils_prompt


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (
            MultiLSCondition.FULL_L1,
            """Rate how easy it is for learners to guess the Japanese word based on the English word, context and clue on a scale from 1 to 5 (1=very easy, 5=very difficult).
English word: publication
English context: It was published in the journal.
Clue: 掲___
Japanese word: 掲載した
Difficulty: 1""",
        ),
        (
            MultiLSCondition.FULL_L2,
            """Rate how easy it is for learners to guess the English word based on the Japanese word, context and clue on a scale from 1 to 5 (1=very easy, 5=very difficult).
Japanese word: 掲載した
Japanese context: 回答も掲載した。
Clue: p__________
English word: publication
Difficulty: 1""",
        ),
        (
            MultiLSCondition.FULL_L2_NO_CONTEXT,
            """Rate how easy it is for learners to guess the English word based on the Japanese word and clue on a scale from 1 to 5 (1=very easy, 5=very difficult).
Japanese word: 掲載した
Clue: p__________
English word: publication
Difficulty: 1""",
        ),
        (
            MultiLSCondition.FULL_L2_DEPLOY,
            """Rate how easy it is for learners to guess the English word based on the Japanese word on a scale from 1 to 5 (1=very easy, 5=very difficult).
Japanese word: 掲載した
English word: publication
Difficulty: 1""",
        ),
        (
            MultiLSCondition.JA_WORD_ONLY,
            """Rate how difficult the Japanese word is for learners on a scale from 1 to 5 (1=very easy, 5=very difficult).
Japanese word: 掲載した
Difficulty: 1""",
        ),
        (
            MultiLSCondition.L1_NAME_ONLY,
            """Rate how difficult the Japanese word is for learners whose native language is English on a scale from 1 to 5 (1=very easy, 5=very difficult).
Japanese word: 掲載した
Difficulty: 1""",
        ),
    ],
)
def test_prompt_snapshots(condition: MultiLSCondition, expected: str) -> None:
    assert (
        make_multils_prompt(
            condition,
            target="掲載した",
            l1_name="English",
            l1_gloss="publication",
            l1_context="It was published in the journal.",
            japanese_context="回答も掲載した。",
            l2_clue="掲___",
            l1_clue="p__________",
        )
        == expected
    )


def test_full_l1_rejects_missing_generated_content() -> None:
    with pytest.raises(ValueError, match="l1_gloss"):
        make_multils_prompt(
            MultiLSCondition.FULL_L1,
            target="掲載した",
            l1_name="English",
        )


@pytest.mark.parametrize(
    "condition",
    [
        MultiLSCondition.FULL_L2,
        MultiLSCondition.FULL_L2_NO_CONTEXT,
    ],
)
def test_l2_clue_conditions_reject_missing_l1_clue(
    condition: MultiLSCondition,
) -> None:
    with pytest.raises(ValueError, match="l1_clue"):
        make_multils_prompt(
            condition,
            target="掲載した",
            l1_name="English",
            l1_gloss="publication",
            japanese_context="回答も掲載した。",
        )


def test_full_l2_deploy_requires_only_l1_name_and_gloss() -> None:
    prompt = make_multils_prompt(
        MultiLSCondition.FULL_L2_DEPLOY,
        target="掲載した",
        l1_name="English",
        l1_gloss="publication",
    )
    assert "Japanese context:" not in prompt
    assert "Clue:" not in prompt
