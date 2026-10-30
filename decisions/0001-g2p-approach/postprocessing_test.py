# %%
import pandas as pd

def get_reading_coverage(lexemes_df: pd.DataFrame):
    return lexemes_df.groupby("language")["reading"].agg(
        reading_coverage=lambda s: s.notna().mean(),
        lexeme_count="size",
    )

lexemes_df = pd.read_json("data/interim/lexemes.jsonl", lines=True)
get_reading_coverage(lexemes_df)
