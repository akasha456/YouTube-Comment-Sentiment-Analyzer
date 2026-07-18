import pandas as pd

COMMENT_COLUMN_ALIASES = (
    "comment",
    "comments",
    "tweet text",
    "tweet_text",
    "tweettext",
    "text",
    "message",
    "body",
    "review",
    "feedback",
)


def normalize_column_name(column):
    return " ".join(str(column).replace("_", " ").strip().lower().split())


def find_comment_column(columns):
    column_names = list(columns)
    normalized = {normalize_column_name(column): str(column) for column in column_names}

    for alias in COMMENT_COLUMN_ALIASES:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized:
            return normalized[normalized_alias]

    for column in column_names:
        normalized_column = normalize_column_name(column)
        if any(token in normalized_column.split() for token in ("comment", "tweet", "text", "review", "feedback")):
            return str(column)

    raise ValueError("CSV must include a comment, Tweet Text, text, review, or feedback column")


def extract_comments_from_dataframe(dataframe):
    comment_column = find_comment_column(dataframe.columns)
    comments = dataframe[comment_column].fillna("").astype(str).str.strip()
    comments = [comment for comment in comments.tolist() if comment]
    if not comments:
        raise ValueError("CSV comment column does not contain any non-empty comments")
    return comments


def load_comments_from_csv(source):
    return extract_comments_from_dataframe(pd.read_csv(source))
