import unittest

import pandas as pd

from comment_loader import extract_comments_from_dataframe, find_comment_column


class CommentLoaderTest(unittest.TestCase):
    def test_loads_xquik_tweet_text_and_filters_blank_rows(self):
        frame = pd.DataFrame(
            {
                "Tweet Created At": ["2026-07-05", "2026-07-05"],
                "Tweet Text": ["Useful walkthrough", "   "],
            }
        )

        self.assertEqual(extract_comments_from_dataframe(frame), ["Useful walkthrough"])

    def test_prefers_exact_comment_alias_over_metadata(self):
        frame = pd.DataFrame(
            {
                "tweet_created_at": ["2026-07-05"],
                "comment": ["Great feature demo"],
            }
        )

        self.assertEqual(find_comment_column(frame.columns), "comment")

    def test_missing_comment_column_raises_clear_error(self):
        frame = pd.DataFrame({"created_at": ["2026-07-05"]})

        with self.assertRaisesRegex(ValueError, "comment, Tweet Text"):
            extract_comments_from_dataframe(frame)


if __name__ == "__main__":
    unittest.main()
