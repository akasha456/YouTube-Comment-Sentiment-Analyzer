import unittest

from youtube_url import get_video_id_from_url


class YouTubeUrlTest(unittest.TestCase):
    def test_extracts_supported_youtube_urls(self):
        video_id = "dQw4w9WgXcQ"
        urls = (
            f"https://www.youtube.com/watch?v={video_id}&feature=share",
            f"https://youtu.be/{video_id}?si=example",
            f"https://www.youtube.com/shorts/{video_id}",
            f"https://www.youtube.com/embed/{video_id}",
            f"https://m.youtube.com/live/{video_id}",
        )

        self.assertEqual([get_video_id_from_url(url) for url in urls], [video_id] * len(urls))

    def test_rejects_non_youtube_hosts_and_invalid_ids(self):
        urls = (
            "https://example.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=too-short",
            "not-a-url",
        )

        self.assertEqual([get_video_id_from_url(url) for url in urls], [None, None, None])


if __name__ == "__main__":
    unittest.main()
