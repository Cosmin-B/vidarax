import unittest

from audio_labels import mapped_predictions, profile_label


class EfficientAtMappingTests(unittest.TestCase):
    def test_scans_past_unmapped_top_twelve_candidates(self) -> None:
        labels = [f"unmapped-{index}" for index in range(12)] + [
            "Whoosh, swoosh, swish"
        ]
        scores = [0.99 - index * 0.01 for index in range(12)] + [0.80]

        self.assertEqual(
            mapped_predictions(labels, scores, "gameplay", 0.35),
            [("whoosh", scores[-1])],
        )

    def test_deduplicates_profile_labels_using_highest_confidence(self) -> None:
        labels = ["Impact", "Thump, thud", "Speech"]
        scores = [0.72, 0.61, 0.95]

        mapped = mapped_predictions(labels, scores, "gameplay", 0.35)

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0][0], "impact")
        self.assertAlmostEqual(mapped[0][1], 0.72)

    def test_gameplay_profile_maps_motion_and_engine_sounds(self) -> None:
        expected = {
            "Whoosh, swoosh, swish": "whoosh",
            "Hiss": "hiss",
            "Aircraft engine": "engine",
            "Mechanisms": "mechanisms",
            "Scrape": "scrape",
        }

        self.assertEqual(
            {label: profile_label("gameplay", label) for label in expected},
            expected,
        )


if __name__ == "__main__":
    unittest.main()
