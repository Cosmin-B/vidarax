import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audio_runtime
from audio_labels import timestamped_transcript_chunks, transcript_text_is_unreliable


class AudioRuntimeTests(unittest.TestCase):
    def test_all_child_processes_drop_foreign_python_and_openssl_state(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SSLKEYLOGFILE": "C:/captures/tls.keys",
                "OPENSSL_CONF": "C:/foreign/openssl.cnf",
                "OPENSSL_MODULES": "C:/foreign/providers",
                "PYTHONHOME": "C:/foreign/python",
                "PYTHONPATH": "C:/foreign/site-packages",
            },
            clear=False,
        ), mock.patch.object(audio_runtime.subprocess, "run") as run:
            run.return_value = audio_runtime.subprocess.CompletedProcess([], 0)
            audio_runtime._run(("python", "-m", "venv", "target"))

        environment = run.call_args.kwargs["env"]
        for name in audio_runtime.UNSAFE_CHILD_ENVIRONMENT:
            self.assertNotIn(name, environment)

    def test_cache_override_controls_all_default_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {"VIDARAX_CACHE_DIR": temporary},
                clear=False,
            ):
                paths = audio_runtime._paths("whisper")

        root = Path(temporary)
        self.assertEqual(paths["cache"], root)
        self.assertEqual(paths["model"], root / "models")
        self.assertEqual(paths["venv"], root / "audio" / "envs" / "whisper")

    def test_install_marker_allows_non_identity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {"VIDARAX_CACHE_DIR": temporary},
                clear=False,
            ):
                paths = audio_runtime._paths("whisper")
                python = audio_runtime._venv_executable(paths["venv"], "python")
                python.parent.mkdir(parents=True)
                python.touch()
                weights = audio_runtime._classifier_weights_path(paths)
                weights.parent.mkdir(parents=True)
                weights.touch()
                marker = {
                    **audio_runtime._expected_marker("whisper"),
                    "installed_at_unix": 1,
                    "install_seconds": 2.5,
                }
                audio_runtime._marker_path(paths).write_text(
                    json.dumps(marker),
                    encoding="utf-8",
                )
                with mock.patch.object(
                    audio_runtime,
                    "_git_head",
                    return_value=audio_runtime.EFFICIENTAT_COMMIT,
                ):
                    self.assertTrue(audio_runtime._ready("whisper", paths))

    def test_profile_environment_keeps_models_outside_the_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {
                    "VIDARAX_CACHE_DIR": temporary,
                    "SSLKEYLOGFILE": "C:/captures/tls.keys",
                    "OPENSSL_CONF": "C:/foreign/openssl.cnf",
                    "OPENSSL_MODULES": "C:/foreign/providers",
                    "PYTHONHOME": "C:/foreign/python",
                    "PYTHONPATH": "C:/foreign/site-packages",
                },
                clear=False,
            ):
                paths = audio_runtime._paths("qwen")
                environment = audio_runtime._environment("qwen", paths)

        self.assertEqual(
            environment["HF_HOME"], str(Path(temporary) / "models" / "huggingface")
        )
        self.assertEqual(
            environment["TORCH_HOME"], str(Path(temporary) / "models" / "torch")
        )
        self.assertEqual(environment["VIDARAX_AUDIO_AUTO_ASR"], "qwen3_asr")
        self.assertEqual(environment["UV_NATIVE_TLS"], "true")
        self.assertNotIn("SSLKEYLOGFILE", environment)
        self.assertNotIn("OPENSSL_CONF", environment)
        self.assertNotIn("OPENSSL_MODULES", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("PYTHONPATH", environment)

    def test_compatible_invoking_python_is_reused(self) -> None:
        with mock.patch.object(audio_runtime.sys, "version_info", (3, 12, 4)):
            self.assertEqual(
                audio_runtime._python_request(),
                audio_runtime.sys.executable,
            )

    def test_incompatible_invoking_python_uses_managed_version(self) -> None:
        with mock.patch.object(audio_runtime.sys, "version_info", (3, 9, 6)):
            self.assertEqual(audio_runtime._python_request(), "3.12")

    def test_default_classifier_is_part_of_runtime_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {"VIDARAX_CACHE_DIR": temporary},
                clear=False,
            ):
                paths = audio_runtime._paths("core")
                python = audio_runtime._venv_executable(paths["venv"], "python")
                python.parent.mkdir(parents=True)
                python.touch()
                marker = {**audio_runtime._expected_marker("core")}
                audio_runtime._marker_path(paths).write_text(
                    json.dumps(marker),
                    encoding="utf-8",
                )
                with mock.patch.object(
                    audio_runtime,
                    "_git_head",
                    return_value=audio_runtime.EFFICIENTAT_COMMIT,
                ):
                    self.assertFalse(audio_runtime._ready("core", paths))

                weights = audio_runtime._classifier_weights_path(paths)
                weights.parent.mkdir(parents=True)
                weights.touch()
                with mock.patch.object(
                    audio_runtime,
                    "_git_head",
                    return_value=audio_runtime.EFFICIENTAT_COMMIT,
                ):
                    self.assertTrue(audio_runtime._ready("core", paths))

    def test_offline_install_never_bootstraps_uv_from_the_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {"VIDARAX_CACHE_DIR": temporary},
                clear=False,
            ):
                paths = audio_runtime._paths("whisper")
                with mock.patch.object(audio_runtime, "_find_uv", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "without --offline"):
                        audio_runtime._ensure_uv(paths, offline=True)


class TranscriptTimestampTests(unittest.TestCase):
    def test_rejects_repetitive_truncated_whisper_output(self) -> None:
        self.assertTrue(
            transcript_text_is_unreliable(
                "Water, fire, fire, fire, fire, fire, fire, fire, fire, fire, fire.",
                14.6,
            )
        )
        self.assertFalse(
            transcript_text_is_unreliable(
                "This audio warning for the hazard rock is not supposed to be there.",
                7.4,
            )
        )

    def test_uses_model_timestamps_instead_of_the_full_vad_window(self) -> None:
        result = {
            "text": "Fuel empty. Was that air hissing sound?",
            "chunks": [
                {"text": "Fuel empty.", "timestamp": (0.4, 1.6)},
                {
                    "text": "Was that air hissing sound?",
                    "timestamp": (8.2, 10.1),
                },
            ],
        }

        self.assertEqual(
            timestamped_transcript_chunks(result, 12_000),
            [
                (400, 1_600, "Fuel empty."),
                (8_200, 10_100, "Was that air hissing sound?"),
            ],
        )

    def test_falls_back_to_one_bounded_segment_without_timestamps(self) -> None:
        self.assertEqual(
            timestamped_transcript_chunks({"text": "Fuel empty."}, 2_500),
            [(0, 2_500, "Fuel empty.")],
        )

    def test_groups_word_timestamps_without_recreating_a_broad_window(self) -> None:
        result = {
            "text": "Watch that rock passing you. Was that air-hissing sound?",
            "chunks": [
                {"text": " Watch", "timestamp": (0.2, 0.5)},
                {"text": " that", "timestamp": (0.5, 0.7)},
                {"text": " rock", "timestamp": (0.7, 1.0)},
                {"text": " passing", "timestamp": (1.0, 1.4)},
                {"text": " you.", "timestamp": (1.4, 1.8)},
                {"text": " Was", "timestamp": (8.2, 8.5)},
                {"text": " that", "timestamp": (8.5, 8.7)},
                {"text": " air", "timestamp": (8.7, 9.0)},
                {"text": "-hissing", "timestamp": (9.0, 9.4)},
                {"text": " sound?", "timestamp": (9.4, 9.8)},
            ],
        }

        self.assertEqual(
            timestamped_transcript_chunks(result, 12_000),
            [
                (200, 1_800, "Watch that rock passing you."),
                (8_200, 9_800, "Was that air-hissing sound?"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
