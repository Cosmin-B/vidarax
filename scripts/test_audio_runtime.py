import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audio_runtime


class AudioRuntimeTests(unittest.TestCase):
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
                {"VIDARAX_CACHE_DIR": temporary},
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


if __name__ == "__main__":
    unittest.main()
