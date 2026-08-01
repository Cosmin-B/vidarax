#!/usr/bin/env python3
"""Install, inspect, and run the Vidarax local audio runtime.

The first run creates a locked uv environment in the user cache. Later runs
reuse the Python environment, package cache, model cache, and pinned
EfficientAT checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union


ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR / "audio"
LOCK_FILE = PROJECT_DIR / "uv.lock"
SERVER_SCRIPT = ROOT_DIR / "scripts" / "audio_perception_server.py"
EFFICIENTAT_COMMIT = os.environ.get(
    "VIDARAX_EFFICIENTAT_COMMIT",
    "a425fdce92572e602a1d5634799bd9f1f2efa806",
)
UV_REQUIREMENT = "uv>=0.10.0,<0.11"
PYTHON_VERSION = "3.12"
DEFAULT_CLASSIFIER = "dymn10_as"
PROFILES = ("core", "whisper", "moonshine", "qwen", "sensevoice", "lfm", "all")
PROFILE_ENGINE = {
    "core": "none",
    "whisper": "whisper",
    "moonshine": "moonshine",
    "qwen": "qwen3_asr",
    "sensevoice": "sensevoice",
    "lfm": "lfm2_5_audio",
    "all": "whisper",
}
UNSAFE_CHILD_ENVIRONMENT = (
    "SSLKEYLOGFILE",
    "OPENSSL_CONF",
    "OPENSSL_MODULES",
    "PYTHONHOME",
    "PYTHONPATH",
)


def _default_cache_dir() -> Path:
    configured = os.environ.get("VIDARAX_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "vidarax" / "Cache"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "vidarax"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vidarax"


def _paths(profile: str) -> dict[str, Path]:
    cache_dir = _default_cache_dir()
    model_dir = Path(
        os.environ.get("VIDARAX_MODEL_CACHE_DIR", cache_dir / "models")
    ).expanduser()
    configured_venv = os.environ.get("VIDARAX_AUDIO_VENV_DIR")
    venv_dir = (
        Path(configured_venv).expanduser()
        if configured_venv
        else cache_dir / "audio" / "envs" / profile
    )
    return {
        "cache": cache_dir,
        "model": model_dir,
        "venv": venv_dir,
        "efficientat": model_dir / "source" / "EfficientAT",
        "uv_cache": cache_dir / "uv",
        "uv_tool": cache_dir / "tools" / "uv",
        "hf": model_dir / "huggingface",
        "torch": model_dir / "torch",
    }


def _venv_executable(venv: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv / directory / f"{name}{suffix}"


def _run(
    command: Sequence[Union[str, os.PathLike[str]]],
    *,
    env: Optional[dict[str, str]] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    rendered = [os.fspath(part) for part in command]
    return subprocess.run(
        rendered,
        check=True,
        env=_child_environment() if env is None else env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _child_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in UNSAFE_CHILD_ENVIRONMENT:
        env.pop(name, None)
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment(profile: str, paths: dict[str, Path]) -> dict[str, str]:
    env = _child_environment()
    env.update(
        {
            "HF_HOME": os.fspath(paths["hf"]),
            "TORCH_HOME": os.fspath(paths["torch"]),
            "UV_CACHE_DIR": os.fspath(paths["uv_cache"]),
            "UV_PROJECT_ENVIRONMENT": os.fspath(paths["venv"]),
            "VIDARAX_EFFICIENTAT_REPO": os.fspath(paths["efficientat"]),
            "VIDARAX_AUDIO_AUTO_ASR": PROFILE_ENGINE[profile],
        }
    )
    # uv 0.10 calls this native TLS. It uses the Windows and macOS trust stores,
    # including enterprise roots, while keeping certificate validation enabled.
    env.setdefault("UV_NATIVE_TLS", "true")
    return env


def _python_request() -> str:
    if (3, 12) <= tuple(sys.version_info[:2]) < (3, 14):
        return sys.executable
    return PYTHON_VERSION


def _uv_version(uv: Path) -> str:
    try:
        result = _run((uv, "--version"), capture=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _find_uv(paths: dict[str, Path]) -> Optional[Path]:
    configured = os.environ.get("VIDARAX_UV")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(found) if (found := shutil.which("uv")) else None,
        _venv_executable(paths["uv_tool"], "uv"),
    ]
    for candidate in candidates:
        if (
            candidate
            and candidate.is_file()
            and _uv_version(candidate).startswith("uv 0.10.")
        ):
            return candidate
    return None


def _ensure_uv(paths: dict[str, Path], offline: bool) -> Path:
    existing = _find_uv(paths)
    if existing:
        return existing
    if offline:
        raise RuntimeError(
            "uv is not cached at the required version. Run once without --offline."
        )
    tool_venv = paths["uv_tool"]
    print(f"audio setup: installing {UV_REQUIREMENT} in {tool_venv}", file=sys.stderr)
    tool_venv.parent.mkdir(parents=True, exist_ok=True)
    _run((sys.executable, "-m", "venv", tool_venv))
    python = _venv_executable(tool_venv, "python")
    _run(
        (
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            UV_REQUIREMENT,
        )
    )
    uv = _venv_executable(tool_venv, "uv")
    if not uv.is_file():
        raise RuntimeError(f"uv installation completed without creating {uv}")
    return uv


def _git_head(repository: Path) -> str:
    if not (repository / ".git").exists():
        return ""
    try:
        result = _run(
            ("git", "-C", repository, "rev-parse", "HEAD"),
            capture=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _ensure_efficientat(paths: dict[str, Path], offline: bool) -> None:
    repository = paths["efficientat"]
    if _git_head(repository) == EFFICIENTAT_COMMIT:
        return
    if offline:
        raise RuntimeError(
            "EfficientAT is not cached at the required revision. "
            "Run once without --offline."
        )
    repository.parent.mkdir(parents=True, exist_ok=True)
    if not (repository / ".git").exists():
        _run(
            (
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/fschmid56/EfficientAT.git",
                repository,
            )
        )
    _run(("git", "-C", repository, "fetch", "--depth=1", "origin", EFFICIENTAT_COMMIT))
    _run(("git", "-C", repository, "checkout", "--detach", EFFICIENTAT_COMMIT))


def _marker_path(paths: dict[str, Path]) -> Path:
    return paths["venv"] / ".vidarax-audio-runtime.json"


def _classifier_weights_path(paths: dict[str, Path]) -> Path:
    return paths["efficientat"] / "resources" / f"{DEFAULT_CLASSIFIER}.pt"


def _expected_marker(profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "lock_sha256": _sha256(LOCK_FILE),
        "python": PYTHON_VERSION,
        "efficientat_commit": EFFICIENTAT_COMMIT,
        "classifier": DEFAULT_CLASSIFIER,
    }


def _read_marker(paths: dict[str, Path]) -> dict[str, Any]:
    try:
        return json.loads(_marker_path(paths).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _ready(profile: str, paths: dict[str, Path]) -> bool:
    python = _venv_executable(paths["venv"], "python")
    marker = _read_marker(paths)
    expected = _expected_marker(profile)
    return (
        python.is_file()
        and all(marker.get(key) == value for key, value in expected.items())
        and _git_head(paths["efficientat"]) == EFFICIENTAT_COMMIT
        and _classifier_weights_path(paths).is_file()
    )


def _warm_default_models(
    profile: str,
    paths: dict[str, Path],
    *,
    offline: bool,
) -> None:
    if offline and not _classifier_weights_path(paths).is_file():
        raise RuntimeError(
            f"{DEFAULT_CLASSIFIER} is not cached. Run once without --offline."
        )
    python = _venv_executable(paths["venv"], "python")
    _run(
        (
            python,
            SERVER_SCRIPT,
            "--warmup",
            "--efficientat-repo",
            paths["efficientat"],
            "--efficientat-model",
            DEFAULT_CLASSIFIER,
            "--auto-asr",
            "none",
        ),
        env=_environment(profile, paths),
    )


def _write_marker(profile: str, paths: dict[str, Path], elapsed_seconds: float) -> None:
    marker = {
        **_expected_marker(profile),
        "installed_at_unix": round(time.time()),
        "install_seconds": round(elapsed_seconds, 3),
    }
    destination = _marker_path(paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
    ) as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def install(profile: str, *, force: bool, offline: bool) -> dict[str, Any]:
    paths = _paths(profile)
    started = time.perf_counter()
    if _ready(profile, paths) and not force:
        return {
            "status": "ready",
            "changed": False,
            "profile": profile,
            "venv": os.fspath(paths["venv"]),
            "models": os.fspath(paths["model"]),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    uv = _ensure_uv(paths, offline)
    env = _environment(profile, paths)
    command: list[Union[str, os.PathLike[str]]] = [
        uv,
        "sync",
        "--project",
        PROJECT_DIR,
        "--locked",
        "--no-dev",
        "--python",
        _python_request(),
    ]
    if profile == "all":
        command.append("--all-extras")
    elif profile != "core":
        command.extend(("--extra", profile))
    if offline:
        command.append("--offline")

    paths["venv"].parent.mkdir(parents=True, exist_ok=True)
    print(
        f"audio setup: syncing locked {profile} environment in {paths['venv']}",
        file=sys.stderr,
    )
    _run(command, env=env)
    _ensure_efficientat(paths, offline)
    _warm_default_models(profile, paths, offline=offline)
    elapsed = time.perf_counter() - started
    _write_marker(profile, paths, elapsed)
    return {
        "status": "installed",
        "changed": True,
        "profile": profile,
        "venv": os.fspath(paths["venv"]),
        "models": os.fspath(paths["model"]),
        "uv": _uv_version(uv),
        "elapsed_seconds": round(elapsed, 3),
    }


def check(profile: str) -> dict[str, Any]:
    paths = _paths(profile)
    uv = _find_uv(paths)
    python = _venv_executable(paths["venv"], "python")
    return {
        "status": "ready" if _ready(profile, paths) else "missing",
        "profile": profile,
        "python": os.fspath(python) if python.is_file() else None,
        "venv": os.fspath(paths["venv"]),
        "models": os.fspath(paths["model"]),
        "efficientat": {
            "path": os.fspath(paths["efficientat"]),
            "revision": _git_head(paths["efficientat"]) or None,
            "expected_revision": EFFICIENTAT_COMMIT,
        },
        "uv": _uv_version(uv) if uv else None,
        "ffmpeg": shutil.which("ffmpeg"),
        "git": shutil.which("git"),
        "marker": _read_marker(paths) or None,
    }


def run_server(
    profile: str,
    *,
    no_install: bool,
    offline: bool,
    server_args: Sequence[str],
) -> None:
    paths = _paths(profile)
    if not _ready(profile, paths):
        if no_install:
            raise RuntimeError(
                f"audio profile {profile} is not installed. "
                f"Run {Path(sys.argv[0]).name} install --profile {profile}."
            )
        result = install(profile, force=False, offline=offline)
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
    python = _venv_executable(paths["venv"], "python")
    env = _environment(profile, paths)
    command = [python, SERVER_SCRIPT, *server_args]
    if os.name == "nt":
        raise SystemExit(
            subprocess.call([os.fspath(part) for part in command], env=env)
        )
    os.execve(
        python,
        [os.fspath(part) for part in command],
        env,
    )


def _profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="whisper",
        help="dependency and default speech-engine profile (default: whisper)",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser(
        "install",
        help="sync a locked audio environment and pinned model source",
    )
    _profile_argument(install_parser)
    install_parser.add_argument("--force", action="store_true")
    install_parser.add_argument("--offline", action="store_true")
    install_parser.add_argument("--json", action="store_true")

    check_parser = subparsers.add_parser(
        "check",
        help="report whether the selected runtime is ready",
    )
    _profile_argument(check_parser)
    check_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser(
        "run",
        help="install on first use, then run the local audio sidecar",
    )
    _profile_argument(run_parser)
    run_parser.add_argument("--no-install", action="store_true")
    run_parser.add_argument("--offline", action="store_true")
    run_parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to audio_perception_server.py after --",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        if args.command == "install":
            result = install(args.profile, force=args.force, offline=args.offline)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(
                    f"audio profile {args.profile}: {result['status']} "
                    f"in {result['elapsed_seconds']}s\n"
                    f"environment: {result['venv']}\n"
                    f"model cache: {result['models']}"
                )
        elif args.command == "check":
            result = check(args.profile)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"audio profile {args.profile}: {result['status']}")
                print(f"environment: {result['venv']}")
                print(f"model cache: {result['models']}")
                if not result["ffmpeg"]:
                    print("missing: ffmpeg", file=sys.stderr)
                if not result["git"]:
                    print("missing: git", file=sys.stderr)
                if result["status"] != "ready":
                    raise SystemExit(1)
        elif args.command == "run":
            server_args = list(args.server_args)
            if server_args[:1] == ["--"]:
                server_args = server_args[1:]
            run_server(
                args.profile,
                no_install=args.no_install,
                offline=args.offline,
                server_args=server_args,
            )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"audio runtime error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
