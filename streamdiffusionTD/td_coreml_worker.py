"""TouchDesigner-independent client for the local CoreML inference service."""

from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


META_LENGTH = struct.Struct(">I")
HEADER = struct.Struct(">IIf")
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 13317


class CoreMLWorker:
    def __init__(self, project_root: Path = DEFAULT_PROJECT_ROOT) -> None:
        self.project_root = Path(project_root)
        self.port = int(os.environ.get("STREAMDIFFUSION_PORT", DEFAULT_PORT))
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._prompt = os.environ.get(
            "STREAMDIFFUSION_PROMPT", "oil painting style, masterpiece, highly detailed"
        )
        self._feedback = float(os.environ.get("STREAMDIFFUSION_FEEDBACK", "0.3"))
        self.last_fps = 0.0
        self.last_error = ""
        self._quality: tuple[str, float] | None = None
        self._quality_lock = threading.Lock()

    def start(self) -> None:
        if self.is_ready():
            return
        python = self.project_root / ".coreml.venv" / "bin" / "python"
        script = self.project_root / "streamdiffusionTD" / "td_coreml_server.py"
        if not python.is_file():
            raise FileNotFoundError(f"CoreML Python environment missing: {python}")
        env = os.environ.copy()
        env["HF_HOME"] = str(self.project_root / "models")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        self._process = subprocess.Popen(
            [
                str(python),
                str(script),
                "--project-root",
                str(self.project_root),
                "--port",
                str(self.port),
            ],
            cwd=str(self.project_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"SDXS worker exited during startup (code {self._process.returncode}); "
                    "inspect the worker output in a terminal for details."
                )
            if self.is_ready():
                return
            time.sleep(0.5)
        raise TimeoutError("Timed out while loading SDXS and CoreML models")

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=0.5) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def process(
        self, rgba: np.ndarray, prompt: str | None = None, feedback: float | None = None
    ) -> tuple[np.ndarray | None, float]:
        rgb = np.ascontiguousarray(rgba[..., :3], dtype=np.uint8)
        if prompt is None or feedback is None:
            configured_prompt, configured_feedback = self.settings()
            prompt = configured_prompt if prompt is None else prompt
            feedback = configured_feedback if feedback is None else feedback
        metadata = json.dumps(
            {
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "prompt": prompt,
                "feedback": feedback,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request_data = META_LENGTH.pack(len(metadata)) + metadata + rgb.tobytes()
        request = urllib.request.Request(
            f"{self.base_url}/frame",
            data=request_data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            width, height, fps = HEADER.unpack_from(payload)
            expected = HEADER.size + width * height * 3
            if len(payload) != expected:
                raise ValueError("Worker returned an incomplete image payload")
            output = np.frombuffer(payload, dtype=np.uint8, offset=HEADER.size).reshape(
                height, width, 3
            ).copy()
            self.last_fps = float(fps)
            self.last_error = ""
            return output, self.last_fps
        except (OSError, urllib.error.URLError, ValueError, struct.error) as exc:
            self.last_error = str(exc)
            return None, self.last_fps

    def set_quality(self, framing: str, sharpness: float) -> None:
        quality = ("fit" if framing == "fit" else "center_crop", min(max(float(sharpness), 0.0), 1.0))
        with self._quality_lock:
            if quality == self._quality:
                if self.is_ready():
                    return
                self._quality = None
        payload = json.dumps(
            {"framing": quality[0], "sharpness": quality[1]},
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/quality",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                if response.status != 200:
                    raise RuntimeError(f"Worker rejected quality controls: {response.status}")
            with self._quality_lock:
                self._quality = quality
        except (OSError, urllib.error.URLError, ValueError, RuntimeError) as exc:
            self.last_error = str(exc)

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            self._prompt = prompt

    def set_feedback(self, feedback: float) -> None:
        with self._lock:
            self._feedback = min(max(float(feedback), 0.0), 0.95)

    def settings(self) -> tuple[str, float]:
        with self._lock:
            return self._prompt, self._feedback
