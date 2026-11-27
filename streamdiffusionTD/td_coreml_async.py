"""Background latest-frame-wins bridge from TouchDesigner to CoreML."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from td_coreml_worker import CoreMLWorker


class StreamRunner:
    def __init__(self, project_root: Path) -> None:
        self._client = CoreMLWorker(project_root)
        self._frames: queue.Queue[tuple[np.ndarray, str, float]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._latest_image: np.ndarray | None = None
        self._image_generation = 0
        self._latest_fps = 0.0
        self._latest_ready = False
        self._latest_error = "Worker stopped"
        self._quality: tuple[str, float] | None = None
        self._enabled = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._start_lock:
            self._closed.clear()
            self._enabled.set()
            should_start = self._thread is None or not self._thread.is_alive()
            if should_start:
                self._thread = threading.Thread(
                    target=self._run, name="StreamDiffusionCoreML", daemon=True
                )
                self._thread.start()
            if self._startup_thread is None or not self._startup_thread.is_alive():
                self._startup_thread = threading.Thread(
                    target=self._initialize_worker,
                    name="StreamDiffusionStartup",
                    daemon=True,
                )
                self._startup_thread.start()

    def set_quality(self, framing: str, sharpness: float) -> None:
        quality = ("fit" if framing == "fit" else "center_crop", min(max(float(sharpness), 0.0), 1.0))
        with self._lock:
            if quality == self._quality:
                return
            self._quality = quality
        threading.Thread(
            target=self._client.set_quality,
            args=quality,
            name="StreamDiffusionQuality",
            daemon=True,
        ).start()

    def _initialize_worker(self) -> None:
        try:
            self._client.start()
            if self._quality is not None:
                quality = self._quality
                threading.Thread(
                    target=self._client.set_quality,
                    args=quality,
                    name="StreamDiffusionQuality",
                    daemon=True,
                ).start()
            with self._lock:
                self._latest_ready = True
                self._latest_error = ""
        except Exception as exc:
            self._client.stop()
            with self._lock:
                self._latest_ready = False
                self._latest_error = f"{type(exc).__name__}: {exc}"[:1000]

    def stop(self) -> None:
        self._enabled.clear()
        self._closed.set()
        self._client.stop()
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._latest_image = None
            self._latest_fps = 0.0
            self._latest_ready = False
            self._latest_error = "Worker stopped"
            self._image_generation += 1

    def submit(self, frame: np.ndarray, prompt: str, feedback: float) -> None:
        if not self._enabled.is_set():
            return
        item = (
            np.ascontiguousarray(frame, dtype=np.uint8),
            str(prompt),
            min(max(float(feedback), 0.0), 0.95),
        )
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(item)
            except queue.Full:
                pass

    def latest(self) -> tuple[np.ndarray | None, int, float, bool, str]:
        with self._lock:
            return (
                self._latest_image,
                self._image_generation,
                self._latest_fps,
                self._latest_ready,
                self._latest_error,
            )

    def _run(self) -> None:
        while not self._closed.is_set():
            if not self._enabled.wait(timeout=0.1):
                continue
            if not self._client.is_ready():
                try:
                    self._client.start()
                    if self._quality is not None:
                        quality = self._quality
                        threading.Thread(
                            target=self._client.set_quality,
                            args=quality,
                            name="StreamDiffusionQuality",
                            daemon=True,
                        ).start()
                    with self._lock:
                        self._latest_ready = True
                        self._latest_error = ""
                except Exception as exc:
                    self._client.stop()
                    with self._lock:
                        self._latest_ready = False
                        self._latest_error = f"{type(exc).__name__}: {exc}"[:1000]
                    self._closed.wait(2.0)
                    continue
            try:
                frame, prompt, feedback = self._frames.get(timeout=0.25)
            except queue.Empty:
                continue
            self._client.set_prompt(prompt)
            self._client.set_feedback(feedback)
            image, fps = self._client.process(frame)
            with self._lock:
                if image is not None:
                    self._latest_image = image
                    self._image_generation += 1
                    self._latest_fps = fps
                    self._latest_ready = True
                    self._latest_error = ""
                else:
                    self._latest_ready = self._client.is_ready()
                    self._latest_error = self._client.last_error
