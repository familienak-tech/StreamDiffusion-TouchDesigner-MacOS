"""Local SDXS/TAESD CoreML worker for TouchDesigner image-to-image frames."""

from __future__ import annotations

import argparse
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


HEADER = struct.Struct(">IIf")
META_LENGTH = struct.Struct(">I")
RENDER_SIZE = 512


class SDXSPipeline:
    def __init__(self, project_root: Path) -> None:
        import coremltools as ct
        import torch
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        self._torch = torch
        self._prompt_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._prompt = ""
        self._prompt_embeds = None
        self._prev_denoised = None
        self._feedback = 0.3
        self._framing = "center_crop"
        self._sharpness = 0.0
        self._fixed_noise = np.random.RandomState(42).randn(
            1, 4, RENDER_SIZE // 8, RENDER_SIZE // 8
        ).astype(np.float16)
        self._image_buffer = np.empty(
            (1, 3, RENDER_SIZE, RENDER_SIZE), dtype=np.float16
        )
        self._latent_buffer = np.empty(
            (1, 4, RENDER_SIZE // 8, RENDER_SIZE // 8), dtype=np.float16
        )
        self._output_buffer = np.empty_like(self._latent_buffer)
        self._normalization_lut = (
            np.arange(256, dtype=np.float32) / 127.5 - 1.0
        ).astype(np.float16)

        model_dir = project_root / "coreml_models"
        self._encoder = ct.models.MLModel(
            str(model_dir / "taesd_encoder_512.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        self._decoder = ct.models.MLModel(
            str(model_dir / "taesd_decoder.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        self._unet = ct.models.MLModel(
            str(model_dir / "unet_sdxs_512.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        for name, model, sample in (
            (
                "TAESD encoder",
                self._encoder,
                {"image": np.zeros((1, 3, RENDER_SIZE, RENDER_SIZE), dtype=np.float16)},
            ),
            (
                "SDXS UNet",
                self._unet,
                {
                    "sample": np.zeros((1, 4, 64, 64), dtype=np.float16),
                    "timestep": np.zeros((1,), dtype=np.float16),
                    "encoder_hidden_states": np.zeros((1, 77, 1024), dtype=np.float16),
                },
            ),
            (
                "TAESD decoder",
                self._decoder,
                {"latent": np.zeros((1, 4, 64, 64), dtype=np.float16)},
            ),
        ):
            model.predict(sample)
            print(f"Warmed up {name}", flush=True)

        model_id = "IDKiro/sdxs-512-0.9"
        self._text_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        ).to("mps")
        self._tokenizer = self._text_pipe.tokenizer
        self._text_encoder = self._text_pipe.text_encoder
        scheduler = EulerDiscreteScheduler.from_config(
            self._text_pipe.scheduler.config
        )
        scheduler.set_timesteps(1, device="mps")
        timestep = scheduler.timesteps[0].cpu().item()
        alpha = scheduler.alphas_cumprod[
            min(int(timestep), len(scheduler.alphas_cumprod) - 1)
        ].item()
        self._timestep = np.array([timestep], dtype=np.float16)
        self._sqrt_alpha = np.float16(np.sqrt(alpha))
        self._sqrt_one_minus_alpha = np.float16(np.sqrt(1.0 - alpha))
        self._inference_times: list[float] = []
        self._frame_count = 0
        self._frame_lock = threading.Lock()

    def _encode_prompt(self, prompt: str) -> np.ndarray:
        tokens = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            embeds = self._text_encoder(tokens.input_ids.to("mps"))[0]
        return embeds.cpu().to(self._torch.float16).numpy()

    def set_quality(self, framing: str, sharpness: float) -> None:
        mode = framing if framing in ("center_crop", "fit") else "center_crop"
        with self._inference_lock:
            self._framing = mode
            self._sharpness = min(max(float(sharpness), 0.0), 1.0)

    def process(self, frame: np.ndarray, prompt: str, feedback: float) -> tuple[np.ndarray, float]:
        with self._inference_lock:
            return self._process_frame(
                frame, prompt, feedback, self._framing, self._sharpness
            )

    def _process_frame(
        self,
        frame: np.ndarray,
        prompt: str,
        feedback: float,
        framing: str,
        sharpness: float,
    ) -> tuple[np.ndarray, float]:
        if prompt != self._prompt:
            self._prompt_embeds = self._encode_prompt(prompt)
            self._prompt = prompt
            self._prev_denoised = None

        image = Image.fromarray(frame, mode="RGB")
        width, height = image.size
        if framing == "fit":
            scale = min(RENDER_SIZE / width, RENDER_SIZE / height)
            fit_width = max(1, int(round(width * scale)))
            fit_height = max(1, int(round(height * scale)))
            resized = image.resize((fit_width, fit_height), Image.Resampling.LANCZOS)
            image = Image.new("RGB", (RENDER_SIZE, RENDER_SIZE))
            image.paste(
                resized,
                ((RENDER_SIZE - fit_width) // 2, (RENDER_SIZE - fit_height) // 2),
            )
        else:
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize(
                (RENDER_SIZE, RENDER_SIZE), Image.Resampling.LANCZOS
            )
        rgb = np.asarray(image, dtype=np.uint8)
        np.copyto(
            self._image_buffer,
            self._normalization_lut[rgb].transpose(2, 0, 1)[np.newaxis],
        )

        start = time.perf_counter()
        encoded = self._encoder.predict({"image": self._image_buffer})
        clean = np.asarray(encoded["latent"], dtype=np.float16)
        if self._prev_denoised is not None and feedback > 0:
            blend = np.float16(min(max(feedback, 0.0), 0.95))
            clean = (1.0 - blend) * clean + blend * self._prev_denoised

        noisy = (
            self._sqrt_alpha * clean
            + self._sqrt_one_minus_alpha * self._fixed_noise
        )
        np.copyto(self._latent_buffer, noisy)
        prediction = self._unet.predict(
            {
                "sample": self._latent_buffer,
                "timestep": self._timestep,
                "encoder_hidden_states": self._prompt_embeds,
            }
        )
        noise = np.asarray(prediction["noise_pred"], dtype=np.float16)
        denoised = (
            noisy - self._sqrt_one_minus_alpha * noise
        ) / self._sqrt_alpha
        self._prev_denoised = denoised.copy()
        np.copyto(self._output_buffer, denoised)

        decoded = self._decoder.predict({"latent": self._output_buffer})
        output = np.asarray(decoded["image"], dtype=np.float32)
        output = output.squeeze(0).transpose(1, 2, 0)
        output = ((output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        sharpness = min(max(float(sharpness), 0.0), 1.0)
        if sharpness > 0.0:
            source = Image.fromarray(output, mode="RGB")
            softened = np.asarray(
                source.filter(ImageFilter.GaussianBlur(radius=0.7)), dtype=np.float32
            )
            output = np.rint(
                np.clip(
                    output.astype(np.float32)
                    + (output.astype(np.float32) - softened) * sharpness,
                    0,
                    255,
                )
            ).astype(np.uint8)
        elapsed = time.perf_counter() - start
        self._inference_times.append(elapsed)
        del self._inference_times[:-30]
        with self._frame_lock:
            self._frame_count += 1
            frame_count = self._frame_count
        fps = 1.0 / max(sum(self._inference_times) / len(self._inference_times), 1e-6)
        return output, fps


def create_handler(pipeline: SDXSPipeline) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            with pipeline._frame_lock:
                frame_count = pipeline._frame_count
            body = json.dumps({"ready": True, "frames": frame_count}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path == "/quality":
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length <= 0 or content_length > 2048:
                        raise ValueError("invalid quality-control payload length")
                    controls = json.loads(self.rfile.read(content_length))
                    pipeline.set_quality(
                        str(controls.get("framing", "center_crop")),
                        float(controls.get("sharpness", 0.0)),
                    )
                    payload = json.dumps({"ok": True}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:
                    payload = json.dumps({"error": str(exc)}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                return
            if self.path != "/frame":
                self.send_error(404)
                return
            try:
                meta_size = META_LENGTH.unpack(self.rfile.read(META_LENGTH.size))[0]
                if meta_size <= 0 or meta_size > 8192:
                    raise ValueError("invalid metadata length")
                meta = json.loads(self.rfile.read(meta_size))
                width, height = int(meta["width"]), int(meta["height"])
                if width <= 0 or height <= 0 or width * height > 8_294_400:
                    raise ValueError("invalid frame dimensions")
                byte_count = width * height * 3
                frame_bytes = self.rfile.read(byte_count)
                if len(frame_bytes) != byte_count:
                    raise ValueError("incomplete frame payload")
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                    height, width, 3
                )
                output, fps = pipeline.process(
                    frame,
                    str(meta.get("prompt", ""))[:2048],
                    float(meta.get("feedback", 0.3)),
                )
                body = HEADER.pack(output.shape[1], output.shape[0], fps) + output.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--port", default=13317, type=int)
    args = parser.parse_args()

    root = args.project_root.resolve()
    model_dir = root / "coreml_models"
    required = (
        model_dir / "unet_sdxs_512.mlpackage/Manifest.json",
        model_dir / "taesd_encoder_512.mlpackage/Manifest.json",
        model_dir / "taesd_decoder.mlpackage/Manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CoreML model artifacts: " + ", ".join(missing))

    pipeline = SDXSPipeline(root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), create_handler(pipeline))
    server.daemon_threads = True
    print(f"SDXS CoreML worker ready at 127.0.0.1:{args.port}", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
