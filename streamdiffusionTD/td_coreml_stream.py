from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


class StreamDiffusionCoreMLExt:
    def __init__(self, ownerComp: Any) -> None:
        self.ownerComp = ownerComp
        self._runner = None
        self._worker_module = None
        self._frame_index = 0
        self._last_generation = -1
        self._last_published_image = None
        self._last_framing = None
        self._last_sharpness = None

    def onInitTD(self) -> None:
        self._runner = None
        self._worker_module = None
        self._frame_index = 0
        self._last_generation = -1
        self._last_published_image = None
        self._last_framing = None
        self._last_sharpness = None

    def onDestroyTD(self) -> None:
        if self._runner is not None:
            self._runner.stop()
            self._runner = None

    def _get_runner(self):
        if self._runner is None:
            import importlib
            root = Path(str(self.ownerComp.par.Modelroot.eval())).expanduser()
            folder = str(root / 'streamdiffusionTD')
            if folder not in sys.path:
                sys.path.insert(0, folder)
            if self._worker_module is None:
                worker_module = sys.modules.get('td_coreml_worker')
                if worker_module is not None:
                    importlib.reload(worker_module)
                module = sys.modules.get('td_coreml_async')
                self._worker_module = importlib.reload(module) if module is not None else importlib.import_module('td_coreml_async')
            self._runner = self._worker_module.StreamRunner(root)
        return self._runner

    def Start(self) -> str:
        if not bool(self.ownerComp.par.Enabled.eval()):
            return 'Enable StreamDiffusion first'
        if self._runner is not None:
            self._runner.stop()
        self._runner = None
        self._last_generation = -1
        self._last_published_image = None
        self._last_framing = None
        self._last_sharpness = None
        runner = self._get_runner()
        runner.start()
        self.ownerComp.store('running', True)
        self.ownerComp.store('error', 'Loading SDXS / CoreML models…')
        return 'Worker startup requested'

    def Stop(self) -> str:
        if self._runner is not None:
            self._runner.stop()
        self._last_generation = -1
        self._last_published_image = None
        self._last_framing = None
        self._last_sharpness = None
        self._frame_index = 0
        self.ownerComp.store('running', False)
        self.ownerComp.store('ready', False)
        self.ownerComp.store('fps', 0.0)
        self.ownerComp.store('error', 'Disabled')
        return 'Worker stopped'

    def CookImage(self, script_op: Any) -> None:
        import traceback
        try:
            if not bool(self.ownerComp.par.Enabled.eval()):
                if bool(self.ownerComp.fetch('running', False)) and self._runner is not None:
                    self._runner.stop()
                self.ownerComp.store('running', False)
                self.ownerComp.store('ready', False)
                self.ownerComp.store('fps', 0.0)
                self.ownerComp.store('error', 'Disabled')
                self._last_generation = -1
                self._last_published_image = None
                self._last_framing = None
                self._last_sharpness = None
                self._frame_index = 0
                self._publish_passthrough(script_op)
                return
            runner = self._get_runner()
            runner.start()
            self.ownerComp.store('running', True)
            source = self.ownerComp.parent().op('resolution_input')
            if source is None or source.width < 1 or source.height < 1:
                self.ownerComp.store('error', 'Input 1 is not connected or has no image')
                self.ownerComp.store('ready', False)
                self._publish_passthrough(script_op)
                return
            self._frame_index += 1
            input_rate = max(1, int(self.ownerComp.par.Inputrate.eval()))
            if self._frame_index % input_rate == 0:
                frame = source.numpyArray(delayed=True)
                if frame is not None and frame.size:
                    rgb = np.asarray(frame[..., :3], dtype=np.float32)
                    if rgb.max() <= 1.0:
                        rgb *= 255.0
                    rgb = np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
                    runner.submit(
                        np.ascontiguousarray(rgb),
                        str(self.ownerComp.par.Prompt.eval()),
                        float(self.ownerComp.par.Feedback.eval()),
                    )
            framing = str(self.ownerComp.par.Framing.eval())
            sharpness = float(self.ownerComp.par.Sharpness.eval())
            if (framing, sharpness) != (self._last_framing, self._last_sharpness):
                runner.set_quality(framing, sharpness)
                self._last_framing = framing
                self._last_sharpness = sharpness
            image, generation, fps, ready, error = runner.latest()
            self.ownerComp.store('ready', bool(ready))
            self.ownerComp.store('fps', float(fps))
            self.ownerComp.store('error', error or ('Loading SDXS / CoreML models…' if not ready else ''))
            if image is None:
                self._publish_passthrough(script_op)
                return
            if generation == self._last_generation:
                if self._last_published_image is not None:
                    script_op.copyNumpyArray(self._last_published_image)
                return
            width = int(self.ownerComp.par.Outputwidth.eval())
            height = int(self.ownerComp.par.Outputheight.eval())
            image = self._resize_rgb(image, width, height)
            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = image
            rgba[..., 3] = 255
            published = np.ascontiguousarray(np.flipud(rgba))
            script_op.copyNumpyArray(published)
            self._last_generation = generation
            self._last_published_image = published
        except Exception as exc:
            self.ownerComp.store('error', f'{type(exc).__name__}: {exc}'[:1000])
            self.ownerComp.store('traceback', traceback.format_exc())
            self.ownerComp.store('ready', False)
            self.ownerComp.store('fps', 0.0)
            self._publish_passthrough(script_op)

    @staticmethod
    def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
        source_height, source_width = image.shape[:2]
        if source_width == width and source_height == height:
            return np.asarray(image, dtype=np.uint8)
        x = np.linspace(0, source_width - 1, width, dtype=np.float32)
        y = np.linspace(0, source_height - 1, height, dtype=np.float32)
        x0 = np.floor(x).astype(np.intp)
        y0 = np.floor(y).astype(np.intp)
        x1 = np.minimum(x0 + 1, source_width - 1)
        y1 = np.minimum(y0 + 1, source_height - 1)
        wx = (x - x0)[None, :, None]
        wy = (y - y0)[:, None, None]
        top = image[y0[:, None], x0[None, :]] * (1.0 - wx) + image[y0[:, None], x1[None, :]] * wx
        bottom = image[y1[:, None], x0[None, :]] * (1.0 - wx) + image[y1[:, None], x1[None, :]] * wx
        return np.rint(top * (1.0 - wy) + bottom * wy).clip(0, 255).astype(np.uint8)

    def _publish_passthrough(self, script_op: Any, frame: Any = None) -> None:
        if frame is None:
            source = self.ownerComp.parent().op('resolution_input')
            if source is not None:
                frame = source.numpyArray(delayed=True)
                if frame is None:
                    frame = source.numpyArray()
        if frame is None or not frame.size:
            width = int(self.ownerComp.par.Outputwidth.eval())
            height = int(self.ownerComp.par.Outputheight.eval())
            blank = np.zeros((height, width, 4), dtype=np.uint8)
            self._last_published_image = blank
            self._last_generation = -1
            script_op.copyNumpyArray(blank)
            return
        frame = np.asarray(frame)
        if frame.dtype.kind == 'f':
            frame = np.rint(np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255)).astype(np.uint8)
        else:
            frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape[-1] == 3:
            rgba = np.empty((*frame.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = frame
            rgba[..., 3] = 255
        else:
            rgba = np.asarray(frame[..., :4], dtype=np.uint8)
        published = np.ascontiguousarray(np.flipud(rgba))
        self._last_published_image = published
        self._last_generation = -1
        script_op.copyNumpyArray(published)

    def CookMetrics(self, script_op: Any) -> None:
        script_op.clear()
        fps = script_op.appendChan('fps')
        fps[0] = float(self.ownerComp.fetch('fps',0.0))
        ready = script_op.appendChan('ready')
        ready[0] = 1.0 if self.ownerComp.fetch('ready',False) else 0.0
