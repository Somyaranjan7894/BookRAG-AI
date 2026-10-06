"""Centralized hardware and inference device abstraction for BookRAG AI Phase 24.

Provides robust CUDA hardware detection, explicit target device resolution,
safe CPU fallback, VRAM introspection, mixed-precision (FP16) management,
and startup/runtime diagnostic telemetry.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union
import torch

from app.core import config
from app.core.logging import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """Manages inference hardware discovery, device assignment, and memory telemetry."""

    _instance: Optional["DeviceManager"] = None

    def __init__(self) -> None:
        self._cuda_available = torch.cuda.is_available()
        self._device_count = torch.cuda.device_count() if self._cuda_available else 0
        self._gpu_name: Optional[str] = None
        if self._cuda_available and self._device_count > 0:
            try:
                self._gpu_name = torch.cuda.get_device_name(0)
            except Exception as exc:
                logger.warning("Failed to query CUDA GPU device name: %s", exc)
                self._gpu_name = "NVIDIA CUDA Device"

    @classmethod
    def get_instance(cls) -> "DeviceManager":
        """Return the shared process-level DeviceManager singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_cuda_available(self) -> bool:
        """Return whether CUDA execution is supported by the current PyTorch runtime."""
        return self._cuda_available

    @property
    def gpu_name(self) -> Optional[str]:
        """Return the detected GPU device model name, or None if CUDA is unavailable."""
        return self._gpu_name

    @property
    def device_count(self) -> int:
        """Return the number of visible CUDA devices."""
        return self._device_count

    def get_vram_info(self, device_index: int = 0) -> Dict[str, float]:
        """Return detailed VRAM memory statistics in Megabytes if CUDA is available."""
        if not self._cuda_available or device_index >= self._device_count:
            return {
                "total_mb": 0.0,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
            }
        try:
            total_bytes = torch.cuda.get_device_properties(device_index).total_memory
            allocated_bytes = torch.cuda.memory_allocated(device_index)
            reserved_bytes = torch.cuda.memory_reserved(device_index)
            free_bytes = total_bytes - reserved_bytes

            return {
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "allocated_mb": round(allocated_bytes / (1024 * 1024), 2),
                "reserved_mb": round(reserved_bytes / (1024 * 1024), 2),
                "free_mb": round(free_bytes / (1024 * 1024), 2),
            }
        except Exception as exc:
            logger.warning("Failed to query VRAM info for device %d: %s", device_index, exc)
            return {
                "total_mb": 0.0,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
            }

    def resolve_device(self, requested_device: Optional[str] = None) -> str:
        """Determine execution device based on configuration and hardware availability.

        Resolution rules:
        - If requested_device is 'auto' or None:
            - If settings.DEVICE is specified and not 'auto', respect global settings.DEVICE.
            - Otherwise ('auto'): Uses 'cuda' if CUDA is available, otherwise safely falls back to 'cpu'.
        - 'cuda': Uses CUDA if available; logs a clear warning and gracefully falls back to 'cpu' if unavailable.
        - 'cuda:N': Uses specified CUDA device index if available and valid (0 <= N < device_count);
          logs a clear warning and falls back to 'cuda:0' (or 'cpu') if index is out of bounds or unavailable.
        - 'cpu': Strictly runs on CPU.
        - Unrecognized string: Logs a warning and safely falls back to 'cpu'.

        Never crashes simply because CUDA is unavailable.
        Never silently claims GPU usage when executing on CPU.

        Args:
            requested_device: Optional override ('auto', 'cuda', 'cpu', 'cuda:0').
                Defaults to settings.DEVICE.

        Returns:
            Canonical device string ('cpu' or 'cuda' or 'cuda:N').
        """
        raw = (requested_device or "").strip().lower()
        active_settings = getattr(config, "settings", None)
        global_setting = (getattr(active_settings, "DEVICE", None) or "auto").strip().lower()

        # If requested is None or empty, fall back to global setting
        if not raw:
            raw = global_setting

        # If requested is 'auto', inherit global setting if global is not 'auto'
        if raw == "auto":
            if global_setting != "auto":
                raw = global_setting
            else:
                return "cuda" if self._cuda_available else "cpu"

        if raw == "cpu":
            return "cpu"

        if raw == "cuda":
            if not self._cuda_available:
                logger.warning(
                    "CUDA device was explicitly requested ('cuda') but PyTorch CUDA support is unavailable. "
                    "Safely falling back to CPU execution."
                )
                return "cpu"
            return "cuda"

        if raw.startswith("cuda:"):
            if not self._cuda_available:
                logger.warning(
                    "CUDA device was explicitly requested ('%s') but PyTorch CUDA support is unavailable. "
                    "Safely falling back to CPU execution.",
                    raw,
                )
                return "cpu"
            idx_str = raw[len("cuda:"):]
            try:
                device_idx = int(idx_str)
                if 0 <= device_idx < self._device_count:
                    return raw
                logger.warning(
                    "Requested CUDA device index %d out of bounds (device_count=%d). Falling back to 'cuda:0'.",
                    device_idx,
                    self._device_count,
                )
                return "cuda:0" if self._device_count > 0 else "cpu"
            except ValueError:
                logger.warning("Malformed CUDA device specifier '%s'. Falling back to 'cuda'.", raw)
                return "cuda"

        logger.warning("Unrecognized device target '%s'. Safely falling back to CPU.", raw)
        return "cpu"

    def get_torch_device(self, requested_device: Optional[str] = None) -> torch.device:
        """Return a resolved torch.device instance."""
        dev_str = self.resolve_device(requested_device)
        return torch.device(dev_str)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Collect runtime device diagnostics for telemetry, logging, and health endpoints."""
        resolved = self.resolve_device()
        is_using_gpu = resolved.startswith("cuda")
        vram = self.get_vram_info() if self._cuda_available else {}

        return {
            "device": resolved.upper(),
            "selected_device": resolved,
            "cuda_available": self._cuda_available,
            "is_using_gpu": is_using_gpu,
            "gpu_name": self._gpu_name if self._cuda_available else None,
            "device_count": self._device_count,
            "vram_mb": vram,
            "pytorch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
        }

    def log_diagnostics(self) -> None:
        """Log a structured hardware and inference device diagnostics banner."""
        diag = self.get_diagnostics()
        if diag["is_using_gpu"]:
            vram = diag["vram_mb"]
            logger.info(
                "Inference Hardware [GPU]: device=%s, gpu_name='%s', vram_total=%.1f MB, free=%.1f MB, cuda_version=%s",
                diag["device"],
                diag["gpu_name"],
                vram.get("total_mb", 0.0),
                vram.get("free_mb", 0.0),
                diag["cuda_runtime_version"],
            )
        else:
            logger.info(
                "Inference Hardware [CPU]: device=%s, cuda_available=%s, gpu_name=%s, pytorch=%s",
                diag["device"],
                diag["cuda_available"],
                diag["gpu_name"] or "None",
                diag["pytorch_version"],
            )

    @contextmanager
    def inference_context(
        self,
        device: Union[str, torch.device],
        use_fp16: Optional[bool] = None,
    ) -> Generator[None, None, None]:
        """Scoped context manager for inference with automatic FP16 autocast on CUDA.

        Args:
            device: Active device string or torch.device.
            use_fp16: Whether to enable FP16 mixed precision. Defaults to settings.USE_FP16.
        """
        dev_str = str(device).lower()
        active_settings = getattr(config, "settings", None)
        default_fp16 = getattr(active_settings, "USE_FP16", True) if active_settings is not None else True
        enable_fp16 = default_fp16 if use_fp16 is None else use_fp16
        is_cuda_dev = "cuda" in dev_str and self._cuda_available

        if self._cuda_available and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float16

        with torch.inference_mode():
            if is_cuda_dev and enable_fp16:
                with torch.autocast(device_type="cuda", dtype=target_dtype):
                    yield
            else:
                yield

    def empty_cache(self) -> None:
        """Clear cached memory allocations on CUDA if available."""
        if self._cuda_available and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                logger.warning("Failed to empty CUDA cache: %s", exc)


# Module-level convenience functions preserving existing signatures
def get_device_manager() -> DeviceManager:
    """Return the DeviceManager singleton."""
    return DeviceManager.get_instance()


def resolve_device(requested_device: Optional[str] = None) -> str:
    """Canonical device resolution function matching previous signature."""
    return get_device_manager().resolve_device(requested_device)


def get_torch_device(requested_device: Optional[str] = None) -> torch.device:
    """Return resolved torch.device instance."""
    return get_device_manager().get_torch_device(requested_device)


def is_cuda_available() -> bool:
    """Return whether CUDA is available."""
    return get_device_manager().is_cuda_available


def get_gpu_name() -> Optional[str]:
    """Return GPU name if available."""
    return get_device_manager().gpu_name


def get_vram_info() -> Dict[str, float]:
    """Return VRAM info."""
    return get_device_manager().get_vram_info()


def empty_cache() -> None:
    """Clear cached memory allocations."""
    get_device_manager().empty_cache()


def inference_context(
    device: Union[str, torch.device],
    use_fp16: Optional[bool] = None,
) -> Any:
    """Return scoped inference context."""
    return get_device_manager().inference_context(device, use_fp16=use_fp16)
