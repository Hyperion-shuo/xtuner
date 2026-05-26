import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ):
        if log_dir is None:
            log_dir = Path()

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self._wandb = None
        self._wandb_run = None
        if self._should_sync_wandb():
            self._init_wandb(log_dir)

        self._writer = SummaryWriter(log_dir=log_dir)

    def _should_sync_wandb(self) -> bool:
        if os.environ.get("XTUNER_WANDB_SYNC_TENSORBOARD") != "1":
            return False

        for rank_env in ("RANK", "RAY_RANK", "LOCAL_RANK"):
            rank = os.environ.get(rank_env)
            if rank not in (None, "", "0"):
                return False
        return True

    def _init_wandb(self, log_dir: Path):
        self._drop_empty_wandb_env()
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "XTUNER_WANDB_SYNC_TENSORBOARD=1 but wandb is not installed in the current Python environment. "
                "Install wandb or unset XTUNER_WANDB_SYNC_TENSORBOARD."
            ) from exc

        try:
            self._wandb = wandb
            self._wandb_run = wandb.init(
                project=self._get_env("WANDB_PROJECT", "xtuner"),
                name=self._get_env("WANDB_NAME"),
                dir=self._get_env("WANDB_DIR", str(log_dir)),
                sync_tensorboard=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize wandb with sync_tensorboard=True. "
                "Check WANDB_* environment variables, wandb login state, or unset XTUNER_WANDB_SYNC_TENSORBOARD."
            ) from exc

    def _get_env(self, name: str, default: str | None = None) -> str | None:
        return os.environ.get(name) or default

    def _drop_empty_wandb_env(self):
        for name in (
            "WANDB_PROJECT",
            "WANDB_NAME",
            "WANDB_DIR",
            "WANDB_MODE",
            "WANDB_ENTITY",
            "WANDB_CACHE_DIR",
            "WANDB_CONFIG_DIR",
        ):
            if os.environ.get(name) == "":
                os.environ.pop(name)

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ):
        self._writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ):
        for tag, scalar_value in tag_scalar_dict.items():
            self._writer.add_scalar(tag, scalar_value, global_step)

    def close(self):
        self._writer.close()
        if self._wandb_run is not None:
            self._wandb.finish()
            self._wandb_run = None
