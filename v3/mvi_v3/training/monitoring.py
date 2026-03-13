from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class TrainingMonitor:
    """Machine-readable training progress monitor.

    Maintains two files:
    - status.json: atomically updated current state (overwritten each update)
    - training.jsonl: append-only event log
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_dir / "status.json"
        self.log_path = self.output_dir / "training.jsonl"
        self._status: dict[str, object] = {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def _atomic_write_json(self, path: Path, data: dict[str, object]) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.write("\n")
            Path(tmp).rename(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def update_status(self, **kwargs: object) -> None:
        self._status.update(kwargs)
        self._status["timestamp"] = self._now_iso()
        self._atomic_write_json(self.status_path, self._status)

    def log_event(self, event_type: str, **kwargs: object) -> None:
        entry = {"event": event_type, "timestamp": self._now_iso(), **kwargs}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self.update_status(state="training", epoch=epoch, total_epochs=total_epochs)
        self.log_event("epoch_start", epoch=epoch)

    def on_epoch_end(
        self,
        *,
        epoch: int,
        total_epochs: int,
        global_step: int,
        total_steps_estimate: int,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        best_epoch: int,
        learning_rate: float,
        early_stop_counter: int,
        patience: int,
        epoch_duration_sec: float,
        eta_sec: float,
    ) -> None:
        self.update_status(
            state="training",
            epoch=epoch,
            total_epochs=total_epochs,
            global_step=global_step,
            total_steps_estimate=total_steps_estimate,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            best_val_loss=round(best_val_loss, 6),
            best_epoch=best_epoch,
            learning_rate=learning_rate,
            early_stop_counter=early_stop_counter,
            patience=patience,
            epoch_duration_sec=round(epoch_duration_sec, 1),
            eta_sec=round(eta_sec, 0),
        )
        self.log_event(
            "epoch_end",
            epoch=epoch,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            learning_rate=learning_rate,
        )

        # Console output
        print(
            f"[EPOCH {epoch}/{total_epochs}] "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"lr={learning_rate:.2e} "
            f"best={best_val_loss:.4f}@{best_epoch} "
            f"patience={early_stop_counter}/{patience}"
        )

    def on_checkpoint_saved(self, path: str, reason: str = "periodic") -> None:
        self.log_event("checkpoint_saved", path=path, reason=reason)

    def on_early_stop(self, epoch: int, best_epoch: int, best_val_loss: float) -> None:
        self.update_status(state="early_stopped")
        self.log_event(
            "early_stop",
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=round(best_val_loss, 6),
        )
        print(f"Early stopping at epoch {epoch}. Best: {best_val_loss:.4f} @ epoch {best_epoch}")

    def on_training_complete(self, epoch: int, best_epoch: int, best_val_loss: float) -> None:
        self.update_status(state="completed")
        self.log_event(
            "training_complete",
            final_epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=round(best_val_loss, 6),
        )
        print(f"Training complete at epoch {epoch}. Best: {best_val_loss:.4f} @ epoch {best_epoch}")
