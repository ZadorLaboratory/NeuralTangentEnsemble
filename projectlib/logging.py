import wandb
from wandb.wandb_run import Run
from dataclasses import dataclass
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

class PrintLogger:
    buffer: Dict[str, Any] = dict()

    @staticmethod
    def _print_item(item):
        if isinstance(item, dict):
            entry = ", ".join(f"{k}: {PrintLogger._print_item(v)}"
                              for k, v in item.items())

            return "{" + entry + "}"
        else:
            return str(item)

    def _print_buffer(self):
        log_entry = ", ".join(f"{k}: {PrintLogger._print_item(v)}"
                              for k, v in self.buffer.items())
        print(log_entry)

    def log(self, entry: Dict[str, Any], commit = True):
        self.buffer.update(entry)
        if commit:
            self._print_buffer()
            self.buffer.clear()

    def log_config(self, config: DictConfig):
        print("CONFIG:")
        print(OmegaConf.to_yaml(config))

    def finish(self):
        print("RUN COMPLETE")

@dataclass
class WandbLogger:
    internal_logger: Run

    @classmethod
    def init(cls, *args, **kwargs):
        wandb.login()
        run = wandb.init(*args, **kwargs)

        return cls(run)

    def log(self, entry: Dict[str, Any], commit = True):
        self.internal_logger.log(entry, commit=commit)

    def log_config(self, config: DictConfig):
        self.internal_logger.config.update(OmegaConf.to_container(config))

    def finish(self):
        return wandb.finish()
