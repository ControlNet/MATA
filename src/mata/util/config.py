from typing import Any
from tensorneko_util.util import Singleton
import tensorneko_util as N


@Singleton
class Config:

    def __init__(self):
        self._base_config_path: str | None = None
        self.model_config: dict[str, Any] = dict()
        self.base_config: dict[str, Any] = dict()

    @property
    def base_config_path(self):
        return self._base_config_path

    @base_config_path.setter
    def base_config_path(self, value):
        if value is not None:
            self._base_config_path = value
            self.base_config = N.read(value)

    @property
    def debug(self):
        return Config.base_config["debug"]

    @debug.setter
    def debug(self, value):
        Config.base_config["debug"] = value
