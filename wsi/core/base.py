import logging
from pathlib import Path
from typing import Union

import numpy

from . import utils

class OutputObject:
    def __init__(
        self, name: str, output_dir_path: Path, level: int = logging.DEBUG, **kw: object
    ):
        self._name = name
        self._output_dir_path = output_dir_path
        self._log_name = f"{self.__class__.__name__} [{self._name}]"
        self._log_file_path = output_dir_path / f"{name}.log"
        self._output_dir_path.mkdir(parents=True, exist_ok=True)
        self._logger = utils.create_logger(
            log_file_path=self._log_file_path, name=self._log_name, level=level
        )
        super(OutputObject, self).__init__(**kw)
