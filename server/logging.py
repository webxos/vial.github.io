import logging
import json


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - %(message)s - %(extra)s"
    ),
)
_base_logger = logging.getLogger(__name__)


class LogHandler:
    def __init__(self, base_logger: logging.Logger):
        self._logger = base_logger

    def log(
        self,
        message: str,
        request_id: str | None = None,
        extra: dict | None = None,
    ):
        extra_dict = extra or {}
        if request_id:
            extra_dict["request_id"] = request_id
        self._logger.info(
            message,
            extra={"extra": json.dumps(extra_dict)},
        )

    def info(
        self,
        message: str,
        request_id: str | None = None,
        extra: dict | None = None,
    ):
        self.log(message, request_id=request_id, extra=extra)

    def error(
        self,
        message: str,
        request_id: str | None = None,
        extra: dict | None = None,
    ):
        extra_dict = extra or {}
        if request_id:
            extra_dict["request_id"] = request_id
        self._logger.error(
            message,
            extra={"extra": json.dumps(extra_dict)},
        )


logger = LogHandler(_base_logger)
