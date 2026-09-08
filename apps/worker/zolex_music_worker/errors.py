class WorkerError(RuntimeError):
    """Base worker failure."""


class ValidationError(WorkerError):
    """Invalid request, media, configuration, or generated manifest."""


class ExternalCommandError(WorkerError):
    """A supervised external command failed."""


class QualityError(WorkerError):
    """A rendered artifact did not pass required quality checks."""

