from aegis_hgx.utils.config import ProjectConfig, load_project_config
from aegis_hgx.utils.logging import configure_logging, get_logger
from aegis_hgx.utils.seeds import set_global_seed


def test_project_config_loads() -> None:
    config = load_project_config()

    assert isinstance(config, ProjectConfig)
    assert config.project.name == "aegis-hgx"
    assert config.project.default_seed >= 0
    assert str(config.paths.data_processed) == "data/processed"


def test_logging_can_be_configured() -> None:
    configure_logging(level="INFO")

    logger = get_logger(__name__)

    logger.warning(__name__)

    assert logger.name == __name__


def test_global_seed_accepts_valid_seed() -> None:
    set_global_seed(42)
