from safecart_ai.config import Settings, get_settings


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings()

    assert settings.env == "development"
    assert str(settings.data_path).replace("\\", "/") == "data/sample"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
