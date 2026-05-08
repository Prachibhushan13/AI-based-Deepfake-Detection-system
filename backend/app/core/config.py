from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI-Based Deepfake Detection System"
    api_v1_prefix: str = "/api/v1"
    secret_key: str = Field("change-me-to-a-long-random-string", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(1440, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    mongodb_uri: str = Field("mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_db: str = Field("deepfake_detection", alias="MONGODB_DB")
    upload_dir: str = Field("backend/uploads", alias="UPLOAD_DIR")
    artifact_dir: str = Field("backend/artifacts", alias="ARTIFACT_DIR")
    model_path: str = Field("backend/artifacts/models/cnn_lstm_best.keras", alias="MODEL_PATH")
    allowed_origins: str = Field("http://localhost:5173", alias="ALLOWED_ORIGINS")
    rate_limit: str = Field("30/minute", alias="RATE_LIMIT")
    frame_stride: int = Field(5, alias="FRAME_STRIDE")
    sequence_length: int = Field(20, alias="SEQUENCE_LENGTH")
    image_size: int = Field(224, alias="IMAGE_SIZE")
    max_upload_mb: int = Field(250, alias="MAX_UPLOAD_MB")
    enable_mock_model: bool = Field(True, alias="ENABLE_MOCK_MODEL")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def upload_path(self) -> Path:
        return self.resolve_path(self.upload_dir)

    @property
    def artifact_path(self) -> Path:
        return self.resolve_path(self.artifact_dir)

    @property
    def model_artifact_path(self) -> Path:
        return self.resolve_path(self.model_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
