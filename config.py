import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Anish UC Project"
    VERSION: str = "1.0.0"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_UPLOAD_SIZE_MB: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
