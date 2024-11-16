from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "SustAIn LLM"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/llm"


settings = Settings()
