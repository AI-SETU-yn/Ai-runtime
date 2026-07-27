from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra='forbid')

    name: str
    base_url: str = Field(validation_alias=AliasChoices('base_url', 'url'))
    transport: str = 'streamable-http'
    endpoint_path: str = '/mcp'

    @field_validator('name', 'base_url', 'transport', 'endpoint_path')
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('value must not be empty')
        return value

    @field_validator('base_url')
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip('/')

    @field_validator('endpoint_path')
    @classmethod
    def require_absolute_path(cls, value: str) -> str:
        if not value.startswith('/'):
            raise ValueError('endpoint_path must start with /')
        return value
