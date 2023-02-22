from pydantic import BaseModel, FilePath
from typing import List


# class User(BaseModel):
#     id: int
#     name = "John Doe"
#     signup_ts: datetime | None = None
#     friends: list[int] = []


class sample(BaseModel):
    text: str
    label: bool


class model(object):
    random_state = 30481
    
    def __init__(self, model_file_path):
        if model_file_path:
            try:
                