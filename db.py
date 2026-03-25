import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Lädt lokale .env (wird nicht gepusht, weil in .gitignore)
load_dotenv()


def _need(name: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v


def _engine(prefix: str):
    host = _need(f"{prefix}_HOST")
    port = int(os.getenv(f"{prefix}_PORT", "3306"))
    dbname = _need(f"{prefix}_NAME")
    user = quote_plus(_need(f"{prefix}_USER"))
    pwd = quote_plus(_need(f"{prefix}_PASSWORD"))

    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{dbname}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


REMOTE = _engine("REMOTE_DB")