import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine

# Versuche .env lokal zu laden (wird nicht gepusht)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Versuche Streamlit Secrets zu laden (für Cloud-Deployment)
try:
    import streamlit as st
    _secrets = dict(st.secrets.get("database", {}))
except Exception:
    _secrets = {}


def _need(name: str) -> str:
    # 1. Streamlit Secrets (Cloud), 2. Env Vars (lokal)
    v = _secrets.get(name) or os.getenv(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing env var: {name}")
    return str(v)


def _engine(prefix: str):
    host = _need(f"{prefix}_HOST")
    port = int(_secrets.get(f"{prefix}_PORT") or os.getenv(f"{prefix}_PORT", "3306"))
    dbname = _need(f"{prefix}_NAME")
    user = quote_plus(_need(f"{prefix}_USER"))
    pwd = quote_plus(_need(f"{prefix}_PASSWORD"))

    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{dbname}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


REMOTE = _engine("REMOTE_DB")