import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def make_engine(prefix: str):
    host = require_env(f"{prefix}_HOST")
    port = int(os.getenv(f"{prefix}_PORT", "3306"))
    db = require_env(f"{prefix}_NAME")
    user = quote_plus(require_env(f"{prefix}_USER"))
    pwd = quote_plus(require_env(f"{prefix}_PASSWORD"))

    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def smoke_test(engine, label: str):
    """
    Prüft:
      - Verbindung möglich?
      - SELECT 1
      - aktuelle DB
      - Server-Version
      - Anzahl Tabellen (nur Metadaten)
    """
    try:
        with engine.connect() as conn:
            ok = conn.execute(text("SELECT 1")).scalar()
            current_db = conn.execute(text("SELECT DATABASE()")).scalar()
            version = conn.execute(text("SELECT VERSION()")).scalar()
            table_count = conn.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE()")
            ).scalar()

        print(f"[OK] {label}")
        print(f"     SELECT 1 -> {ok}")
        print(f"     DATABASE() -> {current_db}")
        print(f"     VERSION()  -> {version}")
        print(f"     Tables     -> {table_count}")

    except SQLAlchemyError as e:
        print(f"[FAIL] {label}")
        # Keine Secrets ausgeben: nur Fehlertyp + Message
        print(f"       {type(e).__name__}: {e}")


def main():
    load_dotenv()  # lädt .env (lokal, nicht committen!)

    local = make_engine("LOCAL_DB")
    remote = make_engine("REMOTE_DB")

    smoke_test(local, "LOCAL_DB")
    smoke_test(remote, "REMOTE_DB")


if __name__ == "__main__":
    main()