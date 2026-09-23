import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR.parent / ".env"

load_dotenv(ENV_FILE)

CSV_FILE = BASE_DIR / "data" / "processed" / "products_clean.csv"


def get_database_url():
    """Build the PostgreSQL connection URL from environment variables."""

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if not all([user, password, database]):
        raise ValueError(
            "Missing PostgreSQL environment variables. "
            "Check the .env file."
        )

    return (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


def load_data_to_postgres():
    """Load the cleaned CSV into PostgreSQL."""

    print("Reading cleaned CSV...")

    df = pd.read_csv(CSV_FILE)

    print(f"Records to load: {len(df)}")

    database_url = get_database_url()

    engine = create_engine(database_url)

    print("Connecting to PostgreSQL...")

    df.to_sql(
        "products",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("Data successfully loaded into PostgreSQL.")
    print("Table created: products")


def main():
    print("Starting database loading...")

    load_data_to_postgres()

    print("Database loading completed successfully.")


if __name__ == "__main__":
    main()