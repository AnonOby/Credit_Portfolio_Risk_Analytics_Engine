import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Add the project root directory to Python path to import config
# This allows running this file standalone or as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config

# Global engine object
engine = None


def get_engine():
    """
    Initialize and return the SQLAlchemy engine.
    Implements a Singleton pattern to ensure only one engine is created.
    """
    global engine
    if engine is None:
        try:
            # Create engine with pool_pre_ping to check connection health
            engine = create_engine(
                config.DB_URL,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging logs
            )

            # Test the connection immediately
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            print("✅ Database connection established successfully.")

        except SQLAlchemyError as e:
            print(f"❌ Error connecting to database: {e}")
            print("Please check your config.py settings and ensure PostgreSQL is running.")
            return None

    return engine


def get_session():
    """
    Create a new database session for transactional operations.
    """
    db_engine = get_engine()
    if db_engine:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        return SessionLocal()
    else:
        return None


def close_connection():
    """
    Dispose of the engine and close all connections.
    """
    global engine
    if engine:
        engine.dispose()
        print("Database connection closed.")
        engine = None