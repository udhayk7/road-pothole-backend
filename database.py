"""
Database configuration module for the Predictive Road Infrastructure System.
Handles SQLAlchemy engine setup, session management, and base model configuration.
Supports PostgreSQL via DATABASE_URL environment variable.
"""

import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    logging.error("DATABASE_URL environment variable is not set.")
    logging.error("Set DATABASE_URL in Railway → Variables.")
    raise RuntimeError("DATABASE_URL is required in production environment.")

# Create SQLAlchemy engine for PostgreSQL
engine = create_engine(DATABASE_URL)

# SessionLocal class will be used to create database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()


def get_db():
    """
    Dependency function that provides a database session.
    Ensures proper cleanup of database connections after each request.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
