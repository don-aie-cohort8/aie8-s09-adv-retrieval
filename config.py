"""
Configuration file for the AIE8 Advanced Retrieval project.
This file contains path configurations and other settings.
"""

from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data file paths
PDF_FILES = DATA_DIR / "*.pdf"
CSV_FILES = DATA_DIR / "*.csv"

# Model configurations
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Vector store configurations
QDRANT_COLLECTION_NAME = "Use Case RAG"
QDRANT_LOCATION = ":memory:"

# Print configuration for verification
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Data Directory Exists: {DATA_DIR.exists()}")
    print(f"PDF Files Pattern: {PDF_FILES}")
