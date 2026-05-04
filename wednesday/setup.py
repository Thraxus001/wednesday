# /setup.py
"""
Setup script for Wednesday AI.
For modern Python packaging, see pyproject.toml
"""

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read dependencies from requirements.txt (optional)
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="wednesday-ai",
    version="0.1.0",
    author="Wednesday AI Team",
    author_email="team@wednesday.ai",
    description="A human-like cognitive AI with Wednesday Addams' personality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wednesday",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
            "ruff>=0.1",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "hypothesis>=6.0",
        ],
        "nlp": [
            "transformers>=4.30",
            "spacy>=3.5",
            "nltk>=3.8",
        ],
        "audio": [
            "speechrecognition>=3.10",
            "librosa>=0.10",
            "pydub>=0.25",
        ],
        "vision": [
            "opencv-python>=4.8",
            "pillow>=10.0",
            "face-recognition>=1.3",
        ],
        "ml": [
            "torch>=2.0",
            "scikit-learn>=1.3",
            "xgboost>=2.0",
        ],
        "database": [
            "sqlalchemy>=2.0",
            "redis>=5.0",
            "faiss-cpu>=1.7",
            "neo4j>=5.0",
        ],
        "api": [
            "fastapi>=0.100",
            "uvicorn>=0.23",
            "websockets>=12.0",
        ],
        "all": [
            "transformers>=4.30",
            "spacy>=3.5",
            "nltk>=3.8",
            "speechrecognition>=3.10",
            "librosa>=0.10",
            "pydub>=0.25",
            "opencv-python>=4.8",
            "pillow>=10.0",
            "face-recognition>=1.3",
            "torch>=2.0",
            "scikit-learn>=1.3",
            "xgboost>=2.0",
            "sqlalchemy>=2.0",
            "redis>=5.0",
            "faiss-cpu>=1.7",
            "neo4j>=5.0",
            "fastapi>=0.100",
            "uvicorn>=0.23",
            "websockets>=12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wednesday=wednesday.__init__:main_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "wednesday": ["py.typed"],
    },
    zip_safe=False,
)