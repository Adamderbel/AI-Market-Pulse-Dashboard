"""Setup script for Market Dashboard."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="market-dashboard",
    version="1.0.0",
    description="AI-powered interactive market dashboard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Market Dashboard Team",
    author_email="",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ]
    },
    entry_points={
        "console_scripts": [
            "market-dashboard=market_dashboard.app:run_app",
            "fetch-data=scripts.fetch_data:main",
            "load-data=scripts.load_to_db:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="finance, dashboard, ai, market-analysis, visualization",
    project_urls={
        "Documentation": "",
        "Source": "",
        "Tracker": "",
    },
)
