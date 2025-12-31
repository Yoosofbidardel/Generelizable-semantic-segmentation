"""
Setup configuration for Generelizable Semantic Segmentation package.
This file defines the package installation and distribution settings.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Core dependencies required for the package to function
install_requires = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "numpy>=1.19.0",
    "opencv-python>=4.5.0",
    "pillow>=8.0.0",
    "scipy>=1.5.0",
    "matplotlib>=3.3.0",
    "scikit-image>=0.17.0",
    "tqdm>=4.50.0",
]

# Development dependencies
extras_require = {
    "dev": [
        "pytest>=6.0",
        "pytest-cov>=2.10",
        "black>=21.0",
        "flake8>=3.8",
        "isort>=5.0",
    ],
    "docs": [
        "sphinx>=3.0",
        "sphinx-rtd-theme>=0.5",
    ],
}

setup(
    name="generelizable-semantic-segmentation",
    version="1.0.0",
    description="A generelizable semantic segmentation framework for computer vision applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yoosof Bidardel",
    author_email="",
    url="https://github.com/Yoosofbidardel/Generelizable-semantic-segmentation",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="semantic segmentation deep learning computer vision generalization",
    project_urls={
        "Source": "https://github.com/Yoosofbidardel/Generelizable-semantic-segmentation",
        "Bug Tracker": "https://github.com/Yoosofbidardel/Generelizable-semantic-segmentation/issues",
    },
    include_package_data=True,
    zip_safe=False,
)
