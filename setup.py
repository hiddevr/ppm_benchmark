from pathlib import Path
import setuptools

# Build command: python setup.py sdist bdist_wheel

req_path = Path(__file__).parent / "requirements.txt"
with req_path.open() as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setuptools.setup(
    name="ppm_benchmark",
    version="0.0.1",
    author="Hidde van Rooijen",
    author_email="hiddevrooijen@hotmail.nl",
    description="A package that provides utilities for generating reliable benchmark datasets for Predictive Process Monitoring models. Based on the debiasing ideas proposed here: https://github.com/hansweytjens/predictive-process-monitoring-benchmarks",
    long_description=(Path("README.md").read_text(encoding="utf-8")),
    long_description_content_type="text/markdown",
    url="https://github.com/hiddevr/ppm_benchmark",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5, <3.9',
    install_requires=requirements,
)