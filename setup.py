from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def _fetch_requirements(path):
    with open(path, encoding="utf-8") as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


setup(
    author="Mingyu Cao, Yaokai Jin, Shiwei Liu, Lu Yin",
    name="bernoulli-opd",
    version=VERSION,
    description="Reference implementation of Bernoulli On-Policy Distillation",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/duterscmy/Bernoulli-OPD",
    license="Apache-2.0",
    packages=find_packages(include=["slime*", "slime_plugins*"]),
    include_package_data=True,
    install_requires=_fetch_requirements(ROOT / "requirements.txt"),
    extras_require={},
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
