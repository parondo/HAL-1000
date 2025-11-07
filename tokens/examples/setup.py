from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hall1000",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Geometric Consciousness AGI Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/HALL-1000-Geometric-AGI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0", 
            "flake8>=5.0.0",
            "pre-commit>=2.20.0",
        ],
        "robotics": [
            "roboticstoolbox-python>=1.0.0",
            "spatialmath-python>=1.0.0",
        ],
        "simulation": [
            "gym>=0.26.0",
            "mujoco>=2.3.0",
            "pybullet>=3.2.0",
        ]
    },
)
