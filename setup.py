from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="transcripy",
    version="0.1.0",
    author="Bartosz Niewolski",
    description="a simple library for converting mp3 and mp4 files to text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/transcripy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai-whisper>=20231117",
        "moviepy>=1.0.3",
        "torch>=2.0.0",
    ],
)

