from setuptools import setup, find_packages

setup(
    name="multi-coding-agent",
    version="1.0.0",
    description="Adaptive, Cost-Aware Multi-Agent Code Generation and Self-Debugging on Open-Weight Language Models",
    author="Anonymous Authors (Under Review)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "tabulate>=0.9.0",
        "tqdm>=4.64.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

