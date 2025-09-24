from setuptools import setup, find_packages

setup(
    name="fire-simulator",
    version="0.1.0",
    description="Cellular-automata wildfire simulator with GeoTIFF support and visualization.",
    author="Anastasios (Tasos) Tsolakis",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24,<3",
        "matplotlib>=3.7,<4",
        "rasterio>=1.3.9,<2",
    ],
    entry_points={
        "console_scripts": [
            "fire-sim = fire_simulator.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fire_simulator": ["data/*.tif", "data/*.tiff"],
    },
    python_requires=">=3.10",
)

