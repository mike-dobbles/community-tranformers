import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="community-transformers",
    version="0.0.5",
    author="Mike Dobbles",
    author_email="mdobbles@yahoo.com",
    description="A package of custom pyspark.ml transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mike-dobbles/community-tranformers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)