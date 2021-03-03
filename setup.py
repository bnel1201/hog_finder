import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hedgiefinder-bnel1201", # Replace with your own username
    version="0.0.1",
    author="Brandon J. Nelson",
    author_email="bnel1201@gmail.com",
    description="A small ML package for finding small hedgehogs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bnel1201/hog_finder",
    project_urls={
        "Bug Tracker": "https://github.com/bnel1201/hog_finder/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
)