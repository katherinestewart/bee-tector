from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="bee_tector",
    version="0.1.0",
    description="BeeTector ML model and API",
    license="MIT",
    author="Katherine Stewart",
    author_email="katherine.stewart1@gmail.com",
    install_requires=requirements,
    packages=find_packages(),
    test_suite="tests",
    include_package_data=True,
)
