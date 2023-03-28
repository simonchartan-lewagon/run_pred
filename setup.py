from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='run_pred',
      description="Le Wagon Project - Batch 1167 - Data Science - Run Prediction",
      install_requires=requirements,
      packages=find_packages()
      )
