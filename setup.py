from setuptools import setup, find_packages
import os
from glob import glob
from os.path import basename
from os.path import splitext
import re

def read(fname: str):
    with open(fname) as f:
        dump = [ s.strip() for s in f.readlines() if not s.isspace() ]

    # convert git dependencies in requirements.txt
    # from pip-compatible to setuptools-compatible
    for (i,line) in enumerate(dump):
        if line.startswith("git+"):
            m = re.search(
                string=line, pattern=r"#egg=(?P<package_name>(\d|[a-zA-Z]|\-)+)"
            )
            package_name = m.groupdict()["package_name"]
            ## https://stackoverlow.com/a/54794506
            dump[i] = f"{package_name} @ {line}"
    return dump

setup(
    name="adme_tools",
    version="0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "run_adme = adme_tools.run_adme:main",
        ]
    },
    author="Yongbin Kim",
    author_email="chem.yongbin@gmail.com",
    description="Wrapper for running computational ADME prediction",
    # install_requires=read("requirements.txt"),
)
