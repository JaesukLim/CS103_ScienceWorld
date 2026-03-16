import sys
import re
import os.path
import zipfile

from setuptools import setup


BASEPATH = os.path.dirname(os.path.abspath(__file__))
JAR_FILE = 'scienceworld.jar'
JAR_PATH = os.path.join(BASEPATH, 'cs103_scienceworld', JAR_FILE)
# Extract ScienceWorld version from JAR file metadata
contents = zipfile.ZipFile(JAR_PATH).open('META-INF/MANIFEST.MF').read().decode('utf-8')
VERSION = re.search(r'\bSpecification-Version: (.*)\b', contents).group(1)

OBJECTS_LUT_FILE = "object_type_ids.tsv"
TASKS_JSON_FILE = "tasks.json"

if not os.path.isfile(JAR_PATH):
    print('ERROR: Unable to find required library:', JAR_PATH)
    sys.exit(1)


setup(
    name='cs103-scienceworld',
    version="0.1.3",
    description='This is forked version of ScienceWorld for KAIST CS103',
    author='flight0454',
    packages=['cs103_scienceworld'],
    include_package_data=True,
    package_dir={'cs103_scienceworld': 'cs103_scienceworld'},
    package_data={'cs103_scienceworld': [JAR_FILE, OBJECTS_LUT_FILE, TASKS_JSON_FILE]},
    url="https://github.com/JaesukLim/CS103_ScienceWorld",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    python_requires='>=3.7',
    install_requires=open('requirements.txt').readlines(),
    extras_require={
        'webserver': open('requirements.txt').readlines() + ['pywebio'],
        'dev': open('requirements-dev.txt').readlines(),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ]
)
