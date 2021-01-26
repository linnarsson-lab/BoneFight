from setuptools import find_packages, setup

# rm -r dist   (otherwise twine will upload the oldest build!)
# python setup.py sdist
# twine upload dist/*

__version__ = '0.1.0'

setup(
	name="bonefight",
	version=__version__,
	packages=find_packages(),
	python_requires='>=3.7',
	install_requires=[
		"tqdm"
	],
	# metadata for upload to PyPI
	author="Sten Linnarsson",
	author_email="sten.linnarsson@ki.se",
	description="Bone Fight, an algorithm for aligning omics datasets",
	license="BSD",
	keywords="ostomachion spatial omics transcriptomics bioinformatics",
	url="https://github.com/linnarsson-lab/GeneralTangram",
	download_url=f"https://github.com/linnarsson-lab/BoneFight/archive/{__version__}.tar.gz",
)
