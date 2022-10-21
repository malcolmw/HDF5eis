import re
import setuptools

version_file = 'hdf5eis/_version.py'
version_line = open(version_file, 'r').read()
version_re = r'^__version__ = ["\']([^"\']*)["\']'
mo = re.search(version_re, version_line, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError(f'Unable to find version string in {version_file}.')

def configure():
# Initialize the setup kwargs
    kwargs = {
            "name": "HDF5eis",
            "version": version,
            "author": "Malcolm C. A. White",
            "author_email": "malcolmw@mit.edu",
            "maintainer": "Malcolm C. A. White",
            "maintainer_email": "malcolmw@mit.edu",
            "url": "http://malcolmw.github.io/HDF5eis",
            "description": """A solution for storing and accessing big,
            multidimensional timeseries data from environmental sensors""",
            "download_url": "https://github.com/malcolmw/HDF5eis.git",
            "platforms": ["linux"],
            "install_requires": ["h5py", "matplotlib", "numpy", "pandas", "scipy"],
            "packages": ["hdf5eis"]
            }
    return kwargs

if __name__ == "__main__":
    kwargs = configure()
    setuptools.setup(**kwargs)
