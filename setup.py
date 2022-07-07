import setuptools

def configure():
# Initialize the setup kwargs
    kwargs = {
            "name": "hdf5eis",
            "version": "0.1a0",
            "author": "Malcolm White",
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
