import setuptools

def configure():
# Initialize the setup kwargs
    kwargs = {
            "name": "hdf5eis",
            "version": "0.0a0",
            "author": "Malcolm White",
            "author_email": "malcolmw@mit.edu",
            "maintainer": "Malcolm White",
            "maintainer_email": "malcolmw@mit.edu",
            "url": "http://malcolmw.github.io/dash",
            "description": "Basic data storage, I/O, processing, and visualization for DAS data using HDF5 backend.",
            "download_url": "https://github.com/malcolmw/HDF5eis.git",
            "platforms": ["linux"],
            "requires": ["numpy", "pandas", "scipy"],
            "packages": ["hdf5eis"]
            }
    return(kwargs)

if __name__ == "__main__":
    kwargs = configure()
    setuptools.setup(**kwargs)
