import h5py
import os
import pathlib

def build_master(master, ddir):
    """
    Traverse the `waveforms` directory of `ddir` and create symbolic
    links to all files in `master`.

    Returns True upon successful completion.
    """
    with h5py.File(master, mode="w") as f5m:
        for path in list_files(ddir.joinpath("waveforms")):
            subpath = pathlib.Path(str(path).replace(str(ddir), ""))
            group = subpath.parent
            with h5py.File(path, mode="r") as f5s:
                for key in f5s:
                    handle = f"{group}/{key}"
                    f5m[handle] = h5py.ExternalLink(path, key)

    return (True)


def list_files(path):
    """
    Generator of paths to all files in `path` and its subdirectories.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            yield (pathlib.Path(root).joinpath(file))
