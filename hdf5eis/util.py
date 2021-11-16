import h5py
import os
import pathlib


def build_master(master, ddir):
    with h5py.File(master, mode="w") as f5m:
        for path in list_files(ddir.joinpath("waveforms")):
            subpath = pathlib.Path(str(path).replace(str(ddir), ""))
            name = str(subpath.parent.joinpath(subpath.stem))
            with h5py.File(path, mode="r") as f5s:
                for key in f5s:
                    handle = f"{name}/{key}"
                    f5m[handle] = h5py.ExternalLink(path, key)


def list_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            yield( pathlib.Path(root).joinpath(file) )
