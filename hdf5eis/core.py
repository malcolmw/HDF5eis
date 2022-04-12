import h5py
import numpy as np
import pandas as pd
import pathlib
import re
import warnings

# Local module import
import gather

warnings.simplefilter(
    action="ignore",
    category=pd.errors.PerformanceWarning
)

WF_INDEX_COLUMNS = [
    "tag",
    "starttime",
    "endtime",
    "sampling_rate",
    "shape"
]


class File(h5py.File,object):
    def __init__(self, *args, overwrite=False, **kwargs):
        """
        An h5py.File subclass for convenient I/O of big, multidimensional
        data from environmental sensors.
        """

        if "mode" not in kwargs:
            kwargs["mode"] = "r"

        self._open_mode = kwargs["mode"]

        self._init_args = args
        self._init_kwargs = kwargs.copy()
        self._init_kwargs["mode"] = "r" if self._init_kwargs["mode"] == "r" else "r+"

        if (
            kwargs["mode"] == "w"
            and overwrite is False
            and pathlib.Path(args[0]).exists()
        ):
            raise (
                ValueError(
                    "File already exists! If you are sure you want to open it"
                    " in write mode, use `overwrite=True`."
                )
            )

        super(File, self).__init__(*args, **kwargs)




    @property
    def metadata(self):
        if not hasattr(self, "_metadata"):
            self._metadata = AuxiliaryAccessor(self, "/metadata")

        return (self._metadata)

    @property
    def products(self):
        if not hasattr(self, "_products"):
            self._products = AuxiliaryAccessor(self, "/products")

        return (self._products)

    @property
    def waveforms(self):
        if not hasattr(self, "_waveforms"):
            self._waveforms = WaveformAccessor(self, "/waveforms")

        return (self._waveforms)


    def list_datasets(self, group=None):
        """
        Return a list of all data sets in group. Skip special "_index" group.
        """
        if group is None:
            group = self
        elif isinstance(group, str):
            group = self[group]

        groups = list()
        for key in group:
            if isinstance(group[key], h5py.Group):
                if key == "index":
                    continue
                else:
                    groups += self.list_datasets(group=group[key])
            else:
                if group.name.split("/")[1] in ("products", "metadata"):
                    if group.name not in groups:
                        groups.append(group.name)
                else:
                    groups.append("/".join((group.name, key)))

        return (groups)


    def reopen(self):
        """
        Reopen the HDF5 file.

        Opens in "r+" mode if object was instantiated with write permissions.
        Opens in "r" mode otherwise.
        """

        super(File, self).__init__(*self._init_args, **self._init_kwargs)


class AccessorBase(object):
    def __init__(self, parent, root):
        self._parent = parent
        self._root = root


    @property
    def parent(self):
        return (self._parent)


    @property
    def root(self):
        return (self._parent.require_group(self._root))


    def add(self, dataf, key):
        """
        Add `dataf` to the appropriate group of the open file under `key`.
        """
        key = "/".join((self.root.name, key))
        filename = self.parent.file.filename
        self.parent.close()
        try:
            dataf.to_hdf(filename, key=key)
        finally:
            self.parent.reopen()


    def link(self, src_file, src_path, key):
        self.root[key] = h5py.ExternalLink(src_file, src_path)


class AuxiliaryAccessor(AccessorBase, object):
    def __getitem__(self, key):
        """
        Read the item under `key` from this group.
        """

        obj = self.root[key]
        dtype = obj.attrs["type"]

        if dtype == "TABLE":
            return (self._read_table(obj))
        elif dtype == "UTF-8":
            return (obj[0].decode())
        else:
            raise (HDF5eisFileFormatError(f"Unknown data type {dtype} for key {key}."))


    def add(self, obj, key):
        """
        Add `dataf` to the appropriate group of the open file under `key`.
        """
        if isinstance(obj, pd.DataFrame):
            self._add_table(obj, key)
        elif isinstance(obj, str):
            self._add_utf8(obj, key)

    def _add_table(self, dataf, key):
        filename = self.parent.file.filename
        pkey = "/".join((self.root.name, key)) # Key in parent
        self.parent.close()
        try:
            dataf.to_hdf(filename, key=pkey)
        finally:
            self.parent.reopen()
        self.root[key].attrs["type"] = "TABLE"


    def _add_utf8(self, data, key):
        self.root.create_dataset(
            key,
            data=[data],
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        self.root[key].attrs["type"] = "UTF-8"


    def _read_table(self, group):
        filename = group.file.filename
        key = group.name

        if filename == self.parent.file.filename:
            self.parent.close()
            try:
                dataf = pd.read_hdf(filename, key)
            except h5py.HDF5ExtError as err:
                raise (err)
            finally:
                self.parent.reopen()
        else:
            group.file.close()
            dataf = pd.read_hdf(filename, key)

        return (dataf)


    def link(self, src_file, src_path, key):
        self.root[key] = h5py.ExternalLink(src_file, src_path)


class HDF5eisFileFormatError(Exception):
    pass


class WaveformAccessor(AccessorBase, object):
    @property
    def index(self):
        if not hasattr(self, "_index"):
            filename = self.parent.file.filename
            self.parent.close()
            try:
                # Try to read from disk first.
                index =  pd.read_hdf(filename, key="/waveforms/index")
            except KeyError:
                # Initialize with empty DataFrame if nothing found on disk.
                index = pd.DataFrame()
            finally:
                self.parent.reopen()

            self._index = index

        return (self._index)


    @index.setter
    def index(self, value):
        self._index = value


    def strftime(self, time):
        """
        Return a formatted string representation of `time`.
        """
        return (time.strftime("%Y%m%dT%H:%M:%S.%fZ"))


    def handle(self, tag, starttime, endtime):
        """
        Returns a properly formatted reference to data specified by
        `tag`, `starttime`, and `endtime`.
        """
        ts = self.strftime(starttime)
        te = self.strftime(endtime)
        handle = "/".join((tag, f"__{ts}__{te}"))
        handle = re.sub("//+", "/", handle)

        return (handle)


    def add(self, data, starttime, sampling_rate, tag="", flush_index=True, **kwargs):
        """
        Add waveforms to the parent HDF5 file.
        """
        if "dtype" not in kwargs:
            kwargs["dtype"] = data.dtype

        datas = self.create_dataset(
            data.shape,
            starttime,
            sampling_rate,
            tag=tag,
            flush_index=flush_index,
            **kwargs
        )

        datas[:] = data

        return (True)


    def create_dataset(self, shape, starttime, sampling_rate, tag="", flush_index=True, **kwargs):
        """
        Returns an empty dataset.
        """
        sampling_interval = pd.to_timedelta(1 / sampling_rate, unit="S")
        nsamples = shape[-1]
        starttime = pd.to_datetime(starttime)
        endtime = starttime + sampling_interval * (nsamples - 1)
        handle = self.handle(tag, starttime, endtime)
        datas = self.root.create_dataset(
            handle,
            shape=shape,
            **kwargs
        )
        datas.attrs["sampling_rate"] = sampling_rate

        row = pd.DataFrame(
            [[tag, starttime, endtime, sampling_rate, shape]],
            columns=WF_INDEX_COLUMNS,
        )
        self.index = pd.concat([self.index, row], ignore_index=True)
        if flush_index is True:
            self.flush_index()

        return (self.root[handle])


    def flush_index(self, reopen=True):
        filename = self.parent.file.filename
        self.parent.close()
        self.index.to_hdf(filename, key="/waveforms/index")
        self.parent.reopen()


    def link(self, src_file, tag=None):
        with File(src_file, mode="r") as f5:
            keys = f5.list_datasets(group="/waveforms")
            index = f5.waveforms.index

        self.root[tag] = h5py.ExternalLink(src_file, "/waveforms")

        sep = "" if tag is None else "/"
        index["tag"] = tag + sep + index["tag"]

        self.index = pd.concat([self.index, index], ignore_index=True)


    def read(self, tag, starttime, endtime):
        """
        Return a hdf5eis.core.Gather object with data between *starttime*
        and *endtime*.

        Positional Arguments
        ====================
        starttime: int, float, str, datetime
            Start time of desired time window.
        endtime: int, float, str, datetime
            End time of desired time window.

        Keyword Arguments
        =================
        traces: list, slice
            Indices of desired traces.

        Returns
        =======
        gather: hdf5eis.core.Gather
            Gather object containing desired data.
        """

        starttime = pd.to_datetime(starttime, utc=True)
        endtime   = pd.to_datetime(endtime, utc=True)

        index = self.index

        if index is None:
            print("No waveforms found.")
            return (None)

        index = index[index["tag"].str.fullmatch(tag)]

        index = index[
             (index["starttime"] < endtime)
            &(index["endtime"] > starttime)
        ]
        index = index.sort_values("starttime")

        if len(index) == 0:
            raise (ValueError("No data found for specified tag and time range."))

        sampling_interval = pd.to_timedelta(1 / index["sampling_rate"], unit="S")
        delta = index["starttime"] - index.shift(1)["endtime"]

        index["segment_id"] = (delta != sampling_interval).cumsum()
        index = index.set_index("segment_id")

        gathers = list()

        for segment_id in index.index.unique():
            rows = index.loc[[segment_id]]
            # There should be a check here to make sure the shape of each
            # chunk is compatible.

            # Make sure the sampling rate doesn't change mid stream.
            assert len(rows["sampling_rate"].unique()) == 1

            first_row      = rows.iloc[0]
            last_row       = rows.iloc[-1]
            sampling_rate  = first_row["sampling_rate"]
            data_starttime = first_row["starttime"]
            data_endtime   = last_row["endtime"]
            istart         = _sample_idx(starttime, data_starttime, sampling_rate)
            iend           = _sample_idx(
                min(endtime, data_endtime),
                data_starttime,
                sampling_rate
            ) + 1
            nsamples       = iend - istart
            offset         = pd.to_timedelta(istart / sampling_rate, unit="S")
            first_sample   = data_starttime + offset
            shape          = (*first_row["shape"][:-1], nsamples)
            data           = np.empty(shape)
            jstart         = 0
            for _, row in rows.iterrows():
                data_starttime = row["starttime"]
                data_endtime   = row["endtime"]
                tag            = row["tag"]
                handle = self.handle(tag, data_starttime, data_endtime)
                istart = _sample_idx(starttime, data_starttime, sampling_rate)
                iend   = _sample_idx(min(endtime, data_endtime), data_starttime, sampling_rate) + 1
                jend   = jstart + (iend - istart)
                data[..., jstart: jend] = self.root[handle][..., istart: iend]
                jstart = jend

            starttime = data_starttime + pd.to_timedelta(istart / sampling_rate, unit="S")
            trace_idxs = np.arange(data.shape[0])
            gathers.append(
                gather.Gather(data, first_sample, sampling_rate, trace_idxs)
            )

        if len(gathers) == 1:
            return (gathers[0])
        else:
            return (gathers)


#def _iter_group(group):
#    """
#    Return a list of all data sets in group. Skip special "_index" group.
#    """
#    groups = list()
#    for key in group:
#        if isinstance(group[key], h5py.Group):
#            if key[0] == "_":
#                continue
#            else:
#                groups += _iter_group(group[key])
#        else:
#            groups.append("/".join((group.name, key)))
#
#    return (groups)
#
#
#def _list_files(path):
#    """
#    Generator of paths to all files in `path` and its subdirectories.
#    """
#    for root, dirs, files in os.walk(path):
#        for file in files:
#            yield (pathlib.Path(root).joinpath(file))
#
#
#def _repr_group(group, indent=""):
#    s = ""
#    for key in group:
#        s += f"{indent}+--{key}\n"
#        if isinstance(group[key], h5py.Group):
#            s += _repr_group(group[key], indent=indent+"|--")
#    return (s)
#
#
def _sample_idx(time, starttime, sampling_rate, right=False):
    """
    Get the index of a sample at a given time, relative to starttime.
    """
    time = pd.to_datetime(time, utc=True)
    delta = (time - starttime).total_seconds()

    if delta < 0:

        return (0)

    else:
        if right is True:
            idx = int(np.ceil(delta * sampling_rate))
        else:
            idx = int(np.floor(delta * sampling_rate))

    return (idx)
