"""
Core functionality for HDF5eis provided by the hdf5eis.File class.
"""
# Standard library imports
import pathlib
import re
import warnings

# Third party imports
import h5py
import numpy as np
import pandas as pd

# Local imports
from . import gather


TS_INDEX_COLUMNS = ["tag", "starttime", "endtime", "sampling_rate", "npts"]
TS_INDEX_DTYPES = {
    "tag": np.dtype("S"),
    "starttime": pd.api.types.DatetimeTZDtype(tz="UTC"),
    "endtime": pd.api.types.DatetimeTZDtype(tz="UTC"),
    "sampling_rate": np.float32,
    "npts": np.int64,
}


class File(h5py.File):
    def __init__(self, *args, overwrite=False, **kwargs):
        """
        An h5py.File subclass for convenient I/O of big, multidimensional
        data from environmental sensors.

        Parameters
        ----------
        *args :
            These are passed directly to the super class initializer.
            Must contain a file path (str or bytes) or a file-like
            object.
        overwrite : bool, optional
            Whether or not to overwrite an existing file if one exists
            at the given location. The default is False.
        **kwargs :
            These are passed directly to super class initializer.

        Raises
        ------

            ValueError if mode="w" and file already exists.

        Returns
        -------
        None.

        """

        if "mode" not in kwargs:
            kwargs["mode"] = "r"

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

        super().__init__(*args, **kwargs)

        self._metadata = AuxiliaryAccessor(self, "/metadata")
        self._products = AuxiliaryAccessor(self, "/products")
        self._timeseries = WaveformAccessor(self, "/timeseries")

    @property
    def metadata(self):
        """
        Provides access to "/metadata" group and associated
        functionality.

        Returns
        -------
        hdf5eis.AuxiliaryAccessor
            Provides access to "/metadata" group and associated
            functionality.

        """
        return self._metadata

    @property
    def products(self):
        """
        Provides access to "/products" group and associated
        functionality.

        Returns
        -------
        hdf5eis.AuxiliaryAccessor
            Provides access to "/products" group and associated
            functionality.

        """
        return self._products

    @property
    def timeseries(self):
        """
        Provides access to "/times" group and associated
        functionality.

        Returns
        -------
        hdf5eis.AuxiliaryAccessor
            Provides access to "/metadata" group and associated
            functionality.

        """
        return self._timeseries


class AccessorBase:
    """
    Abstract base class of Accessor classes.
    """
    def __init__(self, parent, root):
        """
        Initializer.

        Parameters
        ----------
        parent : h5py.File or h5py.Group
            The parent of the group this accessor provides access to.
        root : str
            The name of group nested within parent this accessor
            provides access to.

        Returns
        -------
        None.

        """
        self._parent = parent
        self._root = self._parent.require_group(root)

    @property
    def parent(self):
        """
        The parent of the group to which this accessor provides access.

        Returns
        -------
        h5py.File or h5py.Group
            The parent of the group to which this accessor provides
            access.

        """
        return self._parent

    @property
    def root(self):
        """
        The group to which this accessor provides access.

        Returns
        -------
        h5py.Group
            The group to which this accessor provides access.
        """
        return self._root


    def add_table(self, dataf, key):
        """
        Add DataFrame `dataf` to root group under `key`.

        Parameters
        ----------
        dataf : pandas.DataFrame
            DataFrame to add to root group.
        key : str
            Key value under which to add table.

        Returns
        -------
        None.

        """
        group = self.root.create_group(key)
        group.attrs["type"] = "TABLE"

        self.write_table(dataf, key)


    def write_table(self, dataf, key):
        for column in dataf.columns:
            self.write_column(dataf[column], key)

        if "__INDEX" in self.root[key]:
            del self.root[f"{key}/__INDEX"]

        self.root[key].create_dataset("__INDEX", data=dataf.index.values)


    def write_column(self, column, key):
        """
        Write a single column of data to the self.root Group of
        self.parent.

        Parameters
        ----------
        column : pandas.Series
            Column of data to write.
        key : str
            Key value under which to store data.

        Returns
        -------
        None.

        """
        is_utc_datetime64 = pd.api.types.is_datetime64_any_dtype(column)
        is_utf8 = hasattr(column.dtype, "char") and column.dtype.char == np.dtype("S")

        if is_utc_datetime64:
            if column.dt.tz is None:
                warnings.warn(f"Time zone of '{column}' is not set. Assuming UTC.")
                column = column.dt.tz_localize("UTC")
            column = column.dt.tz_convert("UTC")
            column = column.astype(np.int64)

        if f"{key}/{column.name}" in self.root:
            del self.root[f"{key}/{column.name}"]

        datas = self.root[key].create_dataset(column.name, data=column.values)
        datas.attrs["__IS_UTC_DATETIME64"] = is_utc_datetime64
        datas.attrs["__IS_UTF8"] = is_utf8


    def read_table(self, key):
        group = self.root[key]
        dataf = pd.DataFrame(index=group["__INDEX"][:])
        for column in filter(lambda key: key != "__INDEX", group.keys()):
            dataf[column] = group[column][:]
            if group[column].attrs["__IS_UTC_DATETIME64"] is np.bool_(True):
                dataf[column] = pd.to_datetime(dataf[column], utc=True)
            elif group[column].attrs["__IS_UTF8"] is np.bool_(True):
                dataf[column] = dataf[column].str.decode("UTF-8")

        return dataf


class HDF5eisFileFormatError(Exception):
    pass


class AuxiliaryAccessor(AccessorBase):
    def __getitem__(self, key):
        """
        Read the item under `key` from this group.
        """

        dtype = self.root[key].attrs["type"]

        if dtype == "TABLE":
            return self.read_table(key)
        if dtype == "UTF-8":
            return self.root[key][0].decode()
        raise HDF5eisFileFormatError(f"Unknown data type {dtype} for key {key}.")

    def add(self, obj, key):
        """
        Add `dataf` to the appropriate group of the open file under `key`.
        """
        if isinstance(obj, pd.DataFrame):
            self.add_table(obj, key)
        elif isinstance(obj, str):
            self._add_utf8(obj, key)

    def _add_utf8(self, data, key):
        self.root.create_dataset(
            key, data=[data], dtype=h5py.string_dtype(encoding="utf-8")
        )
        self.root[key].attrs["type"] = "UTF-8"

    def link(self, src_file, src_path, key):
        self.root[key] = h5py.ExternalLink(src_file, src_path)


class WaveformAccessor(AccessorBase):
    @property
    def index(self):
        if not hasattr(self, "_index"):
            if "__TS_INDEX" in self.root:
                self._index = self.read_table("__TS_INDEX")
            else:
                self._index = pd.DataFrame(columns=TS_INDEX_COLUMNS)
                self._index = self._index.astype(TS_INDEX_DTYPES)
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    def add(self, data, starttime, sampling_rate, tag="", flush_index=True, **kwargs):
        """
        Add timeseries to the parent HDF5 file.
        """
        if "dtype" not in kwargs:
            kwargs["dtype"] = data.dtype

        datas = self.create_dataset(
            data.shape,
            starttime,
            sampling_rate,
            tag=tag,
            flush_index=flush_index,
            **kwargs,
        )

        datas[:] = data

        return True

    def create_dataset(
        self, shape, starttime, sampling_rate, tag="", flush_index=True, **kwargs
    ):
        """
        Returns an empty dataset.
        """
        sampling_interval = pd.to_timedelta(1 / sampling_rate, unit="S")
        nsamples = shape[-1]
        starttime = pd.to_datetime(starttime)
        endtime = starttime + sampling_interval * (nsamples - 1)
        handle = build_handle(tag, starttime, endtime)
        datas = self.root.create_dataset(handle, shape=shape, **kwargs)
        datas.attrs["sampling_rate"] = sampling_rate

        row = pd.DataFrame(
            [[tag, starttime, endtime, sampling_rate, shape[-1]]],
            columns=TS_INDEX_COLUMNS,
        )
        self.index = pd.concat([self.index, row], ignore_index=True)

        if flush_index is True:
            self.flush_index()

        return self.root[handle]

    def flush_index(self):
        if "__TS_INDEX" not in self.root:
            self.add_table(self.index.astype(TS_INDEX_DTYPES), "__TS_INDEX")
        else:
            self.write_table(self.index.astype(TS_INDEX_DTYPES), "__TS_INDEX")

    def link_tag(
        self,
        src_file,
        src_tag,
        prefix=None,
        suffix=None,
        new_tag=None,
        flush_index=True
    ):
        """
        Links timeseries data from an external file to the current file.

        Parameters
        ----------
        src_file : str or pathlib.Path
            Path to external file to be linked.
        src_tag : str
            Tag in external file to be linked.
        prefix : str, optional
            Prefix for new tag. The default is None.
        suffix : str, optional
            Suffix for new tag. The default is None.
        new_tag : str, optional
            New tag. The default is None.
        flush_index : bool, optional
            Flush the index to disk. This should be set to True unless
            you know what you are doing. The default is True.

        Returns
        -------
        None.

        """
        assert prefix is None or isinstance(prefix, str)
        assert suffix is None or isinstance(suffix, str)
        assert new_tag is None or isinstance(new_tag, str)

        if new_tag is None:
            new_tag = "" if prefix is None else prefix
            new_tag = "/".join((new_tag, src_tag))
            new_tag = new_tag if suffix is None else "/".join((new_tag, suffix))
            new_tag = new_tag.lstrip("/")

        new_index = list()

        with File(src_file, mode="r") as file:
            index = file.timeseries.index
            index = index[index["tag"].str.match(src_tag)]

            for _, row in index.iterrows():
                src_handle = "/".join(
                    (
                        "/timeseries",
                        build_handle(row["tag"], row["starttime"], row["endtime"]),
                    )
                )
                new_handle = "/".join((new_tag, src_handle.rsplit("/", maxsplit=1)[-1]))
                self.root[new_handle] = h5py.ExternalLink(src_file, src_handle)
                row["tag"] = "/".join((new_tag, row["tag"].lstrip(src_tag))).strip("/")
                new_index.append(row)

        self.index = pd.concat([self.index, pd.DataFrame(new_index)], ignore_index=True)

        if flush_index is True:
            self.flush_index()

    def __getitem__(self, key):
        """
        Return a list of hdf5eis.gather.Gather objects.

        The first element of `key` must be a `str` tag. The last
        element must be a time slice.

        Assumes data are regularly sampled between requested start and end times.
        """

        assert isinstance(key[0], str)
        assert key[-1].start is not None and key[-1].stop is not None

        # key is a tuple
        tag = key[0]
        starttime = pd.to_datetime(key[-1].start, utc=True)
        endtime = pd.to_datetime(key[-1].stop, utc=True)

        index = self.index
        if index is None:
            warnings.warn("The timeseries index is empty.")
            return None

        # Find matching tags.
        index = index[index["tag"].str.fullmatch(tag)]

        # Find datasets within requested time range.
        index = index[(index["starttime"] < endtime) & (index["endtime"] > starttime)]

        if len(index) == 0:
            warnings.warn("No data found for specified tag and time range.")
            return None

        # Read data for each tag.
        gathers = dict()

        for tag in index["tag"].unique():
            gathers[tag] = list()
            _index = index[index["tag"] == tag]

            # Sort values by time.
            _index = _index.sort_values("starttime")

            sampling_interval = pd.to_timedelta(1 / _index["sampling_rate"], unit="S")
            delta = _index["starttime"] - _index.shift(1)["endtime"]

            _index["segment_id"] = (delta != sampling_interval).cumsum()
            _index = _index.set_index("segment_id")

            for segment_id in _index.index.unique():
                rows = _index.loc[[segment_id]]

                # Make sure the sampling rate doesn't change mid stream.
                assert len(rows["sampling_rate"].unique()) == 1

                first_row = rows.iloc[0]
                last_row = rows.iloc[-1]
                sampling_rate = first_row["sampling_rate"]
                data_starttime = first_row["starttime"]
                data_endtime = last_row["endtime"]
                istart = _sample_idx(starttime, data_starttime, sampling_rate)
                iend = (
                    _sample_idx(
                        min(endtime, data_endtime), data_starttime, sampling_rate
                    )
                    + 1
                )
                nsamples = iend - istart
                offset = pd.to_timedelta(istart / sampling_rate, unit="S")
                first_sample = data_starttime + offset

                data, jstart = None, 0
                for _, row in rows.iterrows():
                    data_starttime = row["starttime"]
                    data_endtime = row["endtime"]
                    handle = build_handle(tag, data_starttime, data_endtime)
                    istart = _sample_idx(starttime, data_starttime, sampling_rate)
                    iend = (
                        _sample_idx(
                            min(endtime, data_endtime), data_starttime, sampling_rate
                        )
                        + 1
                    )
                    jend = jstart + (iend - istart)
                    _key = (*key[1:-1], slice(istart, iend))

                    assert len(_key) == self.root[handle].ndim or Ellipsis in _key

                    if data is None:
                        shape = (
                            *get_shape(self.root[handle].shape[:-1], _key[:-1]),
                            nsamples,
                        )
                        data = np.empty(shape, dtype=self.root[handle].dtype)
                    data[..., jstart:jend] = self.root[handle][_key]
                    jstart = jend

                starttime = data_starttime + pd.to_timedelta(
                    istart / sampling_rate, unit="S"
                )
                trace_idxs = np.arange(data.shape[0])
                _gather = gather.Gather(data, first_sample, sampling_rate, trace_idxs)
                gathers[tag].append(_gather)

        return gathers


def get_shape(shape, key):
    new_shape = tuple()
    imax = len(shape) if Ellipsis not in key else key.index(Ellipsis)
    new_shape = tuple((get_slice_length(key[i], shape[i]) for i in range(imax)))
    if imax < len(shape):
        new_shape += shape[imax : imax + len(shape) - len(key) + 1]
        new_shape += tuple((
            get_slice_length(key[i], shape[i])
            for i in range(len(key) - 1, imax, -1)
        ))
    return tuple(filter(lambda k: k != 0, new_shape))


def get_slice_length(obj, max_len):
    if obj == slice(None):
        return max_len
    if isinstance(obj, slice):
        istart = 0 if obj.start is None else obj.start
        iend = max_len if obj.stop is None else obj.stop
        return iend - istart
    if isinstance(obj, int):
        return 0
    raise ValueError


def build_handle(tag, starttime, endtime):
    """
    Returns a properly formatted reference to data specified by
    `tag`, `starttime`, and `endtime`.
    """
    tstart = strftime(starttime)
    tend = strftime(endtime)
    handle = "/".join((tag, f"__{tstart}__{tend}"))
    handle = re.sub("//+", "/", handle)
    handle = handle.lstrip("/")

    return handle


def _sample_idx(time, starttime, sampling_rate, right=False):
    """
    Get the index of a sample at a given time, relative to starttime.
    """
    time = pd.to_datetime(time, utc=True)
    delta = (time - starttime).total_seconds()

    if delta < 0:
        return 0
    if right is True:
        idx = int(np.ceil(delta * sampling_rate))
    else:
        idx = int(np.floor(delta * sampling_rate))
    return idx


def _get_time_fields(dataf):
    is_datetime = pd.api.types.is_datetime64_any_dtype
    return list(filter(lambda key: is_datetime(dataf[key]), dataf.columns))


def strftime(time):
    """
    Return a formatted string representation of `time`.
    """
    return time.strftime("%Y%m%dT%H:%M:%S.%fZ")
