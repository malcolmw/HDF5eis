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


TS_INDEX_COLUMNS = ["tag", "start_time", "end_time", "sampling_rate", "npts"]
TS_INDEX_DTYPES = {
    "tag": np.dtype("S"),
    "start_time": pd.api.types.DatetimeTZDtype(tz="UTC"),
    "end_time": pd.api.types.DatetimeTZDtype(tz="UTC"),
    "sampling_rate": np.float32,
    "npts": np.int64,
}


class File(h5py.File):
    """
    An h5py.File subclass for convenient I/O of big, multidimensional
    timeseries data from environmental sensors. This class provides
    the core functionality for manipulating HDF5eis files.

    """

    def __init__(self, *args, overwrite=False, **kwargs):
        """
        Initialize hdf5eis.File object.

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
        self._timeseries = TimeseriesAccessor(self, "/timeseries")

    @property
    def metadata(self):
        """
        Provides functionality to manipulate the  "/metadata"
        group.

        Returns
        -------
        hdf5eis.AuxiliaryAccessor
            Provides functionality to manipulate the  "/metadata"
            group.
        """
        return self._metadata

    @property
    def products(self):
        """
        Provides functionality to manipulate the  "/products"
        group.

        Returns
        -------
        hdf5eis.AuxiliaryAccessor
            Provides functionality to manipulate the  "/products"
            group.
        """
        return self._products

    @property
    def timeseries(self):
        """
        Provides functionality to manipulate the  "/timeseries"
        group.

        Returns
        -------
        hdf5eis.TimeseriesAccessor
            Provides functionality to manipulate the  "/timeseries"
            group.
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
        """
        Write data table to the parent group under key.

        Parameters
        ----------
        dataf : pandas.DataFrame
            DataFrame to write to disk.
        key : str
            Key under which to write data.

        Returns
        -------
        None.

        """
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
        """
        Read data table stored in root group under key.

        Parameters
        ----------
        key : str
            Key of data table in root group to read.

        Returns
        -------
        dataf : pandas.DataFrame
            The table data stored under key in root group.

        """
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
    """
    An Exception indicating that the current file is improperly
    formatted.

    """


class AuxiliaryAccessor(AccessorBase):
    """
    Accessor class for auxiliary (i.e., /metadata and /products groups)
    data.
    """
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
        Add a table or UTF-8 encoded byte stream to the root group.

        Parameters
        ----------
        obj : pandas.DataFrame, str, or bytes
            DataFrame, string or stream of UTF-8 encoded bytes to add
            to root group under key.
        key : str
            Key under which to add the data object.

        Returns
        -------
        None.

        """
        if isinstance(obj, pd.DataFrame):
            self.add_table(obj, key)
        elif isinstance(obj, (str, bytes)):
            self.add_utf8(obj, key)

    def add_utf8(self, data, key):
        """
        Add UTF-8 encoded data to root group under key.

        Parameters
        ----------
        data : str or bytes
            Data to add to root group under key.
        key : str
            Key under which to add the data object.

        Returns
        -------
        None.

        """
        self.root.create_dataset(
            key, data=[data], dtype=h5py.string_dtype(encoding="utf-8")
        )
        self.root[key].attrs["type"] = "UTF-8"

    def link(self, src_file, src_path, key):
        """
        Create and external link a data table or UTF-8 encoded byte
        stream in the root group under key.

        Parameters
        ----------
        src_file : path-like
            Path to source file containing data to be externally
            linked to root group.
        src_path : str
            Path within source file to dataset or group to be linked.
        key : str
            Key within the root group under which to link external
            data.

        Returns
        -------
        None.

        """
        self.root[key] = h5py.ExternalLink(src_file, src_path)


class TimeseriesAccessor(AccessorBase):
    """
    Accessor class for timeseries data.

    """
    @property
    def index(self):
        """
        Tabular index of contents in /timeseries group.

        Returns
        -------
        pandas.DataFrame
            Index of contents in /timeseries group.

        """
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

    def add(self, data, start_time, sampling_rate, tag="", **kwargs):
        """
        Add timeseries data to the parent HDF5eis file.

        Parameters
        ----------
        data : array-like
            Data array of any shape to add to file.
        start_time : str, int, float, or pandas.Timestamp
            The UTC time of the first sample in data. This value is
            internally converted to a pandas.Timestamp by
            pandas.to_datetime().
        sampling_rate : int, float
            The temporal sampling rate of data in units of samples per
            second.
        tag : str, optional
            Tag to associate with data. The default is "".
        **kwargs :
            Additional keyword arguments are passed directly the
            h5py.Group.create_datset() method and can be used, for
            example, to choose the chunk layout and compression options.

        Returns
        -------
        None.

        """
        if "dtype" not in kwargs:
            kwargs["dtype"] = data.dtype

        datas = self.create_dataset(
            data.shape,
            start_time,
            sampling_rate,
            tag=tag,
            **kwargs,
        )

        datas[:] = data


    def create_dataset(
            self, shape, start_time, sampling_rate, tag="", **kwargs
    ):
        """
        Returns an empty dataset.
        """
        sampling_interval = pd.to_timedelta(1 / sampling_rate, unit="S")
        nsamples = shape[-1]
        start_time = pd.to_datetime(start_time)
        end_time = start_time + sampling_interval * (nsamples - 1)
        handle = build_handle(tag, start_time, end_time)
        datas = self.root.create_dataset(handle, shape=shape, **kwargs)
        datas.attrs["sampling_rate"] = sampling_rate

        row = pd.DataFrame(
            [[tag, start_time, end_time, sampling_rate, shape[-1]]],
            columns=TS_INDEX_COLUMNS,
        )
        self.index = pd.concat([self.index, row], ignore_index=True)
        self.flush_index()

        return self.root[handle]

    def flush_index(self):
        """
        Flush the self.index attribute to disk.

        Returns
        -------
        None.

        """
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
        new_tag=None
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
                        build_handle(row["tag"], row["start_time"], row["end_time"]),
                    )
                )
                new_handle = "/".join((new_tag, src_handle.rsplit("/", maxsplit=1)[-1]))
                self.root[new_handle] = h5py.ExternalLink(src_file, src_handle)
                row["tag"] = "/".join((new_tag, row["tag"].lstrip(src_tag))).strip("/")
                new_index.append(row)

        self.index = pd.concat([self.index, pd.DataFrame(new_index)], ignore_index=True)
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
        start_time = pd.to_datetime(key[-1].start, utc=True)
        end_time = pd.to_datetime(key[-1].stop, utc=True)

        index = self.index
        if index is None:
            warnings.warn("The timeseries index is empty.")
            return None

        # Find matching tags.
        index = index[index["tag"].str.fullmatch(tag)]

        # Find datasets within requested time range.
        index = index[(index["start_time"] < end_time) & (index["end_time"] > start_time)]

        if len(index) == 0:
            warnings.warn("No data found for specified tag and time range.")
            return None

        # Read data for each tag.
        gathers = dict()

        for tag in index["tag"].unique():
            gathers[tag] = list()
            _index = index[index["tag"] == tag]

            # Sort values by time.
            _index = _index.sort_values("start_time")

            sampling_interval = pd.to_timedelta(1 / _index["sampling_rate"], unit="S")
            delta = _index["start_time"] - _index.shift(1)["end_time"]

            _index["segment_id"] = (delta != sampling_interval).cumsum()
            _index = _index.set_index("segment_id")

            for segment_id in _index.index.unique():
                rows = _index.loc[[segment_id]]

                # Make sure the sampling rate doesn't change mid stream.
                assert len(rows["sampling_rate"].unique()) == 1

                first_row = rows.iloc[0]
                last_row = rows.iloc[-1]
                sampling_rate = first_row["sampling_rate"]
                data_start_time = first_row["start_time"]
                data_end_time = last_row["end_time"]
                istart = _sample_idx(start_time, data_start_time, sampling_rate)
                iend = (
                    _sample_idx(
                        min(end_time, data_end_time), data_start_time, sampling_rate
                    )
                    + 1
                )
                nsamples = iend - istart
                offset = pd.to_timedelta(istart / sampling_rate, unit="S")
                first_sample = data_start_time + offset

                data, jstart = None, 0
                for _, row in rows.iterrows():
                    data_start_time = row["start_time"]
                    data_end_time = row["end_time"]
                    handle = build_handle(tag, data_start_time, data_end_time)
                    istart = _sample_idx(start_time, data_start_time, sampling_rate)
                    iend = (
                        _sample_idx(
                            min(end_time, data_end_time), data_start_time, sampling_rate
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

                start_time = data_start_time + pd.to_timedelta(
                    istart / sampling_rate, unit="S"
                )
                trace_idxs = np.arange(data.shape[0])
                _gather = gather.Gather(data, first_sample, sampling_rate, trace_idxs)
                gathers[tag].append(_gather)

        return gathers


def get_shape(shape, key):
    """
    Determine the shape of a sliced array.

    Parameters
    ----------
    shape : tuple
        The shape of the array before being sliced.
    key : tuple
        The slice indices.

    Returns
    -------
    tuple
        The shape of the sliced array.

    """
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
    """
    Determine the length of a slice along a single axis.

    Parameters
    ----------
    obj : slice, int
        The slice index.
    max_len : int
        The maximum possible length of the slice. The length of the
        axis.

    Raises
    ------
    ValueError
        Raises ValueError if the slice index is not a slice or int object.

    Returns
    -------
    int
        The length of the slice.

    """
    if obj == slice(None):
        return max_len
    if isinstance(obj, slice):
        istart = 0 if obj.start is None else obj.start
        iend = max_len if obj.stop is None else obj.stop
        return iend - istart
    if isinstance(obj, int):
        return 0
    raise ValueError


def build_handle(tag, start_time, end_time):
    """
    Build a properly formatted address to the data referred to by
    tag, start_time, and end_time.

    Parameters
    ----------
    tag : TYPE
        DESCRIPTION.
    start_time : TYPE
        DESCRIPTION.
    end_time : TYPE
        DESCRIPTION.

    Returns
    -------
    handle : TYPE
        DESCRIPTION.

    """
    tstart = strftime(start_time)
    tend = strftime(end_time)
    handle = "/".join((tag, f"__{tstart}__{tend}"))
    handle = re.sub("//+", "/", handle)
    handle = handle.lstrip("/")

    return handle


def _sample_idx(time, start_time, sampling_rate, right=False):
    """
    Get the index of a sample at a given time, relative to start_time.
    """
    time = pd.to_datetime(time, utc=True)
    delta = (time - start_time).total_seconds()

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
