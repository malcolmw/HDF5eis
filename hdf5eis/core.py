'''
Core functionality for HDF5eis provided by the hdf5eis.File class.
'''
# Standard library imports
import pathlib
import re
import sys
import warnings

# Third party imports
import h5py
import numpy as np
import pandas as pd

# Local imports
from ._version import __version__
from . import exceptions
from . import gather as gm

NULL_UTF8_FORMAT  = 'NULL'
NULL_TABLE_FORMAT = 'NULL'
STRING_DTYPE = h5py.string_dtype(encoding='utf-8')
TS_INDEX_COLUMNS = ['tag', 'start_time', 'end_time', 'sampling_rate', 'npts']
TS_INDEX_DTYPES = {
    'tag': STRING_DTYPE,
    'start_time': pd.api.types.DatetimeTZDtype(tz='UTC'),
    'end_time': pd.api.types.DatetimeTZDtype(tz='UTC'),
    'sampling_rate': np.float32,
    'npts': np.int64,
}
COMPATIBLE_VERSIONS = ['0.1.0']

if sys.platform in ('darwin', 'linux'):
    UTF8_DECODER = np.vectorize(lambda x: x.decode('UTF-8'))
elif sys.platform == 'win32':
    UTF8_DECODER = lambda x: x
else:
    raise NotImplementedError


class File(h5py.File):
    '''
    An h5py.File subclass for convenient I/O of big, multidimensional
    timeseries data from environmental sensors. This class provides
    the core functionality for building and processing HDF5eis files.

    '''

    def __init__(self, *args, overwrite=False, validate=True, **kwargs):
        '''
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
        validate : bool, optional
            Whether or not to validate the file structure. This may be
            slow for very large files and can be turned off. The default
            is True.
        **kwargs :
            These are passed directly to super class initializer.

        Raises
        ------

            ValueError if mode='w' and file already exists.

        Returns
        -------
            None.

        '''

        if 'mode' not in kwargs:
            kwargs['mode'] = 'r'

        if (
            kwargs['mode'] == 'w'
            and overwrite is False
            and pathlib.Path(args[0]).exists()
        ):
            raise (
                ValueError(
                    'File already exists! If you are sure you want to open it'
                    ' in write mode, use `overwrite=True`.'
                )
            )

        super().__init__(*args, **kwargs)

        self._metadata = AuxiliaryAccessor(self, '/metadata')
        self._products = AuxiliaryAccessor(self, '/products')
        self._timeseries = TimeseriesAccessor(self, '/timeseries')

        if validate is True:
            self.validate()

    @property
    def metadata(self):
        '''
        Provides functionality to manipulate the  '/metadata'
        group.

        Returns
        -------
        hdf5eis.AuxiliaryAccessor
            Provides functionality to manipulate the  '/metadata'
            group.
        '''
        return self._metadata

    @property
    def products(self):
        '''
        Provides functionality to manipulate the  '/products'
        group.

        Returns
        -------
        hdf5eis.AuxiliaryAccessor
            Provides functionality to manipulate the  '/products'
            group.
        '''
        return self._products

    @property
    def timeseries(self):
        '''
        Provides functionality to manipulate the  '/timeseries'
        group.

        Returns
        -------
        hdf5eis.TimeseriesAccessor
            Provides functionality to manipulate the  '/timeseries'
            group.
        '''
        return self._timeseries

    @property
    def version(self):
        '''
        Return the schema version of this file.

        Returns
        -------
        None.
        '''
        return self.attrs['__VERSION']

    def validate(self):
        '''
        Validate file structure and raise error
        '''
        self._validate_version()
        self._validate_keys()
        self._validate_accessors()

    def _validate_keys(self):
        '''
        Check that the file only contains /metadata, /products, and
        /timeseries keys.

        Raises
        ------
        KeyError
            Raises KeyError if an Group other than /metadata, /products,
            or /timeseries is found in file.

        Returns
        -------
        None.

        '''
        for key in self:
            if key not in ('metadata', 'products', 'timeseries'):
                raise KeyError(f'Invalid Group(={key}) found in file.')

    def _validate_accessors(self):
        '''
        Validate /metadata and /products Groups.

        Returns
        -------
        None.

        '''
        for accessor in (self.metadata, self.products, self.timeseries):
            accessor.validate()

    def _validate_version(self):
        '''
        Check that the version of this file is compatible against
        the current code base.

        Raises
        ------
        VersionError
            Raises VersionError if the file version is incompatible
            with the current code version.

        Returns
        -------
        None.
        '''
        # If __VERSION is not set, try to set it.
        if '__VERSION' not in self.attrs:
            self.attrs['__VERSION'] = __version__
        if self.attrs['__VERSION'] not in COMPATIBLE_VERSIONS:
            version = self.attrs['__VERSION']
            raise exceptions.VersionError(
                f'File version(={version}) is not compatible with code '
                f'version(={__version__}).\n'
                f'Compatible versions: {", ".join(COMPATIBLE_VERSIONS)}.'
            )


class AccessorBase:
    '''
    Abstract base class of Accessor classes.
    '''
    def __init__(self, parent, root):
        '''
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

        '''
        self._parent = parent
        self._root = self._parent.require_group(root)

    @property
    def parent(self):
        '''
        The parent of the group to which this accessor provides access.

        Returns
        -------
        h5py.File or h5py.Group
            The parent of the group to which this accessor provides
            access.

        '''
        return self._parent

    @property
    def root(self):
        '''
        The group to which this accessor provides access.

        Returns
        -------
        h5py.Group
            The group to which this accessor provides access.
        '''
        return self._root


    def add_table(self, dataf, key, fmt=NULL_TABLE_FORMAT):
        '''
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

        '''
        self.root.create_group(key)
        self.write_table(dataf, key, fmt=fmt)


    def list_tables(self):
        '''
        Return a list of all tables below root.

        Returns
        -------
        names : list
            A list of names of tables beneath root.

        '''
        names = list()
        def is_table(name, obj):
            if '__TYPE' in obj.attrs and obj.attrs['__TYPE'] == 'TABLE':
                names.append(name)
        self.root.visititems(is_table)
        return names


    def read_table(self, key):
        '''
        Read data table stored in root group under key.

        Parameters
        ----------
        key : str
            Key of data table in root group to read.

        Returns
        -------
        dataf : pandas.DataFrame
            The table data stored under key in root group.

        '''
        group = self.root[key]
        dataf = pd.DataFrame(index=group['__INDEX'][:])
        for column in filter(lambda key: key != '__INDEX', group.keys()):
            dataf[column] = group[column][:]
            if group[column].attrs['__IS_UTC_DATETIME64'] is np.bool_(True):
                dataf[column] = pd.to_datetime(dataf[column], utc=True)
            elif group[column].attrs['__IS_UTF8'] is np.bool_(True):
                dataf[column] = UTF8_DECODER(dataf[column])
        fmt = group.attrs['__FORMAT']

        return dataf, fmt


    def validate(self):
        self.validate_tables()


    def validate_tables(self):
        for name in self.list_tables():
            _validate_table(self.root[name])


    def write_table(self, dataf, key, fmt=NULL_TABLE_FORMAT):
        '''
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

        '''
        group = self.root[key]
        group.attrs['__TYPE'] = 'TABLE'
        group.attrs['__FORMAT'] = NULL_TABLE_FORMAT
        for column in dataf.columns:
            self.write_column(dataf[column], key)

        if '__INDEX' in self.root[key]:
            del self.root[f'{key}/__INDEX']

        self.root[key].create_dataset('__INDEX', data=dataf.index.values)


    def write_column(self, column, key):
        '''
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

        '''
        is_utc_datetime64 = pd.api.types.is_datetime64_any_dtype(column)
        if (
            (
                hasattr(column.dtype, 'char')
                and
                column.dtype.char == np.dtype('S')
            )
            or
            column.dtype == np.dtype('O')
        ):
            is_utf8 = True
        else:
            is_utf8 = False

        if is_utc_datetime64:
            if column.dt.tz is None:
                warnings.warn(f'Time zone of \'{column}\' is not set. Assuming UTC.')
                column = column.dt.tz_localize('UTC')
            column = column.dt.tz_convert('UTC')
            column = column.astype(np.int64)

        if f'{key}/{column.name}' in self.root:
            del self.root[f'{key}/{column.name}']

        values = column.values
        datas = self.root[key].create_dataset(
            column.name,
            data=values.astype(STRING_DTYPE) if is_utf8 else values
        )
        datas.attrs['__IS_UTC_DATETIME64'] = is_utc_datetime64
        datas.attrs['__IS_UTF8'] = is_utf8


class HDF5eisFileFormatError(Exception):
    '''
    An Exception indicating that the current file is improperly
    formatted.

    '''


class AuxiliaryAccessor(AccessorBase):
    '''
    Accessor class for auxiliary (i.e., /metadata and /products groups)
    data.
    '''
    def __getitem__(self, key):
        '''
        Read the item under `key` from this group.
        '''

        dtype = self.root[key].attrs['__TYPE']

        if dtype == 'TABLE':
            return self.read_table(key)
        if dtype == 'UTF-8':
            return UTF8_DECODER(self.root[key][0]), self.root[key].attrs['__FORMAT']
        raise HDF5eisFileFormatError(f'Unknown data type {dtype} for key {key}.')

    def add(self, obj, key, fmt=None):
        '''
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

        '''
        if isinstance(obj, pd.DataFrame):
            fmt = fmt if fmt is not None else NULL_TABLE_FORMAT
            self.add_table(obj, key, fmt=fmt)
        elif isinstance(obj, (str, bytes)):
            fmt = fmt if fmt is not None else NULL_UTF8_FORMAT
            self.add_utf8(obj, key, fmt=fmt)

    def add_utf8(self, data, key, fmt=NULL_UTF8_FORMAT):
        '''
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

        '''
        self.root.create_dataset(
            key, data=[data.encode('utf-8')], dtype=STRING_DTYPE
        )
        self.root[key].attrs['__TYPE'] = 'UTF-8'
        self.root[key].attrs['__FORMAT'] = fmt

    def link(self, src_file, src_path, key):
        '''
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

        '''
        self.root[key] = h5py.ExternalLink(src_file, src_path)


    def list_utf8(self):
        '''
        Return a list of all UTF-8 encoded strings below root.

        Returns
        -------
        names : list
            A list of names of UTF-8 encoded strings beneath root.

        '''
        names = list()
        def is_utf8(name, obj):
            if '__TYPE' in obj.attrs and obj.attrs['__TYPE'] == 'UTF-8':
                names.append(name)
        self.root.visititems(is_utf8)
        return names

    def validate(self):
        self.validate_tables()
        self.validate_utf8()

    def validate_utf8(self):
        for name in self.list_utf8():
            dtype = self.root[name].dtype
            if dtype != STRING_DTYPE:
                raise(exceptions.UTF8FormatError(
                    f'UTF-8 encoded string has the wrong dtype(={dtype}).'
                ))
            if not '__FORMAT' in self.root[name].attrs:
                raise(exceptions.UTF8FormatError(
                    'UTF-8 encoded string has no __FORMAT specified.'
                ))




class TimeseriesAccessor(AccessorBase):
    '''
    Accessor class for timeseries data.

    '''
    @property
    def index(self):
        '''
        Tabular index of contents in /timeseries group.

        Returns
        -------
        pandas.DataFrame
            Index of contents in /timeseries group.

        '''
        if not hasattr(self, '_index'):
            if '__TS_INDEX' in self.root:
                self._index, fmt = self.read_table('__TS_INDEX')
            else:
                self._index = pd.DataFrame(columns=TS_INDEX_COLUMNS)
                self._index = self._index.astype(TS_INDEX_DTYPES)
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    def add(self, data, start_time, sampling_rate, tag='', **kwargs):
        '''
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
            Tag to associate with data. The default is ''.
        **kwargs :
            Additional keyword arguments are passed directly the
            h5py.Group.create_datset() method and can be used, for
            example, to choose the chunk layout and compression options.

        Returns
        -------
        None.

        '''
        if 'dtype' not in kwargs:
            kwargs['dtype'] = data.dtype

        datas = self.create_dataset(
            data.shape,
            start_time,
            sampling_rate,
            tag=tag,
            **kwargs,
        )

        datas[:] = data


    def create_dataset(
            self, shape, start_time, sampling_rate, tag='', **kwargs
    ):
        '''
        Create and return an empty data set.

        Parameters
        ----------
        shape : tuple
            The shape of the empty data set.
        start_time : str, int, float, or pandas.Timestamp
            The UTC time of the first sample in data. This value is
            internally converted to a pandas.Timestamp by
            pandas.to_datetime().
        sampling_rate : int, float
            The temporal sampling rate of data in units of samples per
            second.
        tag : str, optional
            Tag to associate with data. The default is ''.
        **kwargs : TYPE
            Additional keyword arguments are passed directly the
            h5py.Group.create_datset() method and can be used, for
            example, to choose the chunk layout and compression options.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        sampling_interval = pd.to_timedelta(1 / sampling_rate, unit='S')
        nsamples = shape[-1]
        start_time = pd.to_datetime(start_time)
        end_time = start_time + sampling_interval * (nsamples - 1)
        handle = build_handle(tag, start_time, end_time)
        datas = self.root.create_dataset(handle, shape=shape, **kwargs)
        datas.attrs['sampling_rate'] = sampling_rate

        row = pd.DataFrame(
            [[tag, start_time, end_time, sampling_rate, shape[-1]]],
            columns=TS_INDEX_COLUMNS,
        )
        self.index = pd.concat([self.index, row], ignore_index=True)
        self.flush_index()

        return self.root[handle]

    def flush_index(self):
        '''
        Flush the self.index attribute to disk.

        Returns
        -------
        None.

        '''
        if '__TS_INDEX' not in self.root:
            self.add_table(
                self.index.astype(TS_INDEX_DTYPES),
                '__TS_INDEX',
                fmt='TIMESERIES_INDEX'
                )
        else:
            self.write_table(
                self.index.astype(TS_INDEX_DTYPES),
                '__TS_INDEX',
                fmt='TIMESERIES_INDEX'
            )

    def link_tag(
        self,
        src_file,
        src_tag,
        prefix=None,
        suffix=None,
        new_tag=None
    ):
        '''
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

        '''
        assert prefix is None or isinstance(prefix, str)
        assert suffix is None or isinstance(suffix, str)
        assert new_tag is None or isinstance(new_tag, str)

        if new_tag is None:
            new_tag = '' if prefix is None else prefix
            new_tag = '/'.join((new_tag, src_tag))
            new_tag = new_tag if suffix is None else '/'.join((new_tag, suffix))
            new_tag = new_tag.lstrip('/')

        new_index = list()

        with h5py.File(src_file, mode='r') as file:
            accessor = TimeseriesAccessor(file, '/timeseries')
            index = accessor.index
            index = index[index['tag'].str.match(src_tag)]

            for _, row in index.iterrows():
                src_handle = '/'.join((
                    '/timeseries',
                    build_handle(
                        row['tag'],
                        row['start_time'],
                        row['end_time']
                    ),
                ))
                new_handle = '/'.join(
                    (new_tag, src_handle.rsplit('/', maxsplit=1)[-1])
                )
                self.root[new_handle] = h5py.ExternalLink(src_file, src_handle)
                row['tag'] = '/'.join(
                    (new_tag, row['tag'].lstrip(src_tag))
                ).strip('/')
                new_index.append(row)

        self.index = pd.concat(
            [self.index, pd.DataFrame(new_index)],
            ignore_index=True
        )
        self.flush_index()

    def __getitem__(self, key):
        '''
        Return a list of hdf5eis.gather.Gather objects.

        The first element of `key` must be a `str` tag. The last
        element must be a time slice.

        Assumes data are regularly sampled between requested start and end times.
        '''

        assert isinstance(key[0], str)
        assert key[-1].start is not None and key[-1].stop is not None

        # key is a tuple
        tag = key[0]
        start_time = pd.to_datetime(key[-1].start, utc=True)
        end_time = pd.to_datetime(key[-1].stop, utc=True)

        index = self.reduce_index(tag, start_time, end_time)

        # Read data for each tag.
        gathers = dict()

        for tag in index['tag'].unique():
            gathers[tag] = list()

            gathers[tag] = self.read_data(
                index[index['tag'] == tag],
                start_time,
                end_time,
                key
            )

        return gathers


    def read_data(self, index, start_time, end_time, key):
        '''
        Read data between start and end time with respect to the given
        index and key.

        Parameters
        ----------
        index : pandas.DataFrame
            A subset of self.index corresponding to the requested data.
            This DataFrame is the return value of self.reduce_index().
            Every continuous block of data associated with a single tag
            needs to have a unique segment ID. The index of this
            DataFrame must be the segment ID.
        start_time : pandas.Timestamp
            The time of the earliest sample requested.
        end_time : pandas.Timestamp
            The time of the latest sample requested.
        key : tuple
            A tuple of slice indexes with which to slice the data along
            each axis except the last (time) axis.

        Returns
        -------
        gathers : list
            A list of hdf5eis.Gather objects containing requested data.

        '''
        gathers = list()

        for segment_id in index.index.unique():

            gathers.append(
                self.read_segment(
                    index.loc[[segment_id]],
                    start_time,
                    end_time,
                    key
                )
            )

        return gathers

    def read_segment(self, rows, start_time, end_time, key):
        '''
        Read the continuous segment of data corresponding to the given
        set of index rows and between given start and end times.

        Parameters
        ----------
        rows : pandas.DataFrame
            A set of index rows for a continuous segment of data. These
            should be the rows corresponding to a single segment ID as
            determined by self.reduce_index().
        start_time : pandas.Timestamp
            Time of the first requested sample.
        end_time : pandas.Timestamp
            Time of the last requested sample.
        key : tuple
            Slice index with which to slice the data being read. These
            slices will be applied along every storage axis except the
            last (time) axis.

        Returns
        -------
        hdf5eis.Gather
            An hdf5eis.Gather object containing the requested continuous
            segment of data.

        '''
        # Make sure the sampling rate doesn't change mid stream.
        assert len(rows['sampling_rate'].unique()) == 1

        nsamples, first_sample = determine_segment_sample_range(
            rows,
            start_time,
            end_time
        )
        sampling_rate = rows.iloc[0]['sampling_rate']

        data, jstart = None, 0
        for _, row in rows.iterrows():
            sample_range = (
                _sample_idx(
                    start_time,
                    row['start_time'],
                    row['sampling_rate']
                ),
                (
                    _sample_idx(
                        min(end_time, row['end_time']),
                        row['start_time'],
                        row['sampling_rate'],
                        right=True
                    )
                    + 1
                )
            )
            handle_key = (
                build_handle(
                    row['tag'],
                    row['start_time'],
                    row['end_time']
                ),
                (*key[1:-1], slice(*sample_range))
            )

            assert (
                len(handle_key[1]) == self.root[handle_key[0]].ndim
                or
                Ellipsis in handle_key[1]
            )

            if data is None:
                shape = (
                    *get_shape(
                        self.root[handle_key[0]].shape[:-1],
                        handle_key[1][:-1]
                    ),
                    nsamples,
                )
                data = np.empty(shape, dtype=self.root[handle_key[0]].dtype)
            data[
                ...,
                jstart: jstart+sample_range[1]-sample_range[0]
            ] = self.root[handle_key[0]][handle_key[1]]
            jstart += sample_range[1]-sample_range[0]

        return gm.Gather(
            data,
            first_sample,
            sampling_rate
        )


    def reduce_index(self, tag, start_time, end_time):
        '''
        Reduce the index to the set of rows referring to data between
        the given start and endtimes and matching the given tag.

        Parameters
        ----------
        tag : str
            Tag of requested data. Regular expressions are valid here.
        start_time : pandas.Timestamp
            The start time of the requested data.
        end_time : pandas.Timestamp
            The end time of the requested data.

        Returns
        -------
        index : pandas.DataFrame
            The set of rows from self.index matching the given tag and
            time range.

        '''
        index = self.index
        if index is None:
            warnings.warn('The timeseries index is empty.')
            return None

        # Find matching tags.
        index = index[index['tag'].str.fullmatch(tag)]

        # Find datasets within requested time range.
        index = index[
             (index['start_time'] < end_time)
            &(index['end_time'] > start_time)
        ]

        if len(index) == 0:
            warnings.warn('No data found for specified tag and time range.')
            return None

        # Sort values by time.
        index = index.sort_values('start_time')

        sampling_interval = pd.to_timedelta(
            1 / index['sampling_rate'],
            unit='S'
        )
        delta = index['start_time'] - index.shift(1)['end_time']

        index['segment_id'] = (delta != sampling_interval).cumsum()
        index = index.set_index('segment_id')

        return index

    def validate(self):
        self.validate_tables()
        self._validate_ts_index()

    def _validate_ts_index(self):
        '''
        Validate the __TS_INDEX table.

        Raises
        ------
        exceptions.TSIndexError
            Raises TSIndexError if their is a mismatch between
            the __TS_INDEX table and timeseries data.

        Returns
        -------
        None.

        '''
        for _, row in self.index.iterrows():
            handle = build_handle(row['tag'], row['start_time'], row['end_time'])
            if handle not in self.root:
                raise exceptions.TSIndexError(
                    f'Expected DataSet at {handle} not found.'
                )
            elif self.root[handle].shape[-1] != row['npts']:
                raise exceptions.TSIndexError(
                    f'Unexpected number of samples in DataSet at {handle}.'
                )


def determine_segment_sample_range(rows, start_time, end_time):
    '''
    Determine the number of samples and time of the first sample
    for data available in given rows between given start and times.

    Parameters
    ----------
    rows : pandas.DataFrame
        Index rows corresponding to a single continuous data segment.
        This should be a subset of index rows returned by
        hdf5eis.TimeseriesAccessor.reduce_rows() with a single, unique
        segment ID.
    start_time : pandas.Timestamp
        The time of the first requested sample.
    end_time : pandas.Timestamp
        The time of the last requested sample.

    Returns
    -------
    nsamples : int
        The number of available samples in the specified time range.
    first_sample : pandas.Timestamp
        The time of the first available sample in the specified time
        range.


    |-----|-----|-----|-----|-----|-----|
             A                 B
    |-----X-----X-----X-----X-----X-----|

    Samples X are returned for start_time=A, end_time=B.
    '''
    sampling_rate = rows.iloc[0]['sampling_rate']
    data_start_time = rows.iloc[0]['start_time']
    data_end_time = rows.iloc[-1]['end_time']
    istart = _sample_idx(start_time, data_start_time, sampling_rate)
    iend = (
        _sample_idx(
            min(end_time, data_end_time),
            data_start_time,
            sampling_rate,
            right=True
        )
        + 1
    )
    nsamples = iend - istart
    offset = pd.to_timedelta(istart / sampling_rate, unit='S')
    first_sample = data_start_time + offset

    return (nsamples, first_sample)


def get_shape(shape, key):
    '''
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

    '''
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
    '''
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

    '''
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
    '''
    Build a properly formatted address to the data referred to by
    tag, start_time, and end_time.

    Parameters
    ----------
    tag : str
        The tag associated with the data.
    start_time : pandas.Timestamp
        The time of the first sample in the array for which the handle
        is being built.
    end_time : pandas.Timestamp
        The time of the last sample in the array for which the handle
        is being built.

    Returns
    -------
    handle : str
        The address of the data relative to the accessor root.

    '''
    tstart = strftime(start_time)
    tend = strftime(end_time)
    handle = '/'.join((tag, f'__{tstart}__{tend}'))
    handle = re.sub('//+', '/', handle)
    handle = handle.lstrip('/')

    return handle


def _sample_idx(time, start_time, sampling_rate, right=False):
    '''
    Get the index of a sample at a given time, relative to start_time.

    Parameters
    ----------
    time : pandas.Timestamp
        Time of sample to index.
    start_time : pandas.Timestamp
        Time of first sample in the array being indexed.
    sampling_rate : int, float
        Sampling rate (in units of samples per second) of array being
        indexed.
    right : bool, optional
        Return the index of the sample to the immediate right of the
        provided time if it falls in between samples. The default is
        False.

    Returns
    -------
    int
        The index of the sample corresponding to the provided time.

    '''
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
    '''
    Return list of column names with datetime-like dtype.

    Parameters
    ----------
    dataf : pandas.DataFrame
        DataFrame from which to extract names of columns with
        datetime-like dtype.

    Returns
    -------
    list
        List of column names with datetime-like dtype.

    '''
    is_datetime = pd.api.types.is_datetime64_any_dtype
    return list(filter(lambda key: is_datetime(dataf[key]), dataf.columns))


def _validate_table(group):
    '''
    Validate table structure.

    Parameters
    ----------
    group : h5py.Group
        The parent h5py.Group of the table to validate.

    Raises
    ------
    exceptions.TableFormatError
        Raises TableFormatError if any format errors are detected.

    Returns
    -------
    None.

    '''
    # Validate column shapes.
    lengths = list()
    columns = list(group.keys())
    for column in columns:
        shape = group[column].shape
        if len(shape) != 1:
            raise exceptions.TableFormatError(
                f'Columns must have shape=(N,) but shape={shape} found for '
                f'\'{column}\' column in \'{group.name}\' table.'
            )
        lengths.append(shape[-1])
    if len(np.unique(lengths)) != 1:
        msg = f'Columns in \'{group.name}\' table are not all the same length.'
        msg += '\n\nColumn: Length\n--------------'
        for i in range(len(columns)):
            msg += f'\n{columns[i]}: {lengths[i]}'
        raise exceptions.TableFormatError(msg)

    # Check for existence and compatibility of __IS_UTC_DATETIME64 and
    # __IS_UTF8 tags.
    for column in columns:
        # Don't check the __INDEX column because it has different requirements.
        if column == '__INDEX':
            continue
        if '__IS_UTC_DATETIME64' not in group[column].attrs:
            raise(exceptions.TableFormatError(
                f'__IS_UTC_DATETIME64 Attribute missing for \'{column}\' '
                f'column of \'{group.name}\' table.'
            ))
        if '__IS_UTF8' not in group[column].attrs:
            raise(exceptions.TableFormatError(
                f'__IS_UTF8 Attribute missing for \'{column}\' '
                f'column of \'{group.name}\' table.'
            ))
        if (
            group[column].attrs['__IS_UTC_DATETIME64']
            and
            group[column].attrs['__IS_UTF8']
        ):
            raise(exceptions.TableFormatError(
                f'__IS_UTC_DATETIME64 and __IS_UTF8 Attributes are both True '
                f'for \'{column}\' column of \'{group.name}\' table.'
            ))

    # Verify that __IS_UTC_DATETIME64 columns are 64-bit integers.
    for column in columns:
        # Don't check the __INDEX column because it has different requirements.
        if column == '__INDEX':
            continue
        if group[column].attrs['__IS_UTC_DATETIME64']:
            dtype = group[column].dtype
            if dtype != np.int64:
                raise(exceptions.TableFormatError(
                    f'__IS_UTC_DATETIME64 is True for \'{column}\' column '
                    f'of \'{group.name}\' table, but dtype(={dtype}) is not '
                    f'a 64-bit integer.'
                ))
        elif group[column].attrs['__IS_UTF8']:
            dtype = group[column].dtype
            if not dtype == STRING_DTYPE:
                raise(exceptions.TableFormatError(
                    f'__IS_UTF8 is True for \'{column}\' column of '
                    f'\'{group.name}\' table, but dtype(={dtype}) does not '
                    f'appear to be a byte string.'
                ))


def strftime(time):
    '''
    Return a formatted string representation of `time`.

    Parameters
    ----------
    time : pandas.Timestamp
        Time to convert to string representation.

    Returns
    -------
    str
        String representation of time formatted like
        '%Y%m%dT%H:%M:%S.%nZ'.

    '''
    return ''.join((
        f'{time.year:4d}',
        f'{time.month:02d}',
        f'{time.day:02d}',
        f'T{time.hour:02d}',
        f':{time.minute:02d}',
        f':{time.second:02d}',
        f'.{time.microsecond:06d}',
        f'{time.nanosecond:03d}'
    ))
