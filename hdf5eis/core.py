import h5py
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import re
import scipy.signal
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class File(h5py.File):

    def __init__(self, *args, dtype=np.float32, safe_mode=True, **kwargs):
        """
        An h5py.File subclass for convenient I/O of big seismic data.
        """
        
        if "mode" not in kwargs:
            kwargs["mode"] = "r"
        self._open_mode = kwargs["mode"]
        
        self._init_args = args
        self._init_kwargs = kwargs
        self._dtype = dtype

        if (
            kwargs["mode"] == "w"
            and safe_mode is True
            and pathlib.Path(args[0]).exists()
        ):
            raise (
                ValueError(
                    "File already exists! If you are sure you want to open it"
                    " in write mode, use `safe_mode=False`."
                )
            )

        super(File, self).__init__(*args, **kwargs)


    def __str__(self):
        return (_repr_group(self))


    @property
    def dtype(self):
        return (self._dtype)
    
    @property
    def index(self):
        if not hasattr(self, "_index"):
            self._index = pd.read_hdf(self.filename, key="/waveforms/_index")

        return (self._index)
    
    def _set_mode(self, mode):
        self.close()
        
        self._init_kwargs["mode"] = mode
        super(File, self).__init__(*self._init_args, **self._init_kwargs)
        
        
    def _reset_mode(self):
        self.close()
        
        if self._open_mode in ("w", "w+"):
            self._init_kwargs["mode"] = "a"
        else:
            self._init_kwargs["mode"] = self._open_mode

        super(File, self).__init__(*self._init_args, **self._init_kwargs)
        
    
    
    def add_waveforms(self, data, starttime, sampling_rate, tag="", **kwargs):
        sampling_interval = pd.to_timedelta(1 / sampling_rate, unit="S")
        nsamples = data.shape[-1]
        starttime = pd.to_datetime(starttime)
        endtime = starttime + sampling_interval * (nsamples - 1)
        ts = starttime.strftime("%Y%m%dT%H:%M:%S.%fZ")
        te = endtime.strftime("%Y%m%dT%H:%M:%S.%fZ")
        handle = "/".join(("waveforms", tag, f"__{ts}__{te}"))
        handle = re.sub("//+", "/", handle)
        datas = self.create_dataset(
            handle,
            data=data,
            **kwargs
        )
        datas.attrs["sampling_rate"] = sampling_rate
            
        return (True)


    def gather(self, starttime, endtime, tag, traces=slice(None)):
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
        gather: dash.core.Gather
            Gather object containing desired data.
        """
        
        starttime = pd.to_datetime(starttime, utc=True)
        endtime   = pd.to_datetime(endtime, utc=True)

        index = self.index
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
                handle         = row["handle"]
                istart = _sample_idx(starttime, data_starttime, sampling_rate)
                iend   = _sample_idx(min(endtime, data_endtime), data_starttime, sampling_rate) + 1
                datas = self["/".join(("waveforms", tag, handle))]
                jend = jstart + (iend - istart)
                data[..., jstart: jend] = datas[..., istart: iend]
                jstart = jend
 
            starttime = data_starttime + pd.to_timedelta(istart / sampling_rate, unit="S")
            trace_idxs = np.arange(data.shape[0])
            gather = Gather(data, first_sample, sampling_rate, trace_idxs)
            gathers.append(gather)
            
        if len(gathers) == 1:
            return (gathers[0])
        else:
            return (gathers)

        
    def link_files(self, path, prefix=""):
        """
        ***Deprecated.***
        
        Traverse the directory specified by `path` and create symbolic
        links to all files.

        Returns True upon successful completion.
        """

        ddir = pathlib.Path(path)
        
        for path in _list_files(ddir):
            subpath = pathlib.Path(str(path).replace(str(ddir), ""))
            group = str(subpath.parent)
            try:
                with h5py.File(path, mode="r") as f5s:
                    for key in f5s["/waveforms"]:
                        handle = "/".join(("/waveforms", prefix, group, key))
                        self[handle] = h5py.ExternalLink(path, key)
            except PermissionError:
                print(f"Warning: Could not read file {path}")

        self.update_index()

        return (True)
    
    def link_data(self, path, prefix="", suffix=""):
        """
        Traverse the directory specified by `path` and create symbolic
        links to all contained data sets.

        Returns True upon successful completion.
        """

        root = pathlib.Path(path)
        
        if root.is_file():
            self._link_file(root, root.parent, prefix=prefix, suffix=suffix)
        
        else:        
            for path in _list_files(root):
                try:
                    self._link_file(path, root, prefix=prefix, suffix=suffix)
                except:
                    print(path)

        self.update_index()

        return (True)


    def _link_data_set(self, datas, path, root, prefix="", suffix=""):
        tag = "/".join((
            "/waveforms",
            prefix,
            str(path.parent).removeprefix(str(root)),
            datas.parent.name.removeprefix("/waveforms"),
            suffix,
        ))
        tag = re.sub("//+", "/", tag)
        basename = datas.name.split("/")[-1]
        handle = str(pathlib.Path(tag, basename))
        self[handle] = h5py.ExternalLink(path, datas.name)


    def _link_file(self, path, root, prefix="", suffix=""):
        with h5py.File(path, mode="r") as f5s:
            for handle in _iter_group(f5s["/waveforms"]):
                datas = f5s[handle]
                self._link_data_set(datas, path, root, prefix=prefix, suffix=suffix)
    
    def update_index(self):

        self._set_mode("r")

        rows = list()

        for path in _iter_group(self["/waveforms"]):
            handle = path.split("/")[-1]
            starttime = handle.split("__")[1]
            tag = "/".join(path.split("/")[2:-1])
            shape = self[path].shape
            sampling_rate = self[path].attrs["sampling_rate"]
            if isinstance(sampling_rate, np.ndarray):
                sampling_rate = sampling_rate[0]
            row = (starttime, tag, handle, shape, sampling_rate)
            rows.append(row)

        columns = ["starttime", "tag", "handle", "shape", "sampling_rate"]
        dataf = pd.DataFrame(rows, columns=columns)
        dataf["starttime"] = pd.to_datetime(dataf["starttime"])
        sampling_interval = pd.to_timedelta(1 / dataf["sampling_rate"], unit="S")
        nsamples = dataf["shape"].apply(lambda x: x[-1])
        dataf["endtime"] = dataf["starttime"] + sampling_interval * (nsamples - 1)

        self._index = dataf
        filename = self.filename
        self.close()
        dataf.to_hdf(filename, key="/waveforms/_index")

        self._reset_mode()

        return (True)
        
        

class Gather(object):

    def __init__(self, data, starttime, sampling_rate, trace_idxs):
        """
        A class to contain, process, and visualize DAS data.

        Positional Arguments
        ====================
        data: np.ndarray
            2D array of waveform data with trace index along the first
            axis and time along the second.
        starttime: int, float, str, datetime
            Time of first sample in *data*.
        sampling_rate: int, float
            Sampling rate of *data*.
        trace_idxs: array-like
            1D array of indices of traces in *data*.
        """
        self._data = data
        self._starttime = pd.to_datetime(starttime, utc=True)
        self._sampling_rate = sampling_rate
        self._trace_idxs = np.array(trace_idxs)

    @property
    def data(self):
        """
        Waveform data.
        """
        return (self._data)

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def dataf(self):
        """
        Contained data as a pandas.DataFrame.
        """

        data = self.data

        if data.ndim == 2:
            dataf = pd.DataFrame(
                data.T,
                index=self.times,
                columns=self.trace_idxs
            )

        elif data.ndim == 3:
            nrec, nchan, nsamp = data.shape
            rec = np.repeat(self.trace_idxs, nchan)
            chan = np.tile(range(nchan), nrec)
            columns = pd.MultiIndex.from_arrays(
                (rec, chan),
                names=("receiver", "channel")
            )
            dataf = pd.DataFrame(
                data.reshape(-1, data.shape[-1]).T,
                columns=columns,
                index=self.times
            )
        else:
            raise (NotImplementedError())

        return(dataf)


    @property
    def datas(self):
        """
        Contained data as a obspy.Stream.
        """
        import obspy
        stream = obspy.Stream()
        for i, d in enumerate(self.data):
            trace = obspy.Trace(
                data=d.copy(),
                header=dict(
                    starttime=self.starttime,
                    sampling_rate=self.sampling_rate,
                    network="??",
                    station=f"{i+1:04d}",
                    location="??",
                    channel="???"
                )
            )
            stream.append(trace)

        return (stream)

    @property
    def endtime(self):
        """
        Time of last sample in self.data.
        """
        return (
              self.starttime
            + pd.to_timedelta((self.nsamples-1) / self.sampling_rate, unit="S")
        )
    @property
    def f(self):
        """
        Frequency coordinates of data in fk domain.
        """
        f = np.fft.fftfreq(self.nsamples, d=self.sampling_interval)
        return (f)

    @property
    def k(self):
        """
        Wavenumber coordinates of data in fk domain.
        """
        k = np.fft.fftfreq(self.ntraces)

        return (k)

    @property
    def norm_coeff(self):
        """
        Array of normalization constants.
        """
        if self._data.ndim == 3:
            reshape = np.atleast_3d
        else:
            reshape = lambda x: np.transpose(np.atleast_2d(x))

        return (reshape(np.max(np.abs(self.data), axis=-1)))

    @property
    def ntraces(self):
        """
        Number of traces in data.
        """
        return (self.data.shape[0])

    @property
    def nsamples(self):
        """
        Number of time samples in data.
        """
        return (self.data.shape[-1])

    @property
    def nyquist(self):
        """
        Nyquist frequency of data.
        """
        return (self.sampling_rate / 2)

    @property
    def sampling_interval(self):
        """
        Time interval between data samples.
        """
        return (1 / self.sampling_rate)

    @property
    def sampling_rate(self):
        """
        Temporal sampling rate of data.
        """
        return (self._sampling_rate)

    @property
    def starttime(self):
        """
        Time of first sample in data.
        """
        return (self._starttime)

    @property
    def times(self):
        """
        Time coordinates of data in data domain.
        """
        times = pd.date_range(
            start=self.starttime,
            periods=self.nsamples,
            freq=f"{1/self.sampling_rate}S"
        )
        return (times)

    @property
    def trace_idxs(self):
        """
        Trace indices of data in data domain.
        """
        return (self._trace_idxs)


    def apply_fk_mask(self, mask):
        """
        Apply fk-mask, *mask*, to data in place.
        """
        self._data = np.real(np.fft.ifft2(np.fft.fft2(self.data) * mask))


    def bandpass(self, horder=2, locut=1, hicut=300):
        """
        Bandpass data in place.
        """
        nyq = self.sampling_rate / 2
        sos = scipy.signal.butter(horder, [locut/nyq, hicut/nyq], btype="bandpass", output="sos")
        self._data = scipy.signal.sosfiltfilt(sos, self.data)


    def copy(self):
        gather = Gather(
            self.data.copy(),
            self.starttime,
            self.sampling_rate,
            self.trace_idxs.copy()
        )

        return (gather)


    def decimate(self, factor):
        """
        Decimate data in place.
        """
        self._data = scipy.signal.decimate(self.data, factor)
        self._sampling_rate /= factor


    def demean(self):
        """
        Demean data in place.
        """
        if self._data.ndim == 3:
            reshape = np.atleast_3d
        else:
            reshape = lambda x: np.transpose(np.atleast_2d(x))
        self._data = self.data - reshape(np.mean(self.data, axis=-1))


    def denormalize(self, norm):
        """
        Denormalize data in place.
        """
        self._data = self.data * norm


    def normalize(self, norm_coeff=None):
        """
        Normalize data in place.
        """
        if norm_coeff is None:
            norm_coeff = self.norm_coeff

        self._data = self.data / norm_coeff


    def plot(self, domain="time", type="DAS", **kwargs):
        """
        Plot data in time or fk domain.
        """

        if domain == "time":

            if type == "DAS":

                return (self._plot_time_domain(**kwargs))

            elif type == "geophone":

                return (self._plot_geophone(**kwargs))


        elif domain == "fk":

            return (self._plot_fk_domain(**kwargs))

    def sample_idx(self, time, right=False):
        """
        Return the sampling index corresponding to the given time.
        """
        return (
            _sample_idx(
                time, self.starttime, self.sampling_rate, right=right
            )
        )


    def trim(self, starttime=None, endtime=None, right=False):
        """
        Trim data to desired start and endtime (in place).
        """
        if starttime is not None:
            istart = self.sample_idx(starttime, right=right)
        else:
            istart = 0

        if endtime is not None:
            iend   = self.sample_idx(endtime, right=right)
        else:
            iend = self.nsamples

        self._data = self.data[..., istart: iend]
        delta = pd.to_timedelta(istart / self.sampling_rate, unit="S")
        self._starttime = self.starttime + delta


    def write(self, path, tag="", **kwargs):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with File(path, mode="a") as f5:
            f5.add_waveforms(self.data, self.starttime, self.sampling_rate, tag=tag, **kwargs)

        return (True)


    def write_binary(self, path, dtype=None):
        """
        Write data to `path` in raw binary format. Optionally cast
        data to `dtype`.
        """
        d = self.data.flatten()
        if dtype is not None and d.dtype != dtype:
            d = d.astype(dtype)
        with open(path, "wb") as outfile:
            d.tofile(outfile)

        return (True)


    def _write(self, path, **kwargs):
        """
        DEPRECATED
        Write data to disk.
        """
        starttime = self.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")
        endtime = self.endtime.strftime("%Y-%m-%dT%H:%M:%S.%f")
        with h5py.File(path, mode="w") as f5out:
            ds = f5out.create_dataset(
                f"waveforms/{starttime}__{endtime}",
                data=self.data,
                dtype=np.float32,
                **kwargs
            )
            ds.attrs["starttime"] = starttime
            ds.attrs["sampling_rate"] = self.sampling_rate
            ds.attrs["sample_interval"] = 1 / self.sampling_rate


    def _plot_fk_domain(self, amax=None, figsize=None, cmap=plt.get_cmap("magma")):
        fft = np.fft.fft2(self.data)
        k   = np.fft.fftfreq(self.ntraces)
        f   = np.fft.fftfreq(self.nsamples, d=self.sampling_interval)

        fig, ax = plt.subplots(figsize=figsize)

        qmesh = ax.pcolorfast(
            np.fft.fftshift(f),
            np.fft.fftshift(k),
            np.abs(np.fft.fftshift(fft)),
            vmin=0,
            vmax=np.percentile(np.abs(fft), 99) if amax is None else amax,
            cmap=cmap
        )

        ax.set_xlim(0, np.max(f))
        ax.set_xlabel("Frequency ($s^{-1})$")
        ax.set_ylabel("Wavenumber ($m^{-1}$)");

        cbar = fig.colorbar(qmesh, ax=ax)
        cbar.set_label("Amplitude")

        plt.tight_layout()

        return (ax)


    def _plot_time_domain(self, amax=None, figsize=None, cmap=plt.get_cmap("bone")):

        if amax is None:
            amax = np.amax(np.abs(self.data))
        mpltimes = mpl.dates.date2num(self.times)

        fig, ax = plt.subplots(figsize=figsize)
        qmesh = ax.pcolorfast(
            self.trace_idxs,
            mpltimes,
            self.data.T,
            cmap=cmap,
            vmin=-amax,
            vmax=amax
        )

        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.set_xlim(self.trace_idxs[0], self.trace_idxs[-1])
        ax.set_xlabel("Trace index")

        locator = mpl.dates.AutoDateLocator()
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(locator))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.set_ylim(mpltimes[0], mpltimes[-1])
        ax.invert_yaxis()
        ax.set_ylabel("Time")

        cbar = fig.colorbar(qmesh, ax=ax)
        cbar.set_label("Amplitude")

        plt.tight_layout()

        return (ax)

    def _plot_geophone(self):
        fig, ax = plt.subplots()
        for itrace, trace in enumerate(self.data[:, 0]):
            ax.plot(trace + itrace, self.times, color="k", linewidth=1)

        ax.set_xlabel("Trace index")
        ax.set_ylim(self.times.min(), self.times.max())
        ax.invert_yaxis()
        locator = mpl.dates.AutoDateLocator()
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(locator))
        plt.tight_layout()


def build_fk_notch_mask(k0, alpha, k, n):
    mask = 1 - np.exp(-np.abs(alpha*(k0-k)))
    mask = np.repeat(np.atleast_2d(mask), n, axis=0).T

    return (mask)


def build_fk_slope_mask(m, sigma, f, k):
    mask   = np.zeros((len(k), len(f)))

    for i in range(1, len(f)):
        for sign in (-1, 1):
            mask[:, i] += scipy.stats.norm(
                loc=sign*m*f[i],
                scale=sigma
            ).pdf(k)

    return (mask)


def plot_fk_mask(f, k, mask, figsize=None, cmap=plt.get_cmap("plasma")):
    fig, ax = plt.subplots(figsize=figsize)
    qmesh = ax.pcolorfast(
        np.fft.fftshift(f),
        np.fft.fftshift(k),
        np.fft.fftshift(mask),
        cmap=cmap
    )
    ax.set_xlabel("Frequency ($s^{-1}$)")
    ax.set_xlim(0, np.max(f))

    ax.set_ylabel("Wavenumber ($m^{-1}$)")

    cbar = fig.colorbar(qmesh, ax=ax)
    cbar.set_label("Amplitude")

    plt.tight_layout()

    return (ax)


def _iter_group(group):
    """
    Return a list of all data sets in group. Skip special "_index" group.
    """
    groups = list()
    for key in group:
        if key == "_index":
            continue
        elif isinstance(group[key], h5py.Group):
            groups += _iter_group(group[key])
        else:
            groups.append("/".join((group.name, key)))

    return (groups)


def _list_files(path):
    """
    Generator of paths to all files in `path` and its subdirectories.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            yield (pathlib.Path(root).joinpath(file))


def _repr_group(group, indent=""):
    s = ""
    for key in group:
        s += f"{indent}+--{key}\n"
        if isinstance(group[key], h5py.Group):
            s += _repr_group(group[key], indent=indent+"|--")
    return (s)


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
