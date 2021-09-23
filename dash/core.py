import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import scipy.signal


class HDF5DAS(h5py.File):
    def __init__(self, *args, dtype=np.float32, safe_mode=True, **kwargs):
        """
        An h5py.File subclass for convenient I/O of DAS data.
        """
        self._dtype = dtype
        if (
            "mode" in kwargs
            and kwargs["mode"] == "w"
            and safe_mode is True
            and pathlib.Path(args[0]).exists()
        ):
            raise (
                ValueError(
                    "File already exists! If you are sure you want to open it"
                    " in write mode, use `safe_mode=False`."
                )
            )
        super(HDF5DAS, self).__init__(*args, **kwargs)


    @property
    def dtype(self):
        return (self._dtype)


    def gather(self, starttime, endtime, traces=slice(None)):
        """
        Return a dash.core.Gather object with data between *starttime*
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
        for key in self["waveforms"].keys():
            data_starttime, data_endtime = key.split("__")
            data_starttime = pd.to_datetime(data_starttime, utc=True)
            data_endtime = pd.to_datetime(data_endtime, utc=True)
            if starttime <= data_endtime and endtime >= data_starttime:
                ds = self[f"waveforms/{key}"]
                sampling_rate = ds.attrs["sampling_rate"]
                istart = sample_index(starttime, data_starttime, sampling_rate)
                iend   = sample_index(endtime, data_starttime, sampling_rate)
                data = ds[traces, istart: iend]
                starttime = data_starttime + pd.to_timedelta(istart / sampling_rate, unit="S")
                trace_idxs = np.arange(ds.shape[0])[traces]
                gather = Gather(data, starttime, sampling_rate, trace_idxs)

                return (gather)

        raise(ValueError("Invalide time range specified."))

    def write(self, gather, **kwargs):
        """
        Write data to disk.
        """
        starttime = gather.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")
        endtime = gather.endtime.strftime("%Y-%m-%dT%H:%M:%S.%f")
        ds = self.create_dataset(
            f"waveforms/{starttime}__{endtime}",
            data=gather.data,
            dtype=self.dtype,
            **kwargs
        )
        ds.attrs["starttime"] = starttime
        ds.attrs["sampling_rate"] = gather.sampling_rate
        ds.attrs["sample_interval"] = 1 / gather.sampling_rate


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
        df = pd.DataFrame(
            self.data.T,
            index=self.times,
            columns=self.trace_idxs
        )
        return (df)

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
        return (self.data.shape[1])

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
        self._data = self.data - np.atleast_2d(np.mean(self.data, axis=1)).T


    def normalize(self):
        """
        Normalize data in place.
        """
        self._data = self.data / np.atleast_2d(np.max(np.abs(self.data), axis=1)).T


    def plot(self, domain="time", **kwargs):
        """
        Plot data in time or fk domain.
        """

        if domain == "time":

            return (self._plot_time_domain(**kwargs))

        elif domain == "fk":

            return (self._plot_fk_domain(**kwargs))

    def sample_index(self, time, rounder=np.round):
        """
        Return the sampling index corresponding to the given time.
        """
        return (
            sample_index(
                time, self.starttime, self.sampling_rate, rounder=rounder
            )
        )


    def trim(self, starttime=None, endtime=None, lrounder=np.round, urounder=np.round):
        """
        Trim data to desired start and endtime (in place).
        """
        if starttime is not None:
            istart = self.sample_index(starttime, rounder=lrounder)
        else:
            istart = 0

        if endtime is not None:
            iend   = self.sample_index(endtime, rounder=urounder)
        else:
            iend = self.nsamples

        self._data = self.data[:, istart: iend]
        delta = pd.to_timedelta(istart / self.sampling_rate, unit="S")
        self._starttime = self.starttime + delta


    def write(self, path, **kwargs):
        with HDF5DAS(path, mode="w") as f5:
            f5.write(self, **kwargs)

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


def sample_index(time, starttime, sampling_rate, rounder=np.round):
    """
    Return the sampling index corresponding to the given time.
    """
    time = pd.to_datetime(time, utc=True)
    delta = (time - starttime).total_seconds()
    if delta < 0:
        return (0)
    else:
        return (np.uint(rounder(delta * sampling_rate)))
