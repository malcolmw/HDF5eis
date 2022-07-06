"""
A module implementing the Gather container class for multidimensional
timeseries data.

"""

import pandas as pd


class Gather:
    """
    A container class for multidimensional timeseries data.

    """

    def __init__(self, data, start_time, sampling_rate):
        """
        A container class for multidimensional timeseries data.

        Parameters
        ----------
        data : numpy.ndarray
            Array of multidimensional timeseries data with time
            oriented along the last axis.
        start_time : pandas.Timestamp
            Time of first sample in array.
        sampling_rate : int, float
            Sampling rate (in units of samples per second) of timeseries
            data in array.

        Returns
        -------
        None.

        """
        self._data = data
        self._start_time = pd.to_datetime(start_time, utc=True)
        self._sampling_rate = sampling_rate

    @property
    def data(self):
        """
        Multidimensional timeseries array with time oriented along the
        last axis.

        Returns
        -------
        np.ndarray
            Multidimensional timeseries array with time oriented along the
            last axis.

        """
        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for data attribute.

        Parameters
        ----------
        value : np.ndarray
            Multidimensional timeseries array with time oriented along the
            last axis.

        Returns
        -------
        None.

        """
        self._data = value

    @property
    def end_time(self):
        """
        The time of the last sample in this Gather.

        Returns
        -------
        pandas.Timestamp
            The time of the last sample in this Gather.

        """
        return (
            self.start_time
            + pd.to_timedelta((self.nsamples-1) / self.sampling_rate, unit="S")
        )

    @property
    def nsamples(self):
        """
        Number of time samples in this Gather.

        Returns
        -------
        int
            Number of time samples in this Gather.

        """
        return self.data.shape[-1]

    @property
    def sampling_interval(self):
        """
        Time interval (in seconds) between data samples.

        Returns
        -------
        float
            Time interval (in seconds) between data samples.

        """
        return 1 / self.sampling_rate

    @property
    def sampling_rate(self):
        """
        Temporal sampling rate (in units of samples per second) of this
        Gather.

        Returns
        -------
        float
            Temporal sampling rate (in units of samples per second) of this
            Gather.

        """
        return self._sampling_rate

    @property
    def start_time(self):
        """
        Time of first sample in this Gather.

        Returns
        -------
        pandas.Timestamp
            Time of first sample in this Gather.

        """
        return self._start_time

    @property
    def times(self):
        """
        Time coordinates of samples in this Gather.

        Returns
        -------
        times : pandas.DatetimeIndex
            TIme coordinates of samples in this Gather.

        """
        times = pd.date_range(
            start=self.start_time,
            periods=self.nsamples,
            freq=f"{1/self.sampling_rate}S"
        )
        return times

    def copy(self):
        """
        Return a deep copy of this Gather.

        Returns
        -------
        gather : hdf5eis.Gather
            A deep copy of this Gather.

        """
        gather = Gather(
            self.data.copy(),
            self.start_time,
            self.sampling_rate
        )

        return gather
