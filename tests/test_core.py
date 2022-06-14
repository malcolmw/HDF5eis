#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:32:43 2022

@author: malcolmw
"""
# Standard library imports
import string
import tempfile
import unittest

# Third-party imports
import h5py
import numpy as np
import pandas as pd

# Local imports
import hdf5eis

class TestAccessorBaseClassMethods(unittest.TestCase):

    def test_column_io(self):
        with h5py.File(tempfile.TemporaryFile(), mode="w") as file:
            accessor = hdf5eis.core.AccessorBase(file, "/root")
            accessor.root.create_group("TEST")
            # Test float IO
            self._test_float_io(accessor)
            # Test int IO
            self._test_int_io(accessor)
            # Test string IO
            self._test_string_io(accessor)
            # Test DateTime IO
            self._test_datetime_io(accessor)

    def test_table_io(self):
        dataf = random_table(1024)

        with h5py.File(tempfile.TemporaryFile(), mode="w") as file:
            accessor = hdf5eis.core.AccessorBase(file, "/root")
            accessor._add_table(dataf, "TEST")
            columns = np.sort(dataf.columns)
            dataf["string"]= dataf["string"].str.decode("UTF-8")
            self.assertTrue(np.all(
                dataf[columns] == accessor._read_table("TEST")[columns]
            ))

    def _test_float_io(self, accessor):
        for dtype in (np.float16, np.float32, np.float64, np.float128):
            data = pd.Series(
                np.random.rand(1024).astype(dtype),
                name=str(dtype)
            )
            accessor._write_column(data, "TEST")
            handle = f"TEST/{dtype}"
            self.assertTrue(accessor.root[handle].dtype == dtype)
            self.assertTrue(
                np.all(data == accessor.root[handle][:])
            )
            self.assertFalse(accessor.root[handle].attrs["__IS_UTF8"])
            self.assertFalse(accessor.root[handle].attrs["__IS_UTC_DATETIME64"])

    def _test_int_io(self, accessor):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            data = pd.Series(
                np.random.randint(
                    np.iinfo(dtype).min,
                    np.iinfo(dtype).max,
                    size=(1024,)
                ).astype(dtype),
                name=str(dtype)
            )
            accessor._write_column(data, "TEST")
            handle = f"TEST/{dtype}"
            self.assertTrue(accessor.root[handle].dtype == dtype)
            self.assertTrue(
                np.all(data == accessor.root[handle][:])
            )
            self.assertFalse(accessor.root[handle].attrs["__IS_UTF8"])
            self.assertFalse(accessor.root[handle].attrs["__IS_UTC_DATETIME64"])

    def _test_string_io(self, accessor):
        strings = random_strings(1024)
        accessor._write_column(strings, "TEST")
        handle = "TEST/string"
        self.assertTrue(np.all(strings == pd.Series(accessor.root[handle][:])))
        self.assertTrue(accessor.root[handle].attrs["__IS_UTF8"])
        self.assertFalse(accessor.root[handle].attrs["__IS_UTC_DATETIME64"])

    def _test_datetime_io(self, accessor):
        times = random_times(1024)
        accessor._write_column(times, "TEST")
        handle = "TEST/time"
        self.assertTrue(np.all(
            times == pd.Series(pd.to_datetime(accessor.root[handle][:], utc=True))
        ))
        self.assertFalse(accessor.root[handle].attrs["__IS_UTF8"])
        self.assertTrue(accessor.root[handle].attrs["__IS_UTC_DATETIME64"])


class TestAuxiliaryAccessorClassMethods(unittest.TestCase):
    def test_add(self):
        with h5py.File(tempfile.TemporaryFile(), mode="w") as file:
            accessor = hdf5eis.core.AuxiliaryAccessor(file, "/root")
            dataf = random_table(1024)
            accessor.add(dataf, "TEST")
            dataf["string"] = dataf["string"].str.decode("UTF-8")
            columns = np.sort(dataf.columns)
            self.assertTrue(
                np.all(dataf[columns] == accessor["TEST"][columns])
            )
            string = random_string(max_length=8192)
            accessor.add(string, "string")
            self.assertTrue(string == accessor["string"])


class TestWaveformAccessorClassMethods(unittest.TestCase):
    def test_add(self):
        shape = (8, 8, 3, 500)
        now = pd.Timestamp.now(tz="UTC")
        delta = pd.to_timedelta(1/100, unit="S")
        with h5py.File(tempfile.TemporaryFile(), mode="w") as file:
            accessor = hdf5eis.core.WaveformAccessor(file, "/root")
            
            # Test float IO
            data_out = np.random.rand(*shape)
            for dtype in (np.float16, np.float32, np.float64, np.float128):
                accessor.add(
                    data_out.astype(dtype), 
                    now,
                    100,
                    tag=str(dtype)
                )
                data_in = accessor[
                    str(dtype), :, :, :, now: now+delta*500*60
                ][str(dtype)][0].data
                self.assertTrue(np.all(data_in == data_out.astype(dtype)))
                
            # Test int IO
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                data_out = np.random.randint(
                    np.iinfo(dtype).min,
                    np.iinfo(dtype).max,
                    size=shape
                ).astype(dtype)
                accessor.add(
                    data_out,
                    now,
                    100,
                    tag=str(dtype)
                )
                data_in = accessor[
                    str(dtype), :, :, :, now: now+delta*500*60
                ][str(dtype)][0].data
                self.assertTrue(np.all(data_in == data_out))
        

def random_string(max_length=32):
    return "".join(
        np.random.choice(
            list(string.ascii_letters + string.digits),
            size=np.random.randint(1, max_length+1)
        )
    )

def random_strings(nstrings):
    return pd.Series(
        [random_string() for i in range(nstrings)],
        name="string"
    ).astype(np.dtype("S"))


def random_table(nrows):
    dataf = pd.DataFrame()
    for dtype in (np.float16, np.float32, np.float64, np.float128):
        dataf[str(dtype)] = np.random.rand(nrows).astype(dtype)
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        dataf[str(dtype)] = np.random.randint(
            np.iinfo(dtype).min,
            np.iinfo(dtype).max,
            (nrows,)
        ).astype(dtype)

    dataf["string"] = random_strings(nrows)
    dataf["time"] = random_times(nrows)
    return dataf

def random_times(ntimes):
    return pd.Series(
        pd.to_datetime(
            np.random.randint(
                0,
                pd.Timestamp.now().timestamp(),
                size=(ntimes,)
            )*1e9,
            utc=True
        ),
        name="time"
    )


if __name__ == "__main__":
    unittest.main()
