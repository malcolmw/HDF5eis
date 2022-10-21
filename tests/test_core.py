#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:32:43 2022

@author: malcolmw
"""
# Standard library imports
import pathlib
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

    def test_table_io(self):
        dataf = random_table(1024)

        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AccessorBase(file, "/root")
            accessor.add_table(dataf, "TEST")
            columns = np.sort(dataf.columns)
            table, fmt = accessor.read_table("TEST")
            self.assertTrue(np.all(
                dataf[columns] == table[columns]
            ))


    def test_float_io(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AccessorBase(file, "/root")
            accessor.root.create_group("TEST")
            for dtype in (np.float16, np.float32, np.float64):
                data = pd.Series(
                    np.random.rand(1024).astype(dtype),
                    name=str(dtype)
                )
                accessor.write_column(data, "TEST")
                handle = f"TEST/{dtype}"
                self.assertTrue(accessor.root[handle].dtype == dtype)
                self.assertTrue(
                    np.all(data == accessor.root[handle][:])
                )
                self.assertFalse(accessor.root[handle].attrs["__IS_UTF8"])
                self.assertFalse(
                    accessor.root[handle].attrs["__IS_UTC_DATETIME64"]
                )

    def test_int_io(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AccessorBase(file, "/root")
            accessor.root.create_group("TEST")
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                data = pd.Series(
                    np.random.randint(
                        np.iinfo(dtype).min,
                        np.iinfo(dtype).max,
                        size=(1024,),
                        dtype=dtype
                    ),
                    name=str(dtype)
                )
                accessor.write_column(data, "TEST")
                handle = f"TEST/{dtype}"
                self.assertTrue(accessor.root[handle].dtype == dtype)
                self.assertTrue(
                    np.all(data == accessor.root[handle][:])
                )
                self.assertFalse(accessor.root[handle].attrs["__IS_UTF8"])
                self.assertFalse(
                    accessor.root[handle].attrs["__IS_UTC_DATETIME64"]
                )

    def test_string_io(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AccessorBase(file, "/root")
            accessor.root.create_group("TEST")
            strings = random_strings(1024)
            accessor.write_column(strings, "TEST")
            handle = "TEST/string"
            self.assertTrue(np.all(
                strings
                ==
                hdf5eis.core.UTF8_DECODER(pd.Series(accessor.root[handle][:]))
            ))
            self.assertTrue(accessor.root[handle].attrs["__IS_UTF8"])
            self.assertFalse(accessor.root[handle].attrs["__IS_UTC_DATETIME64"])

    def test_datetime_io(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AccessorBase(file, "/root")
            accessor.root.create_group("TEST")
            times = random_times(1024)
            accessor.write_column(times, "TEST")
            handle = "TEST/time"
            self.assertTrue(np.all(
                times == pd.Series(pd.to_datetime(accessor.root[handle][:], utc=True))
            ))
            self.assertFalse(accessor.root[handle].attrs["__IS_UTF8"])
            self.assertTrue(accessor.root[handle].attrs["__IS_UTC_DATETIME64"])


class TestAuxiliaryAccessorClassMethods(unittest.TestCase):
    def test_add(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, "/root")
            dataf = random_table(1024)
            accessor.add(dataf, "TEST")
            columns = np.sort(dataf.columns)
            table, fmt = accessor["TEST"]
            self.assertTrue(
                np.all(dataf[columns] == table[columns])
            )
            my_string = random_string(max_length=8192)
            accessor.add(my_string, "string")
            self.assertTrue(my_string == accessor["string"][0])

    def test_validate0(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Ensure that properly formatted tables don't raise an error.
            dataf = random_table(1024)
            accessor.add(dataf, 'TEST')
            accessor.validate()
            del(accessor.root['TEST'])


    def test_validate_column_length(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Add a column with a bad length and ensure it raises a
            # TableFormatError.
            dataf = random_table(1024)
            accessor.add(dataf, 'TEST')
            accessor.root.create_dataset(
                'TEST/Bad Column',
                data=np.random.rand(1025)
            )
            with self.assertRaises(hdf5eis.exceptions.TableFormatError):
                accessor.validate()
            del(accessor.root['TEST'])

    def test_validate_missing_is_utc_datetime64_attr(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Delete __IS_UTC_DATETIME64 Attribute from one column and ensure
            # that TableFormatError is raised.
            dataf = random_table(1024)
            accessor.add(dataf, 'TEST')
            del(accessor.root['TEST/string'].attrs['__IS_UTC_DATETIME64'])
            with self.assertRaises(hdf5eis.exceptions.TableFormatError):
                accessor.validate()
            del(accessor.root['TEST'])

    def test_validate_missing_is_utf8_attr(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Delete __IS_UTF8 Attribute from one column and ensure
            # that TableFormatError is raised.
            dataf = random_table(1024)
            accessor.add(dataf, 'TEST')
            del(accessor.root['TEST/string'].attrs['__IS_UTF8'])
            with self.assertRaises(hdf5eis.exceptions.TableFormatError):
                accessor.validate()
            del(accessor.root['TEST'])

    def test_validate_is_utc_datetime64_and_is_utf8_both_true(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Set __IS_UTC_DATETIME64 and __IS_UTF8 Attributes both to True and
            # ensure that TableFormatError is raised.
            dataf = random_table(1024)
            accessor.add(dataf, 'TEST')
            accessor.root['TEST/string'].attrs['__IS_UTF8'] = True
            accessor.root['TEST/string'].attrs['__IS_UTC_DATETIME64'] = True
            with self.assertRaises(hdf5eis.exceptions.TableFormatError):
                accessor.validate()
            del(accessor.root['TEST'])


    def test_validate_is_utc_datetime64_true_wrong_dtype(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Set __IS_UTC_DATETIME64 to True for a column with the wrong dtype
            # ensure that TableFormatError is raised.
            dataf = random_table(1024)
            accessor.add(dataf, 'TEST')
            accessor.root[f'TEST/{np.int16}'].attrs['__IS_UTC_DATETIME64'] = True
            with self.assertRaises(hdf5eis.exceptions.TableFormatError):
                accessor.validate()
            del(accessor.root['TEST'])

    def test_validate_is_utf8_true_wrong_dtype(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Set __IS_UTF8 to True for a column with the wrong dtype
            # ensure that TableFormatError is raised.
            dataf = random_table(1024)
            accessor.add(dataf, 'TEST')
            accessor.root[f'TEST/{np.int16}'].attrs['__IS_UTF8'] = True
            with self.assertRaises(hdf5eis.exceptions.TableFormatError):
                accessor.validate()
            del(accessor.root['TEST'])

    def test_validate_utf8_string_wrong_dtype(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.AuxiliaryAccessor(file, '/root')
            # Create a DataSet with __TYPE == UTF-8 but an invalid dtype and
            # ensure a UTF8FormatError is raised.
            accessor.root.create_dataset(
                'TEST', data=np.random.rand(32)
            )
            accessor.root['TEST'].attrs['__TYPE'] = 'UTF-8'
            with self.assertRaises(hdf5eis.exceptions.UTF8FormatError):
                accessor.validate()
            del(accessor.root['TEST'])


class TestTimeseriesAccessorClassMethods(unittest.TestCase):

    def test_link_tag(self):
        shape = (8, 8, 3, 500)
        now = pd.Timestamp.now(tz="UTC")
        delta = pd.to_timedelta(1/100, unit="S")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            path1 = temp_dir.joinpath("file1.hdf5")
            path2 = temp_dir.joinpath("file2.hdf5")
            path3 = temp_dir.joinpath("file3.hdf5")
            path4 = temp_dir.joinpath("file4.hdf5")
            master_path = temp_dir.joinpath("master.hdf5")
            for ipath, path in enumerate((path1, path2, path3, path4)):
                with h5py.File(path, mode="w") as file:
                    accessor = hdf5eis.core.TimeseriesAccessor(file, "/timeseries")
                    accessor.add(
                        np.random.rand(*shape).astype(np.float32),
                        now,
                        100,
                        tag=f"file{ipath+1}"
                    )
            with h5py.File(master_path, mode="w") as file:
                accessor = hdf5eis.core.TimeseriesAccessor(file, "/timeseries")
                accessor.link_tag(path1, "file1")
                accessor.link_tag(
                    path2,
                    "file2",
                    prefix="prefix",
                    suffix="suffix"
                )
                accessor.link_tag(path3, "file3", new_tag="FILE3")
                accessor.link_tag(
                    path4,
                    "file4",
                    new_tag="FILE4",
                    prefix="prefix"
                )
                self.assertEqual(accessor.index.loc[0, "tag"], "file1")
                self.assertEqual(
                    accessor.index.loc[1, "tag"],
                    "prefix/file2/suffix"
                )
                self.assertEqual(accessor.index.loc[2, "tag"], "FILE3")
                self.assertEqual(accessor.index.loc[3, "tag"], "FILE4")
                _ = accessor["file1", :, :, :, now:now+delta*500]
                _ = accessor["prefix/file2/suffix", :, :, :, now:now+delta*500]
                _ = accessor["FILE3", :, :, :, now:now+delta*500]
                _ = accessor["FILE4", :, :, :, now:now+delta*500]




    def test_add_float(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.TimeseriesAccessor(file, "/timeseries")
            shape = (8, 8, 3, 500)
            now = pd.Timestamp.now(tz="UTC")
            delta = pd.to_timedelta(1/100, unit="S")
            data_out = np.random.rand(*shape)
            for dtype in (np.float16, np.float32, np.float64):
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

    def test_add_int(self):
        with tempfile.TemporaryFile() as tmp_file, \
             h5py.File(tmp_file, mode="w") as file \
        :
            accessor = hdf5eis.core.TimeseriesAccessor(file, "/timeseries")
            shape = (8, 8, 3, 500)
            now = pd.Timestamp.now(tz="UTC")
            delta = pd.to_timedelta(1/100, unit="S")
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                data_out = np.random.randint(
                    np.iinfo(dtype).min,
                    np.iinfo(dtype).max,
                    size=shape,
                    dtype=dtype
                )
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
    )


def random_table(nrows):
    dataf = pd.DataFrame()
    for dtype in (np.float16, np.float32, np.float64):
        dataf[str(dtype)] = np.random.rand(nrows).astype(dtype)
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        dataf[str(dtype)] = np.random.randint(
            np.iinfo(dtype).min,
            np.iinfo(dtype).max,
            (nrows,),
            dtype=dtype
        )

    dataf["string"] = random_strings(nrows)
    dataf["time"] = random_times(nrows)
    return dataf

def random_times(ntimes):
    return pd.Series(
        pd.to_datetime(
            np.random.randint(
                0,
                pd.Timestamp.now().timestamp(),
                size=(ntimes,),
                dtype=np.int64
            )*1e9,
            utc=True
        ),
        name="time"
    )


if __name__ == "__main__":
    unittest.main()
