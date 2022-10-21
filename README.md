# HDF5eis Python API (read H-D-F-Size)
#### A solution for handling big, multidimensional timeseries data from environmental sensors in HPC applications.

HDF5eis is designed to
1. store primitive timeseries data with any number of dimensions;
2. store auxiliary and meta data in columnar format or as UTF-8 encoded byte streams alongside timeseries data;
3. provide a single point of *fast* access to diverse data distributed across many files; and
4. simultaneously leverage existing technology and minimize external dependencies.

```python
import hdf5eis

with hdf5eis.File("demo.hdf5", mode="w") as demo_file:
    # Add some random multidimensional timeseries data to the demo.hdf5 file.
    first_sample_time = "2022-01-01T00:00:00Z"
    sampling_rate = 100
    demo_file.timeseries.add(
        np.random.rand(32, 16, 8, 16, 32, 1000),
        first_sample_time, 
        sampling_rate,
        tag="random"
    )
    
    # Data can be efficiently retrieved using hybrid dictionary (with regular expression parsing)
    # and array metaphors.
    start_time, end_time = "2022-01-01T00:01:00Z", "2022-01-01T00:02:00Z"
    sliced_data = demo_file.timeseries["rand*", 8:12, ..., 0, start_time: end_time]
```

# Installation

```bash
>$ pip install hdf5eis
```
