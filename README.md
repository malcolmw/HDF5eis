# HDF5eis (read H-D-F-Size)
A solution for handling big, multidimensional timeseries data from environmental sensors in HPC applications.

```python
import hdf5eis

with hdf5eis.File("demo.hdf5", mode="w") as demo_file:
    # Generate some random multidimensional timeseries data.
    random_data = np.random.rand(32, 16, 8, 16, 64, 1000)
    
    # Add it to the demo.hdf5 file.
    demo_file.timeseries.add(
        random_data,
        "2022-01-01T00:00:00Z", # This is the time of the first sample.
        100,                    # This is the temporal sampling rate (samples per second).
        tag="random"            # This is a tag to differentiate collections of data.
    )
    
    # Data can be efficiently retrieved using hybrid dictionary-like and array
    # slicing syntax.
    start_time, end_time = "2022-01-01T00:01:00Z", "2022-01-01T00:02:00Z"
    sliced_data = demo_file.timeseries[
        "random", # This is the tag of the requested data. Regular expressions permitted.
        8:12,     # This is a slice along the first storage axis.
        ...,      # This is a complete slice along all intermediate storage axes.
        0,        # This is a slice along the second to last storage axis.
        start_time: end_time # This is a slice along the time axis.
    ]
```
