columns = dataf.columns.values
dtypes = dataf.dtypes.values
dtypes = [DTYPE_MAPPING[dt] for dt in  dtypes]
dtype = np.dtype(list(zip(columns, dtypes)))

data = np.array(list(map(lambda row: tuple(row[1]), dataf.iterrows())), dtype=dtype)
