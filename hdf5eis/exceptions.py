class TableFormatError(Exception):
    '''
    Improperly formatted column.
    '''
    pass

class TSIndexError(Exception):
    '''
    Mismatch between __TS_INDEX table and timeseries data.
    '''
    pass

class UTF8FormatError(Exception):
    '''
    Invalide UTF-8 string.
    '''

class VersionError(Exception):
    '''
    File version is incompatible with code version.
    '''
    pass