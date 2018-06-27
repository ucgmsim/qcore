"""
Functions and classes to load data that doesn't belong elsewhere.
"""

def load_im_file(csv_file, all_psa = False):

    # process column names
    use_cols = []
    col_names = []
    with open(csv_file, 'r') as f:
        raw_cols = f.readline().strip().split(',')
    for i, c in enumerate(raw_cols):
        # filter out pSA that aren't round numbers, duplicates
        if c not in names and \
                all_psa or (c.startswith('pSA_') and len(c) < 12):
            use_cols.append(i)
            col_names.append(c)

    # create numpy datatype
    dtype = [(n, np.float32) for n in col_names]
    # first 2 columns are actually strings
    dtype[0] = ('station', '|S7')
    dtype[1] = ('component', '|S4')

    # load all at once
    d = np.loadtxt(csv_file, dtype = dtype, delimiter = ',', \
                   skiprows = 1, usecols = tuple(use_cols))
