
from astropy.io import ascii


def main(name, data, mag, e_mag, col1, e_col1, col2, e_col2, col1_n, col2_n):
    """
    """
    # Replace data.
    data['Gmag'], data['e_Gmag'] = mag, e_mag
    data[col1_n], data[col2_n] = col1, col2

    # Add data
    data['e_' + col1_n], data['e_' + col2_n] = e_col1, e_col2

    # Write in 'input/' folder
    ascii.write(data, 'input/' + name + ".dat", overwrite=True)
