
from astropy.io import ascii


def main(name, data, e_col1, e_col2, e_col3, col1_n, col2_n, col3_n):
    """
    """
    # # Replace data.
    # data['Gmag'], data['e_Gmag'] = mag, e_mag
    # data[col1_n], data[col2_n] = col1, col2

    # Add errors for the colors
    data['e_' + col1_n], data['e_' + col2_n], data['e_' + col3_n] =\
        e_col1, e_col2, e_col3

    # Write in 'input/' folder
    ascii.write(data, 'input/' + name + ".dat", overwrite=True)
