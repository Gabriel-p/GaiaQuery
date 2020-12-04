
import os
import ast
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from uncertainties import ufloat
from uncertainties import unumpy as unp
from modules import getData, writeOut, makePlot
from modules._version import __version__


def main():
    """
    Explore data downloaded via the astroquery package.

    Vizier Gaia DR2 column names:
    http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=I/345/gaia2

    About the negative parallaxes in Gaia DR2 data:
    https://astronomy.stackexchange.com/q/26250/354
    https://astronomy.stackexchange.com/q/26071/354

    """
    print("\n*******************")
    print(" GaiaQuery {}".format(__version__))
    print("*******************")

    read, DR, Gmax, babusiaux_filters, clusters = readInput()

    if DR == '2':
        # Gaia DR2
        cat = 'I/345/gaia2'
        rv_col = 'RV'
    elif DR == '3':
        # Gaia EDR3
        cat = 'I/350/gaiaedr3'
        rv_col = 'RVDR2'
    else:
        raise ValueError("DR value '{}' is not supported".format(DR))

    # Define color names
    col1_n, col2_n, col3_n = 'BP-RP', 'BP-G', 'G-RP'

    for folders in ['input', 'output']:
        if not os.path.exists(folders):
            os.makedirs(folders)

    for clust in clusters:
        center, box_s = (clust['cent_ra'], clust['cent_dec']), clust['box_s']

        data = getData.main(
            cat, Gmax, clust['name'], center, box_s, read)

        N_old = len(data)
        print("{} data read, {} sources".format(clust['name'], N_old))

        if babusiaux_filters:
            data = babusiaux_filt(data)
            print("Filters applied, {:.1f}% of data lost".format(
                100. - (len(data) * 100.) / N_old))

        mag, e_mag = data['Gmag'], data['e_Gmag']
        col1, col2, col3 = data[col1_n], data[col2_n], data[col3_n]
        if read is False:
            print("Obtaining magnitudes/colors and their uncertainties")
            e_col1, e_col2, e_col3 = uncertMags(
                DR, data, col1_n, col2_n, col3_n)

            print("Write output file in input/ folder")
            writeOut.main(
                clust['name'], data, e_col1, e_col2, e_col3, col1_n, col2_n,
                col3_n)
        else:
            e_col1, e_col2 = data['e_' + col1_n], data['e_' + col2_n]

        print("Plotting")
        makePlot.main(
            center, box_s, Gmax, babusiaux_filters,
            rv_col, clust['name'], data['RA_ICRS'],
            data['DE_ICRS'], mag, e_mag, col1, e_col1, col2, e_col2, col3,
            data['Plx'], data['pmRA'], data['e_pmRA'], data['pmDE'],
            data['e_pmDE'], data[rv_col], col1_n, col2_n, col3_n)

    print("\nEnd")


def babusiaux_filt(data):
    """
    Babusiaux et al. (2018) HRD filters.
    """
    m1 = (data['RPlx'] > 0.)  # 10.
    m2 = (data['RFG'] > 50.)
    m3 = (data['RFBP'] > 20.)  # 20.
    m4 = (data['RFRP'] > 20.)  # 20.
    m5 = (data['E_BR_RP_'] > 1. + 0.015 * (data['BPmag'] - data['RPmag']) ** 2)
    # m6 = (data['Gmag'] < 1.e6)
    m6 = (data['E_BR_RP_'] < 1.3 + 0.06 * (data['BPmag'] - data['RPmag']) ** 2)
    m7 = (data['Nper'] > 8)
    m8 = (data['chi2AL'] / (data['NgAL'] - 5.) < 1.44 * np.clip(
          np.exp(-.4 * (data['Gmag'] - 19.5)), a_min=None, a_max=1.))
    mask = m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8
    for i, m in enumerate([m1, m2, m3, m4, m5, m6, m7, m8]):
        print("  m" + str(i + 1) + " removes {} sources".format(
            len(data) - m.data.sum()))

    return data[mask]


def uncertMags(DR, data, col1_n, col2_n, col3_n):
    """
    # Gaia DR2 zero points:

    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/
    chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html#Ch5.T2

    The G magnitude error:
        unp.std_devs(mag_d['G'])
    is equivalent to:
        np.sqrt((1.0857*(data['e_FG']/data['FG']))**2 + .0018**2)
    as defined in Eq 5.26 of the link above.

    These values are larger than the 'e_Gmag' column given by Vizier by up to
    ~0.002 for the brightest stars, and go to zero for the faintest. I don't
    really know why. I asked Vizier and their answer was:

    > The e_Gmag uncertainties in VizieR were added by the CDS and,
    > apparently, do not take into account the Vegamag corrections; we will
    > continue the investigation.
    >
    > In the meantime, ignoring the e_Gmag in VizieR and using the formula
    > given by Gaia DR2 seems to be the right solution."

    # Gaia EDR3 zero points:

    https://www.cosmos.esa.int/web/gaia/edr3-passbands

    """
    # Zero points for the G,BP,RP magnitudes.
    if DR == '2':
        # Updated October 2017
        Zp_G = ufloat(25.6914396869, 0.0011309370)
        Zp_BP = ufloat(25.3488107670, 0.0004899854)
        Zp_RP = ufloat(24.7626744847, 0.0035071711)
    elif DR == '3':
        Zp_G = ufloat(25.6873668671, 0.0027553202)
        Zp_BP = ufloat(25.3385422158, 0.0027901700)
        Zp_RP = ufloat(24.7478955012, 0.0037793818)

    # Fluxes
    I_G = unp.uarray(data['FG'], data['e_FG'])
    I_BP = unp.uarray(data['FBP'], data['e_FBP'])
    I_RP = unp.uarray(data['FRP'], data['e_FRP'])

    # Magnitudes
    mag_d = {
        'G': Zp_G + -2.5 * unp.log10(I_G),
        'BP': Zp_BP + -2.5 * unp.log10(I_BP),
        'RP': Zp_RP + -2.5 * unp.log10(I_RP)}

    # import matplotlib.pyplot as plt
    # Gmag_new = unp.nominal_values(mag_d['RP'])
    # plt.scatter(data['Gmag'], data['RPmag'] - Gmag_new, alpha=.5)
    # plt.ylabel("Gmag_CDS - Gmag_here")
    # plt.xlabel("Gmag_CDS")
    # plt.show()

    col11, col12 = col1_n.split('-')
    col21, col22 = col2_n.split('-')
    col31, col32 = col3_n.split('-')
    # Colors
    col1 = mag_d[col11] - mag_d[col12]
    col2 = mag_d[col21] - mag_d[col22]
    col3 = mag_d[col31] - mag_d[col32]

    # Uncertainties
    e_col1, e_col2, e_col3 = unp.std_devs(col1), unp.std_devs(col2),\
        unp.std_devs(col3)

    return e_col1, e_col2, e_col3


def readInput():
    """
    Read 'cluster_in.dat' data file.
    """
    with open("clusters_in.dat", 'r') as f:
        i = 0
        for line in f:
            if i != 1:
                if not line.startswith('#') and line != '\n':
                    params = line.split()
                    i = 1
            else:
                data = ascii.read(f.read())

    read, DR, Gmax, babusiaux_filters = ast.literal_eval(params[0]),\
        params[1], params[2], ast.literal_eval(params[3])

    clusters = Table(data, names=('name', 'cent_ra', 'cent_dec', 'box_s'))

    return read, DR, Gmax, babusiaux_filters, clusters


if __name__ == '__main__':
    # To see available catalogs:
    # catalog_list = Vizier.find_catalogs('Pan-STARRS')
    # catalogs = Vizier.get_catalogs(catalog_list.keys())
    # print(catalogs)

    main()
