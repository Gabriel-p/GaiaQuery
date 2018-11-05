
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
from modules import getData, writeOut, makePlot

# GaiaDR2 = 'I/345/gaia2'
# Hardcoded to work on Gaia DR2 data.
cat = 'I/345/gaia2'


def main(
    clusters, col1_n='BP-RP', col2_n='BP-G', read=True,
        babusiaux_filters=False):
    """
    Explore data downloaded via the astroquery package.

    Vizier Gaia DR2 column names:
    http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=I/345/gaia2

    About the negative parallaxes in Gaia DR2 data:
    https://astronomy.stackexchange.com/q/26250/354
    https://astronomy.stackexchange.com/q/26071/354

    Vizier Pan-STARRS1 column names:
    http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=II/349&-to=3

    """
    for clust in clusters:

        data = getData.main(
            cat, clust['name'], clust['center'], clust['box_s'], read)

        N_old = len(data)
        print("{} data read, {} sources".format(clust['name'], N_old))

        if babusiaux_filters:
            data = babusiaux_filt(data)
            print("Filters applied, {:.1f}% of data lost".format(
                100. - (len(data) * 100.) / N_old))

        print("Obtaining magnitudes/colors and their uncertainties")
        mag, e_mag, col1, e_col1, col2, e_col2 = uncertMags(
            data, col1_n, col2_n)

        print("Write output file in input/ folder")
        writeOut.main(
            clust['name'], data, mag, e_mag, col1, e_col1, col2, e_col2,
            col1_n, col2_n)

        print("Plotting")
        makePlot.main(
            clust['name'], clust['center'], clust['clust_rad'],
            clust['e_mmax'], clust['e_c1max'], clust['e_c2max'],
            clust['plx_min'], clust['plx_max'],
            data['RA_ICRS'], data['DE_ICRS'],
            mag, e_mag, col1, e_col1, col2, e_col2,
            data['Plx'], data['pmRA'], data['e_pmRA'], data['pmDE'],
            data['e_pmDE'], data['RV'], col1_n, col2_n,
            babusiaux_filters)


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


def uncertMags(data, col1_n, col2_n):
    """
    Gaia DR2 zero points:

    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/
    chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html#Ch5.T2

    """

    # Zero points for the G,BP,RP magnitudes.
    Zp_G = ufloat(25.6884, 0.0018)
    Zp_BP, Zp_RP = ufloat(25.3514, 0.0014), ufloat(24.7619, 0.0019)

    # Factor that converts the instrumental flux to calibrated flux.
    c_G, c_BP, c_RP = 10**(Zp_G / -2.5), 10**(Zp_BP / -2.5), 10**(Zp_RP / -2.5)

    # Fluxes
    I_G = unp.uarray(data['FG'], data['e_FG'])
    I_BP = unp.uarray(data['FBP'], data['e_FBP'])
    I_RP = unp.uarray(data['FRP'], data['e_FRP'])

    # Magnitudes
    mag_d = {
        'G': -2.5 * unp.log10(c_G * I_G), 'BP': -2.5 * unp.log10(c_BP * I_BP),
        'RP': -2.5 * unp.log10(c_RP * I_RP)}

    col11, col12 = col1_n.split('-')
    col21, col22 = col2_n.split('-')
    # Colors
    col1 = mag_d[col11] - mag_d[col12]
    col2 = mag_d[col21] - mag_d[col22]

    return unp.nominal_values(mag_d['G']), unp.std_devs(mag_d['G']),\
        unp.nominal_values(col1), unp.std_devs(col1),\
        unp.nominal_values(col2), unp.std_devs(col2),


if __name__ == '__main__':
    # To see available catalogs:
    # catalog_list = Vizier.find_catalogs('Pan-STARRS')
    # catalogs = Vizier.get_catalogs(catalog_list.keys())
    # print(catalogs)

    clusters = [
        {'name': 'TEST', 'center': (95.05864, -73.41445), 'box_s': ".1deg",
         'clust_rad': .05, 'e_mmax': 0.1, 'e_c1max': .1, 'e_c2max': 0.1,
         'plx_min': .0, 'plx_max': 5.},
        # {'name': 'GAIA1', 'center': (101.47, -16.75), 'box_s': "1deg",
        #  'clust_rad': .15, 'e_mmax': 0.05, 'e_c1max': .1, 'e_c2max': 0.1,
        #  'plx_min': 0., 'plx_max': 5.},
        # {'name': 'GAIA2', 'center': (28.12, 53.04), 'box_s': "1deg",
        #  'clust_rad': .07, 'e_mmax': 0.1, 'e_c1max': .2, 'e_c2max': 0.2,
        #  'plx_min': .0, 'plx_max': 5.},
        # {'name': 'GAIA3', 'center': (95.05864, -73.41445), 'box_s': "1deg",
        #  'clust_rad': .05, 'e_mmax': 0.1, 'e_c1max': .1, 'e_c2max': 0.1,
        #  'plx_min': .0, 'plx_max': 5.},
        # {'name': 'GAIA4', 'center': (56.36793, 52.89297), 'box_s': "1deg",
        #  'clust_rad': .059, 'e_mmax': 0.1, 'e_c1max': .1, 'e_c2max': 0.1,
        #  'plx_min': .0, 'plx_max': 5.},
        # {'name': 'GAIA5', 'center': (110.79779, -29.71947), 'box_s': "1deg",
        #  'clust_rad': .05, 'e_mmax': 0.1, 'e_c1max': .1, 'e_c2max': 0.1,
        #  'plx_min': .0, 'plx_max': 5.},
        # {'name': 'GAIA6', 'center': (122.09798, -23.70648), 'box_s': "1deg",
        #  'clust_rad': .06, 'e_mmax': 0.1, 'e_c1max': .1, 'e_c2max': 0.1,
        #  'plx_min': .0, 'plx_max': 5.},
        # {'name': 'GAIA7', 'center': (84.69075, 30.49822), 'box_s': "1deg",
        #  'clust_rad': .035, 'e_mmax': 0.1, 'e_c1max': .1, 'e_c2max': 0.1,
        #  'plx_min': .0, 'plx_max': 5.}
    ]

    main(clusters, read=False)
