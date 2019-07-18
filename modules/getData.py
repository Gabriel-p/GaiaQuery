
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.units as u
from astroquery.vizier import Vizier


def main(cat, name, center, box_s, read):
    """
    Download data using astroquery.
    """
    if read:
        print("\nReading {} data file.".format(name))
        data = ascii.read('input/' + name + '.dat')
    else:
        print("\nDownloading data for {}, from {}.".format(name, cat))
        # Unlimited rows, all columns
        v = Vizier(row_limit=-1, columns=['all'])

        result = v.query_region(coord.SkyCoord(
            ra=center[0], dec=center[1], unit=(u.deg, u.deg), frame='icrs'),
            width=box_s, catalog=[cat])
        data = result[cat]

    return data
