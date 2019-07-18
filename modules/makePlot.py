
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.offsetbox as offsetbox
import matplotlib.gridspec as gridspec
import warnings


def main(
    name, center, rad, e_mmax, e_c1max, e_c2max, plx_min, plx_max,
        ra, dec, mag, e_mag, col1, e_col1, col2, e_col2,
        plx, pmRA, e_pmRA, pmDE, e_pmDE, radv, col1_n, col2_n,
        babusiaux_filters):
    """
    """
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(30, 25))
    gs = gridspec.GridSpec(10, 12)

    mag_n = 'G'
    col_sizes = [[], []]
    col_names = ['RA'] + [mag_n, col1_n, col2_n] + ['Plx', 'pmRA', 'RV']
    for i, col in enumerate([ra, mag, col1, col2, plx, pmRA, radv]):
        # Count valid data for each column
        col_sizes[0].append(col_names[i])
        col_sizes[1].append(col[~col.mask].size)

    ax1 = plt.subplot(gs[0:2, 0:2])
    ax1.bar(col_sizes[0], col_sizes[1])
    fig.autofmt_xdate()

    ax2 = plt.subplot(gs[0:2, 2:4])
    ax2.set_title(r"$N_{{T}}={},\, rad={:.3f}\,[deg]$".format(
        ra.size, rad), fontsize=8)
    plt.xlabel("RA [deg]")
    plt.ylabel("DEC [deg]")
    ra_min = max(min(ra), center[0] - 3. * rad)
    ra_max = min(max(ra), center[0] + 3. * rad)
    de_min = max(min(dec), center[1] - 3. * rad)
    de_max = min(max(dec), center[1] + 3. * rad)
    msk = (ra > ra_min) & (ra < ra_max) & (dec > de_min) & (dec < de_max)
    ax2.scatter(ra[msk], dec[msk], s=star_size(mag[msk]), c='k')
    # Radius
    circle = plt.Circle(center, rad, color='red', lw=1.5, fill=False)
    fig.gca().add_artist(circle)
    plt.xlim(ra_min, ra_max)
    plt.ylim(de_min, de_max)
    ax2.invert_xaxis()

    ax4 = plt.subplot(gs[0:2, 4:6])
    ax4.scatter(mag, e_col1, label='e' + col1_n, s=5, lw=0., alpha=0.5)
    ax4.scatter(mag, e_col2, label='e' + col2_n, s=5, lw=0., alpha=0.5)
    ax4.scatter(mag, e_mag, label='e' + mag_n, s=5, lw=0., alpha=0.5)
    ax4.axhline(e_mmax, ls='--', c='g')
    ax4.axhline(e_c1max, ls='--', c='r')
    ax4.axhline(e_c2max, ls='--', c='r')
    plt.xlabel(mag_n)
    plt.ylim(-0.01, .5)
    plt.legend()

    # distance to cluster's center
    d_col = distance.cdist([center], np.array([ra, dec]).T)[0]

    # Mask for photometric diagrams
    msk = (d_col < rad) & (e_mag < e_mmax) & (e_col1 < e_c1max) &\
        (e_col2 < e_c2max)

    ax5 = plt.subplot(gs[2:4, 0:2])
    ax5.set_title(
        "N(r<{}, e_mag<{}, e_c1<{}, e_c2<{})={}".format(
            rad, e_mmax, e_c1max, e_c2max, msk.data.sum()), fontsize=8)
    plt.xlabel(col1_n)
    plt.ylabel(mag_n)
    ax5.scatter(col1[msk], mag[msk], s=4, lw=.1, edgecolor='w')
    # no masked elements
    msk4 = (~col1[msk].mask) & (~mag[msk].mask)
    x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
        'mag', col1[msk][msk4], mag[msk][msk4])
    plt.xlim(x_min_cmd, x_max_cmd)
    plt.ylim(y_min_cmd, y_max_cmd)

    ax6 = plt.subplot(gs[2:4, 2:4])
    plt.xlabel(col2_n)
    plt.ylabel(mag_n)
    ax6.scatter(col2[msk], mag[msk], s=4, lw=.1, edgecolor='w')
    # no masked elements
    msk4 = (~col2[msk].mask) & (~mag[msk].mask)
    x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
        'mag', col2[msk][msk4], mag[msk][msk4])
    plt.xlim(x_min_cmd, x_max_cmd)
    plt.ylim(y_min_cmd, y_max_cmd)

    ax9 = plt.subplot(gs[2:4, 4:6])
    plt.xlabel(col1_n)
    plt.ylabel(col2_n)
    ax9.scatter(col1[msk], col2[msk], s=4, lw=.1, edgecolor='w')
    # no masked elements
    msk4 = (~col1[msk].mask) & (~col2[msk].mask)
    x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
        'col', col1[msk][msk4], col2[msk][msk4])
    plt.xlim(x_min_cmd, x_max_cmd)
    plt.ylim(y_min_cmd, y_max_cmd)

    if plx_min < 0.:
        print("Setting plx_min=0.")
        plx_min = 0.
    ax8 = plt.subplot(gs[4:6, 0:2])
    ax8.set_title("{} < Plx [mas] < {}".format(plx_min, plx_max), fontsize=8)
    plt.xlabel("Plx [mas]")
    msk2 = (d_col < rad) & (e_mag < e_mmax) &\
        (e_col1 < e_c1max) & (e_col2 < e_c2max) &\
        (plx > plx_min) & (plx < plx_max)
    y, x, _ = ax8.hist(plx[msk2], bins=75)
    p_max_mas = (.5 * (x[y.argmax()] + x[y.argmax() + 1]))
    d_max_pc = 1000. / p_max_mas
    ax8.axvline(p_max_mas, ls='--', c='r')
    plx_lt_zero = 100. * plx[plx < 0.].size / plx.size
    ob = offsetbox.AnchoredText(
        "Plx_max={:.0f} [pc] ({:.3f} [mas])\nPlx<0: {:.1f}%".format(
            d_max_pc, p_max_mas, plx_lt_zero),
        pad=0.2, loc=1, prop=dict(size=9))
    ob.patch.set(alpha=0.85)
    ax8.add_artist(ob)
    plt.xlim(0., 3.)

    ax7 = plt.subplot(gs[4:6, 2:4])
    ax7.set_title("N(r<, e_G<, e_XP<, <Plx<)={}".format(
        msk2.data.sum()), fontsize=8)
    plt.xlabel("RA [deg]")
    plt.ylabel("DEC [deg]")
    cmap = cm.viridis_r
    norm = Normalize(vmin=0., vmax=p_max_mas)
    ax7.scatter(ra[msk2], dec[msk2], s=4, c=cmap(norm(plx[msk2])))
    ax7.invert_xaxis()
    im = plt.scatter(ra[msk2], dec[msk2], s=0, c=plx[msk2], cmap=cmap)
    cbar_ax = fig.add_axes([0.313, 0.53, 0.005, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=5)
    plt.clim(0., 2. * p_max_mas)

    ax3 = plt.subplot(gs[4:6, 4:6])
    plt.xlabel("pmRA [mas/yr]")
    plt.ylabel("pmDEC [mas/yr]")
    msk3 = (pmRA > -30) & (pmDE > -30) & (pmRA < 30) &\
        (pmDE < 30) & (d_col < rad)
    pmRA_f, pmDE_f, epmRA_f, epmDE_f = pmRA[msk3], pmDE[msk3],\
        e_pmRA[msk3], e_pmDE[msk3]
    ax3.set_title("N(r<rad, |pmX|<30)={}".format(pmRA_f.size), fontsize=8)
    cmap = cm.viridis
    norm = Normalize(vmin=d_col[msk3].min(), vmax=rad)
    ax3.errorbar(
        pmRA_f, pmDE_f, yerr=epmDE_f, xerr=epmRA_f, fmt='none', elinewidth=.35,
        ecolor=cmap(norm(d_col[msk3])))
    im = plt.scatter(pmRA_f, pmDE_f, s=0, c=d_col[msk3], cmap=cmap)
    cbar_ax = fig.add_axes([0.48, 0.53, 0.005, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=5)

    # Ignore warning issued by colorbar.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()

    if babusiaux_filters:
        out_name = 'output/gaia_babusiaux_' + name + '.png'
    else:
        out_name = 'output/gaia_' + name + '.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight')


def star_size(mag, N=None, min_m=None):
    '''
    Convert magnitudes into intensities and define sizes of stars in
    finding chart.
    '''
    # Scale factor.
    if N is None:
        N = len(mag)
    if min_m is None:
        min_m = min(mag)
    factor = 500. * (1 - 1 / (1 + 150 / N ** 0.85))
    return 0.1 + factor * 10 ** ((np.array(mag) - min_m) / -2.5)


def diag_limits(yaxis, phot_x, phot_y):
    '''
    Define plot limits for *all* photometric diagrams.
    '''
    # TODO deprecated
    # min_x, max_x, min_y, max_y = kde_limits(phot_x, phot_y)

    x_median, x_std = np.median(phot_x), 1.5 * np.std(phot_x)
    min_x, max_x = x_median - x_std, x_median + x_std
    y_median, y_std = np.median(phot_y), np.std(phot_y)
    min_y, max_y = y_median - y_std, y_median + y_std

    # Define diagram limits.
    x_min_cmd, x_max_cmd = min_x - 1.25, max_x + 1.25
    y_min_cmd = max_y + 1.25
    # If photometric axis y is a magnitude, make sure the brightest star
    # is always plotted.
    if yaxis == 'mag':
        y_max_cmd = min(phot_y) - 1.
    else:
        y_max_cmd = min_y - 1.

    return x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd
