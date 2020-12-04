
import numpy as np
# from scipy.spatial import distance
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import matplotlib.offsetbox as offsetbox
import matplotlib.gridspec as gridspec
# import warnings


def main(
    center, box_s, Gmax, babusiaux_filters, rv_col, name, ra, dec, mag, e_mag,
    col1, e_col1, col2, e_col2, col3, plx, pmRA, e_pmRA, pmDE, e_pmDE, radv,
        col1_n, col2_n, col3_n, Nmax=25000):
    """
    """

    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(30, 25))
    gs = gridspec.GridSpec(10, 12)

    mag_n = 'G'
    col_sizes = [[], []]
    col_names = ['RA'] + [mag_n, col1_n, col2_n, col3_n]\
        + ['Plx', 'pmRA', rv_col]
    for i, col in enumerate([ra, mag, col1, col2, col3, plx, pmRA, radv]):
        # Count valid data for each column
        col_sizes[0].append(col_names[i])
        try:
            col_sizes[1].append(col[~col.mask].size)
        except AttributeError:
            col_sizes[1].append(col.size)

    ax1 = plt.subplot(gs[0:2, 0:2])
    ax1.set_title(r"$N_{{ra}}={}$".format(ra.size), fontsize=10)
    ax1.bar(col_sizes[0], col_sizes[1])
    fig.autofmt_xdate()

    # Select at most Nmax random stars to plot
    if ra.size > Nmax:
        print("WARNING: too many stars ({}). Plotting {} random stars".format(
            ra.size, Nmax))
        ra, dec, mag, e_mag, col1, e_col1, col2, e_col2, plx, pmRA, e_pmRA,\
            pmDE, e_pmDE, radv = random_choice(
                ra, dec, mag, e_mag, col1, e_col1, col2, e_col2, plx, pmRA,
                e_pmRA, pmDE, e_pmDE, radv, Nmax)

    ax2 = plt.subplot(gs[0:2, 2:4])
    ax2.set_title("Field: [{} x {}]".format(box_s, box_s), fontsize=10)
    plt.xlabel("RA [deg]")
    plt.ylabel("DEC [deg]")
    # ra_min = max(min(ra), center[0] - 3. * rad)
    # ra_max = min(max(ra), center[0] + 3. * rad)
    # ra_rng = ra_max - ra_min
    # de_min = max(min(dec), center[1] - 3. * rad)
    # de_max = min(max(dec), center[1] + 3. * rad)
    # de_rng = de_max - de_min
    # rade_rng = min(ra_rng, de_rng) / 2.
    ax2.scatter(ra, dec, s=star_size(mag), c='k')
    # Radius
    # circle = plt.Circle(center, rad, color='red', lw=1.5, fill=False)
    # fig.gca().add_artist(circle)
    # plt.xlim(center[0] - rade_rng, center[0] + rade_rng)
    # plt.ylim(center[1] - rade_rng, center[1] + rade_rng)
    ax2.invert_xaxis()

    ax4 = plt.subplot(gs[0:2, 4:6])
    ax4.scatter(mag, e_col1, label='e' + col1_n, s=5, lw=0., alpha=0.5)
    ax4.scatter(mag, e_col2, label='e' + col2_n, s=5, lw=0., alpha=0.5)
    ax4.scatter(mag, e_mag, label='e' + mag_n, s=5, lw=0., alpha=0.5)
    # ax4.axhline(e_mmax, ls='--', c='g')
    # ax4.axhline(e_c1max, ls='--', c='r')
    # ax4.axhline(e_c2max, ls='--', c='r')
    plt.xlabel(mag_n)
    plt.ylim(-0.001, .1)
    plt.legend()

    # # distance to cluster's center
    # d_col = distance.cdist([center], np.array([ra, dec]).T)[0]
    # # Mask for photometric diagrams
    # msk = (d_col < rad) & (e_mag < e_mmax) & (e_col1 < e_c1max) &\
    #     (e_col2 < e_c2max)

    ax5 = plt.subplot(gs[2:4, 0:2])
    ax5.set_title("G < {} [mag]".format(Gmax), fontsize=10)
    plt.xlabel(col1_n)
    plt.ylabel(mag_n)
    ax5.scatter(col1, mag, s=4, lw=.1, edgecolor='w')
    ax5.invert_yaxis()
    # x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
    #     'mag', col1, mag)
    # plt.xlim(x_min_cmd, x_max_cmd)
    # plt.ylim(y_min_cmd, y_max_cmd)

    ax6 = plt.subplot(gs[2:4, 2:4])
    plt.xlabel(col2_n)
    plt.ylabel(mag_n)
    ax6.scatter(col2, mag, s=4, lw=.1, edgecolor='w')
    ax6.invert_yaxis()
    # x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
    #     'mag', col2, mag)
    # plt.xlim(x_min_cmd, x_max_cmd)
    # plt.ylim(y_min_cmd, y_max_cmd)

    ax7 = plt.subplot(gs[2:4, 4:6])
    plt.xlabel(col1_n)
    plt.ylabel(col2_n)
    ax7.scatter(col1, col2, s=4, lw=.1, edgecolor='w')
    ax7.invert_yaxis()
    # x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
    #     'col', col1, col2)
    # plt.xlim(x_min_cmd, x_max_cmd)
    # plt.ylim(y_min_cmd, y_max_cmd)

    plx_min, plx_max = 0., np.mean(plx) + 3. * np.std(plx)
    ax8 = plt.subplot(gs[4:6, 0:2])
    ax8.set_title(
        "{} < Plx [mas] < {:.3f}".format(plx_min, plx_max), fontsize=10)
    plt.xlabel("Plx [mas]")
    msk2 = (plx > plx_min) & (plx < plx_max)
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

    ax3 = plt.subplot(gs[4:6, 2:4])
    plt.xlabel("pmRA [mas/yr]")
    plt.ylabel("pmDEC [mas/yr]")
    # msk_in = d_col <= rad
    rain_mean, rain_std = pmRA.mean(), pmRA.std()
    dein_mean, dein_std = pmDE.mean(), pmDE.std()
    ra_min, ra_max = rain_mean - 3. * rain_std, rain_mean + 3. * rain_std
    de_min, de_max = dein_mean - 3. * dein_std, dein_mean + 2. * dein_std
    # ax3.set_title(
    #     "N(r<rad, |pmX|<30)={}".format(pmRA[msk_in].size), fontsize=10)
    ax3.scatter(pmRA, pmDE, c='k', marker='.', s=3, lw=.3, alpha=.5)
    # ax3.scatter(
    #     pmRA[msk_in], pmDE[msk_in], c=d_col[msk_in], s=10, alpha=.25,
    #     cmap=cm.viridis, zorder=5)
    plt.xlim(ra_min, ra_max)
    plt.ylim(de_min, de_max)
    ax3.invert_xaxis()

    # ax9 = plt.subplot(gs[4:6, 4:6])

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


def random_choice(
    ra, dec, mag, e_mag, col1, e_col1, col2, e_col2, plx, pmRA, e_pmRA, pmDE,
        e_pmDE, radv, Nmax):
    """
    """
    idxs = np.random.choice(ra.size, Nmax)

    return ra[idxs], dec[idxs], mag[idxs], e_mag[idxs], col1[idxs],\
        e_col1[idxs], col2[idxs], e_col2[idxs], plx[idxs], pmRA[idxs],\
        e_pmRA[idxs], pmDE[idxs], e_pmDE[idxs], radv[idxs]
