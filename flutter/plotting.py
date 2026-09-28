"""Shared figure style. Call set_style() at the top of every figure notebook."""
import matplotlib


def set_style():
    rc = matplotlib.rcParams
    rc['font.serif'] = ['Times'] + rc['font.serif']
    rc['font.size'] = 6
    rc['text.usetex'] = False
    rc['ps.usedistiller'] = 'xpdf'
    rc['font.family'] = 'serif'
    rc['font.weight'] = 'normal'
    rc['mathtext.fontset'] = 'cm'
