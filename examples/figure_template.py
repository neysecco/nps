'''
This is a set of commands to set a standard figure layout.
This should be called in the main code with:

exec(open("<PATH TO THIS FILE>").read())
'''

import matplotlib as mpl

# Set default options (these will work for all plots of the script)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}' #for \text command
mpl.rcParams['font.size'] = 23
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['lines.linewidth'] = 2.0

# Get cycle of colors for label text
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]