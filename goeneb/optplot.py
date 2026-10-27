import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import argparse

from helper import bettersplit
from file_sys_io import qread

colors = [np.array([0.4136032, 0.2038992, 0.463532, 1]),    # purple
          np.array([0.7068016, 0.6019496, 0.731766, 1]),    # light purple
          np.array([0.1275680, 0.5669490, 0.550556, 1]),    # turquoise
          np.array([0.5637840, 0.7834745, 0.775278, 1]),    # light turquoise
          np.array([0.1742414, 0.5206197, 0.185548, 1]),    # green
          np.array([0.5871207, 0.7603098, 0.592774, 1])]    # light green

tokcal = 627.5094740631

def main():
    # initialize the argparser
    parser = argparse.ArgumentParser(prog='optplot',description='Plotting the collected data by the NEB program.')
    parser.add_argument('-f', '--filename',
                        help='Enter the file that should be visualized. No file opens the user input.')
    parser.add_argument('-s', '--savefile', 
                        help='Enter the path where the file should be saved. No file means displaying the image directly.')
    args = parser.parse_args()


    # check if the script was called with a path as argument
    if args.filename is not None:
        filepath = args.filename

        # save the image
        if args.savefile is not None:
            savepath =  args.savefile
            save_optplots(filepath, savepath)
            return
        # show the image
        else: 
            show_optplots(filepath)

    # otherwise, enter loop of requesting filepath input,
    while True:
        user_input = input('Enter filepath of log file to display. ' +
                           'Enter nothing to terminate program.\n')
        if user_input == '':
            break
        filepath = Path(user_input)
        show_optplots(filepath)
        print('\n')

# ----------------------------------------------------------------------------------
# Functions for drawing and saving

def optplots(filepath):
    """
    Function to plot the NEB log file in 4 subfigures.
    """
    # Load file
    linetokens = load_optlog(filepath)

    # Get all values from the file
    sv_names, sv_array = get_singleval_array(linetokens)
    enprofiles_array = get_energy_profile_array(linetokens)
    gradprofiles_array = get_gradnorm_profile_array(linetokens)
    orth_gradprofiles_array = get_orthgradnorm_profile_array(linetokens)
    raw_projcoords_array = get_raw_projcoords_array(linetokens)
    rxncoords_array = get_rxncoords_array(linetokens)

    # Create the figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    [ax1, ax2, ax3, ax4] = axes.flatten()

    # Do the plotting
    do_convsig_plot(ax1, sv_names, sv_array)
    do_profile_plot(ax2, enprofiles_array, rxncoords_array)
    do_projpath_plot(ax3, raw_projcoords_array)
    do_gnorm_plot(ax4, gradprofiles_array, orth_gradprofiles_array)

    plt.tight_layout()

def show_optplots(filepath):
    """
    Function to display the Plot directly.
    """
    optplots(filepath)
    plt.show()

def save_optplots(filepath, savepath):
    """
    Function to save the Plot to the desired path.
    """
    optplots(filepath)
    plt.savefig(savepath) 

# -----------------------------------------------------------------------------------
# Functions for reading in data

def load_optlog(filepath):
    """
    Read the log file and return the better split token lines.
    """
    try:
        lines = qread(filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        raise
    except Exception as e:
        print(f"Unable to read file {filepath}: {e}")
        raise
    token_lines = [bettersplit(line, delim=',\n') for line in lines]
    return token_lines

def get_singleval_array(token_lines):
    """
    Get the single values like RMSF and AbsF.
    """
    field_name_lists = []
    field_val_lists = []

    for line in token_lines:
        field_names, field_vals = find_singlevals(line)
        field_name_lists.append(field_names)
        field_val_lists.append(field_vals)

    # find the total number of columns
    fname_set = set()
    for fname_list in field_name_lists:
        fname_set.update(fname_list)
    fname_set = sorted(list(fname_set))

    val_array = np.zeros((len(field_name_lists), len(fname_set)))
    val_array[:, :] = np.nan

    for i in range(len(val_array)):
        for j in range(len(fname_set)):
            name = fname_set[j]

            if name not in field_name_lists[i]:
                continue
            else:
                val_ind = field_name_lists[i].index(name)
                val_array[i, j] = field_val_lists[i][val_ind]

    return fname_set, val_array

# -------------------------------------------------------------------------------
# Functions for the specific plots

def do_convsig_plot(ax, sv_names, sv_array, min=False):
    """
    Function for doing the convergence signal plot. 
    Needs:
    - ax: the axis of the plot
    - sv_names: names of the single values 
    - sv_array: the values of the single values, same order
    """
    # depending on mode and length of single values
    # the colors should be differently ordered

    # plot all single values
    for i in range(len(sv_names)):
        values = sv_array[:, i].flatten()
        ls = '--' if 'CI' in sv_names[i] else '-'
        this_color = colors[0] if 'RMS' in sv_names[i] else colors[2]
        this_color = this_color if 'o' in sv_names[i] or 'CI' in sv_names[i] else lighten_color(this_color, 0.45)
        ax.plot(values, color=this_color, ls=ls)

    # make pretty
    ax.set_title('Convergence measures')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Value [$E_\mathrm{h}/\mathrm{\AA}$]')
    ax.grid(True, which='both')
    ax.set_yscale('log')

    legend_labels = [name.replace('_o', r'$^\perp$') for name in sv_names]
    ax.legend(legend_labels)

    return ax


def do_profile_plot(ax,
                    enprofiles_array,
                    rxncoords_array,
                    color=(0,0.1,0.6),
                    lastiter_color=colors[3]):
    """
    Do the energy profile plot of the path, previous iterations get
    more transparent.
    """
    for i in range(len(enprofiles_array)):
        energies = enprofiles_array[i]
        energies = (energies - energies[0]) * tokcal
        etas = rxncoords_array[i]

        # plot the last iteration in a different color
        if i == len(enprofiles_array) - 1:
            ax.plot(etas, energies, 'o-', color=lastiter_color)

        # all previous iterations
        else:
            base_col = list(colors[0])
            alpha = alpha_scalefunc(i, len(enprofiles_array) - 1)
            base_col[3] = alpha  # Set new alpha
            ax.plot(etas, energies, color=tuple(base_col))

    # make pretty
    ax.set_title('Energy profile')
    ax.set_xlabel('Approx. Reaction Coordinate')
    ax.set_ylabel('Energy [kcal/mol]')
    ax.grid(True, which='both')

    return ax

def alpha_scalefunc(index, maxindex, starta=0.2, stopa=1.0):
    """A Function that gives an alpha value for decreasing opacity."""
    fac = (maxindex - index) / maxindex
    alpha = stopa * (1.0 - fac) + starta * fac
    return alpha

def do_projpath_plot(ax, raw_projcoords_array, color=colors[1],
                     startcolor=colors[3], endcolor=colors[2]):
    """
    Do the projected path plot. Needs the projected coordinates.
    """
    nimgs = int((raw_projcoords_array.shape[1] / 2))
    proj = raw_projcoords_array.reshape(-1, nimgs, 2)
    shifted_projcoords_array = (proj - proj[0,0,:]).reshape(-1, nimgs*2)

    start_img_xs = []
    end_img_xs = []
    start_img_ys = []
    end_img_ys = []

    # Reorder the list to be plottable, and plot the images' traces
    for i in range(nimgs):
        img_xs = shifted_projcoords_array[:, i*2]
        img_ys = shifted_projcoords_array[:, i*2+1]

        start_img_xs.append(img_xs[0])
        end_img_xs.append(img_xs[-1])
        start_img_ys.append(img_ys[0])
        end_img_ys.append(img_ys[-1])

        plot_path(ax, img_xs, img_ys, color=color)

    # Plot start and end path
    plot_path(ax, start_img_xs, start_img_ys, color=startcolor, markerstyle='o', 
              linestyle='-', alpha=0.5)
    plot_path(ax, end_img_xs, end_img_ys, color=endcolor, markerstyle='o', linestyle='-')

    # make pretty
    ax.set_title('2D projection')
    ax.set_xlabel(r'$\alpha$ [$\mathrm{\AA}$]')
    ax.set_ylabel(r'$\beta$ [$\mathrm{\AA}$]')
    ax.grid(True, which='both')

    return ax

def plot_path(ax, xvals, yvals, color, markerstyle=None, linestyle='', alpha=1):
    """
    Helper function to plot any path.
    """
    if markerstyle is None:
        ax.plot(xvals, yvals, color=color, alpha=alpha)
    else:
        ax.plot(xvals, yvals, color=color,
                marker=markerstyle,
                linestyle=linestyle, alpha=alpha)

def do_gnorm_plot(ax, gradprofiles_array, orth_gradprofiles_array=None, normal_gnorm=True, min=False):
    """
    Do the gradient plot.
    """
    # Normal gradient
    if (type(orth_gradprofiles_array) != np.ndarray) or (normal_gnorm == True) or min:
        values = np.array(gradprofiles_array).T

        if not min:
            ax.imshow(values, aspect='auto', cmap='viridis', norm=mcolors.LogNorm())
        else:
            ax.imshow(np.abs(values), aspect='auto', cmap='viridis')

    # Gradient with orthogonal gradient
    else:
        par_gradprofiles_array = np.sqrt(np.array(gradprofiles_array)**2 - 
                                         np.array(orth_gradprofiles_array)**2)
        values = (par_gradprofiles_array**2 / np.array(gradprofiles_array)**2).T
        ax.imshow(1-values, aspect='auto', cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=1))

    # make pretty
    if not min:
        ax.set_ylabel('Image index')
    else:
        ax.set_ylabel('Gradient dimension')
    ax.set_title('Gradient per iteration')
    ax.set_xlabel('Iteration')

    return ax

# --------------------------------------------------------------------------
# Functions for getting the right values

def find_singlevals(token_line):
    """Find all single values."""
    # find where the multival fields start, the first will be 'energies'
    try:
        if 'energies' in token_line:
            end_index = token_line.index('energies')
        elif 'gradient' in token_line:
            end_index = token_line.index('gradient')
        else:
            raise ValueError("Line does not contain 'energies' or 'gradient'. " + 
                            "Something went wrong with the NEB Logger.")
    except Exception as e:
        print(f"Error parsing line {token_line}: {e}")
        raise

    # skip the first two tokens, which contain the number of images
    sval_tokens = token_line[2:end_index]

    field_names = sval_tokens[::2]
    field_vals  = [safe_float(x) for x in sval_tokens[1::2]]

    return field_names, field_vals


def find_nimgs(token_line):
    return int(token_line[1])

def get_energy_profile_array(token_lines):
    # assumes all iterations will have the same number of images
    nimgs = find_nimgs(token_lines[0])
    energies_array = np.zeros((len(token_lines), nimgs))

    for i in range(len(token_lines)):
        token_line = token_lines[i]
        start_index = token_line.index('energies') + 1
        stop_index = start_index + nimgs
        energies = np.array([safe_float(item) for item in
                            token_line[start_index : stop_index]])
        energies_array[i] = energies

    return energies_array

def get_rxncoords_array(token_lines):
    # assumes all iterations will have the same number of images
    nimgs = find_nimgs(token_lines[0])
    rxncoords_array = np.zeros((len(token_lines), nimgs))

    for i in range(len(token_lines)):
        token_line = token_lines[i]
        start_index = token_line.index('approx_rxn_crds') + 1
        stop_index = start_index + nimgs
        rxncoords = np.array([safe_float(item) for item in
                              token_line[start_index : stop_index]])
        rxncoords_array[i] = rxncoords

    return rxncoords_array

def get_gradnorm_profile_array(token_lines):
    # assumes all iterations will have the same number of images
    nimgs = find_nimgs(token_lines[0])
    gradnorms_array = np.zeros((len(token_lines), nimgs - 2))

    for i in range(len(token_lines)):
        token_line = token_lines[i]
        start_index = token_line.index('gradnorms') + 1
        stop_index = start_index + nimgs - 2

        gradnorms = np.array([safe_float(item) for item in
                              token_line[start_index : stop_index]])
        gradnorms_array[i] = gradnorms

    return gradnorms_array

def get_orthgradnorm_profile_array(token_lines):
    # assumes all iterations will have the same number of images
    if any(token == 'orth_gradnorms' for token in token_lines[0]):
        nimgs = find_nimgs(token_lines[0])
        orth_gradnorms_array = np.zeros((len(token_lines), nimgs - 2))

        for i in range(len(token_lines)):
            token_line = token_lines[i]
            start_index = token_line.index('orth_gradnorms') + 1
            stop_index = start_index + nimgs - 2
            orth_gradnorms = np.array([safe_float(item) for item in
                                       token_line[start_index : stop_index]])
            orth_gradnorms_array[i] = orth_gradnorms
        return orth_gradnorms_array
    else:
        return None

def get_raw_projcoords_array(token_lines):
    # assumes all iterations will have the same number of images
    nimgs = find_nimgs(token_lines[0])
    projcoords_array = np.zeros((len(token_lines), nimgs * 2))
    for i in range(len(token_lines)):
        token_line = token_lines[i]
        start_index = token_line.index('projcoords') + 1
        stop_index = start_index + nimgs * 2
        projcoords = np.array([safe_float(item) for item in
                               token_line[start_index : stop_index]])
        projcoords_array[i] = projcoords
    return projcoords_array

# ---------------------------------------------------------------------------
# miscellaneous helper functions

def safe_float(val):
    try:
        if val in (None, 'None'):
            return np.nan
        return float(val)
    except Exception:
        return np.nan

def lighten_color(color, blend_factor):
    # Blend the color with white using the provided blend factor
    r, g, b, a = color
    white = np.array([1, 1, 1, a])
    lightened_color = color * (1 - blend_factor) + white * blend_factor
    # Ensure the values don't exceed 1
    lightened_color = np.clip(lightened_color, 0, 1)
    return lightened_color

if __name__ == '__main__':
    main()
