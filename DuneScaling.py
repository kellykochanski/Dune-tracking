from matplotlib import pyplot as plt
import csv
import numpy as np
import pandas as pd
from itertools import islice

"""
Designed to track crest, foot, and dwend points of dunes from 120122AA time
lapse footage, where dunes are seen in profile and there is no scale known.
Data is input in a .csv file with three lines per dune in a step. Dune class
is borrowed from Kelly's code.
- MY 2/8/2018
"""

# Input file stored in same folder.
fname = '170122AA_full.csv' 


class Dune:
    """
    This is a class which holds information about a snow dune.
    Each dune is described by three points: one on the crest, one on the foot,
    and one on the dwend.
    These points are defined for each point in time as the dune moves
    across Niwot Ridge.

    To use:
    #Create a dune:
    >> new_dune = Dune(name)
    # Put points in the dune using command
    >> new_dune.add_position(step, (x1,y1), (x2,y2), (x3,y3))
    # Collect points into dataframe
    >> new_dune.finalize()
    # Do some analysis,
    #  e.g. plot position of dune crest against step number:
    >> plt.plot(new_dune['Step'], new_dune['Top'])
    """
    def __init__(self, name):
        """
        This function lists all the properties that a snow dune should have.
        Every python class needs an __init__ function.
        Call this as:
        $new_dune = Dune(name)
        """

        # Give every dune a name. Useful for reference.
        self.name = name

        # Create an empty data frame - this will be a time series
        # containing the dune's position at several points in time.
        self.position = pd.DataFrame(columns=[
                'Step', 'Crest', 'Foot', 'Dwend'])

    def __repr__(self):
        return("Dune, Length: {}".format(len(self.position)))

    def add_position(self, step, crest, foot, dwend):
        """This function adds a dune position in a step of the video.
        Crest, foot and dwend are (x,y) tuples representing position.
        Step is an integer.
        """
        pos = {'Step': step, 'Crest': crest, 'Foot': foot, 'Dwend': dwend}
        self.position = self.position.append(pos, ignore_index=True)


def find_points(group):
    """Finds the dwend, crest, and foot points of a group of three lines from
    the input csv. Returns a list of tuples in the order dwend, crest, foot."""
    line0 = group[0]
    line1 = group[1]
    line2 = group[2]
    # Pull the points
    p0 = (float(line0[2]), float(line0[3]))
    p1 = (float(line1[2]), float(line1[3]))
    p2 = (float(line2[2]), float(line2[3]))
    pts = [p0, p1, p2]
    xi = [pts[0][0], pts[1][0], pts[2][0]]
    minx = np.where(xi == np.min(xi))[0][0]
    dwend = pts.pop(minx)
    yi = [pts[0][1], pts[1][1]]
    maxy = np.where(yi == np.max(yi))[0][0]
    crest = pts.pop(maxy)
    foot = pts.pop()
    return (crest, foot, dwend)


def add_and_finalize(dunes, name, step, crest, foot, dwend):
    """Add points provided to the dune of given name or make a new dune with
    those points"""
    if name in dunes.keys():
        # Add points and finalize
        target_dune = dunes[name]
        target_dune.add_position(step, crest, foot, dwend)
        dunes[name] = target_dune
    else:
        # Make a new dune, add point, and finalize
        new_dune = Dune(name)
        new_dune.add_position(step, crest, foot, dwend)
        dunes[name] = new_dune
    return dunes


def clear_group(group, current):
    """Empty group list and get new name and step"""
    group = []
    group.append(current)
    name = current[1]
    step = float(current[0])
    return (group, name, step)


def read_data(fname):
    """ Read the csv file with all the dune point data"""
    # 1. Open and read the file
    with open(fname, 'r') as datafile:
        # 0. Set up the empty data structure to be filled
        dunes = dict()
        group = []
        name = 'Start'
        step = -1
        datareader = csv.reader(datafile, dialect='excel')
        for line in islice(datareader, 1, None):
            current = line[:]
            cname = current[1]
            cstep = float(current[0])
            if (name == cname) and (step == cstep):
                group.append(current)
            else:   # 2. For each set of three lines,
                # pull dwend, crest and foot points
                if len(group) == 3:
                    (crest, foot, dwend) = find_points(group)
                    # 3. Assign points to appropriate dune and time
                    dunes = add_and_finalize(
                            dunes, name, step, crest, foot, dwend)
                else:
                    # Print something about how many lines and what dune/time
                    print('Irregular dune {} at step {} has {} lines.'.format(
                            name, step, len(group)))
                # Clear group, add current, and update name and step
                (group, name, step) = clear_group(group, current)
        if len(group) == 3:     # Check end-of-loop group
            (crest, foot, dwend) = find_points(group)
            dunes = add_and_finalize(dunes, name, step, crest, foot, dwend)
        return dunes


def get_n(ser, n):
    """Get the nth value of items from array ser and return them in an array"""
    lst = []
    for row in ser:
        lst.append(row[n])
    arr = np.array(lst)
    return arr


# 4. Get heights, widths, slopes, velocities, etc.
def get_dims(dunes):
    """Add dimension information to """
    for name in dunes.keys():
        dune = dunes[name]
        # Make sure dune is sorted by step
        dune.position.set_index('Step')
        dune.position.sort_index
        # Get t and each x and y series
        t = dune.position['Step']*10   # Units of seconds
        xf = get_n(dune.position['Foot'], 0)
        yf = get_n(dune.position['Foot'], 1)
        xc = get_n(dune.position['Crest'], 0)
        yc = get_n(dune.position['Crest'], 1)
        xd = get_n(dune.position['Dwend'], 0)
        yd = get_n(dune.position['Dwend'], 1)
        dt = np.diff(t)
        dx = np.diff(xc)
        v = abs(dx/dt)
        dune.position['Height'] = yc-yf
        dune.position['Width'] = xc-xd
        dune.position['Slope'] = (yc-yd)/(xc-xd)
        dune.position['Sbase'] = (yf-yd)/(xf-xd)
        dune.position['X'] = xc
        dune.position['Y'] = yf
        dune.position['Velocity'] = np.insert(v, 0, np.nan)
        dunes[name] = dune
    return dunes


# 5. Plot everything
def makeplot(var_x, var_y, label, dunes):
    """Make the actual plots from all dunes for var_x and var_y"""
    # Initialize plot
    plt.figure()
    # Iterate through dunes to fill in points
    for name in dunes.keys():
        df = dunes[name].position
        x = df[var_x]
        y = df[var_y]
        color = df['Y']/270
        # Plot the points
#        plt.plot(x, y, '.-')    # For plotting lines for each dune with points
        plt.scatter(x, y, c=color, cmap='RdYlGn', vmin=0, vmax=1)
    # Add labels
    plt.xlabel(label[var_x])
    plt.ylabel(label[var_y])
    plt.title('{} vs. {}'.format(var_x, var_y))
    plt.show()
    # Save the plot
    plt.savefig(var_x+var_y)
    # Close the plot
    plt.close()


def plot_all(dunes):
    """Plot all combinations of variables"""
    variables = ['X', 'Y', 'Height', 'Width', 'Slope', 'Sbase', 'Velocity']
    label = {'X': 'X, Crest position', 'Y': 'Y, Foot position',
             'Height': 'Height, Crest-Foot', 'Width': 'Width, Crest-Dwend',
             'Slope': 'Slope, Dwend to Crest', 'Sbase': 'Slope, Dwend to Foot',
             'Velocity': 'Velocity, per second'}
    i = 0
    n = 0
    for var in variables:
        var_x = var
        # Plot var vs. all variables after var
        ind = range((i+1), 7)
        if len(ind) == 0:
            return
        for index in ind:
            var_y = variables[index]
            makeplot(var_x, var_y, label, dunes)
            n += 1
        i += 1
    print('All {} plots have been made.'.format(str(n)))


def main(fname):
    # Read data from file
    dunes = read_data(fname)
    # Add dimensions and velocities to dune structures
    dunes = get_dims(dunes)
    # Plot everything possible
    plot_all(dunes)
    return dunes


dunes = main(fname)
