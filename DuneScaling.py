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
    
        # Temporary data storage for individual crests/feet/dwends
        # If we have some feet without associated crests, etc, they will
        # stay here but not go into the 'position' column
        self.feet   = pd.DataFrame(columns=['Step', 'Foot'])
        self.crests = pd.DataFrame(columns=['Step', 'Crest'])
        self.dwends = pd.DataFrame(columns=['Step', 'Dwend'])

    def __repr__(self):
        return("Dune, Length: {}".format(len(self.position)))

    def add_position(self, step, crest, foot, dwend):
        """This function adds a dune position in a step of the video.
        Crest, foot and dwend are (x,y) tuples representing position.
        Step is an integer.
        """
        pos = {'Step': step, 'Crest': crest, 'Foot': foot, 'Dwend': dwend}
        self.position = self.position.append(pos, ignore_index=True)
    
    def combine_points(self):
        
        print "Combining points!" 
        self.feet   = self.feet.sort_values('Step')
        self.crests = self.crests.sort_values('Step')
        self.dwends = self.dwends.sort_values('Step')
        
        self.feet   = self.feet.set_index('Step')
        self.crests = self.crests.set_index('Step')
        self.dwends = self.dwends.set_index('Step')
        
        all_positions = self.feet.join(self.crests)
        all_positions = all_positions.join(self.dwends)
        
        self.position = all_positions
        # Drop all steps/objects which are missing a crest, foot, or dwend
        self.position = self.position.dropna()
        
        print "Dune position df : "
        print self.position

class Parameters:
    # Structure carries parameters for running calculations on dunes
    def __init__(self, test_name):
        self.test_name = test_name
        # DEFAULT VALUES - SHOULD BE OVERWRITTEN IN SETUP
        # Convert steps of tracker time to seconds of real time
        self.steps2s = float('nan')
        # Will we save?
        params.saving_on = False
        # Using a test data set?
        params.istest    = False
        params.fname     = "NaN"        
    def __repr__(self):
        return("Parameter vector for test")

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

def read_data_Madonna(fname):
    """ Read the csv file with all the dune point data that Madonna read,
    data is stored in format of circles with three points per dune"""
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
            
        # KK added to make closer to chelsea version march 24
        for name in dunes.keys():
            dune = dunes[name]
            # Make sure dune is sorted by step
            dune.position.set_index('Step')
            dune.position.sort_index
        return dunes
    
def read_data_Chelsea(fname):
    """Read the csv file with data that Chelsea on 24/jan/2018 took,
    data is stored as individual points
    This will only work if dunes are named 'DXNNNN t x y'  where
    X is a one-character dune type, and NNNN is an integer of any length""" 
    with open(fname, 'rU') as datafile:
        
        print "Reading file : " + fname
        
        # Set up the empty data frame:
        # one for dunes without all three points (foot, crest, downwind)
        incomplete_dunes = dict()
        # one for dunes with everything.
        complete_dunes = dict()
        pointtype  = 'n/a'
        dunenumber = -9999
        positions_on_current_point = []
        times_on_current_point     = []
        
        datareader = csv.reader(datafile, dialect='excel')
        for line in islice(datareader, 0, None):
            currentline = line[:]
            print "reading line: " + str(currentline)
            firstitem = currentline[0]
            # If line is a single string starting with D, it's a name line
            # for a new dune
            if firstitem[0] in ['D', 'R']:
                # First save the last dune
                if firstitem[0] == 'D':
                    incomplete_dunes = add_point_from_chelsea_csv(incomplete_dunes, 
                                               dunenumber, pointtype,
                                               times_on_current_point, 
                                               positions_on_current_point)
                elif firstitem[1] == 'R':
                    print "This is a ripple! We're not saving ripples yet."
            # Now read the info from the new dune
            if firstitem[0] == 'D':
                # This is a dune!
                pointtype  = firstitem[1]
                dunenumber = int(firstitem[2:-6])
                positions_on_current_point = []
                times_on_current_point     = []
            elif firstitem[0] == 'R':
                # This is a ripple!
                pointtype  = firstitem[1]
                dunenumber = int(firstitem[2:-6])
                positions_on_current_point = []
                times_on_current_point     = []
            # If the line is not a single string, it's some data for the current dune
            else:
                time = currentline[0]
                xpos = currentline[1]
                ypos = currentline[2]
                times_on_current_point.append(float(time))
                positions_on_current_point.append( (float(xpos), float(ypos)) )
                
                
    # Final position needs to be added since loop won't return to top
    incomplete_dunes = add_point_from_chelsea_csv(incomplete_dunes, 
                               dunenumber, pointtype,
                               times_on_current_point, 
                               positions_on_current_point)
        # Once all points are added, complete the dunes by attaching
        # feet with crests, dwends, and sorting out the step numbers
    for dunenumber in incomplete_dunes.keys():
        if dunenumber != -9999:
            dune = incomplete_dunes[dunenumber]
            dune.combine_points()
            complete_dunes[dunenumber] = dune
    
    return complete_dunes
                
    

def add_point_from_chelsea_csv(incomplete_dunes, dunenumber, 
                                               pointtype,
                                               times_on_current_point, 
                                               positions_on_current_point):
    """Add one point from one dunes to the dune dictionary: dunes with less
    than three points are stored in incomplete dunes, dunes with all three 
    points (foot, crest, end) are stored in complete_dunes"""
    #Has some other data from this dune been read?
    if dunenumber not in incomplete_dunes.keys():
        # Create a new dune item for this object
        incomplete_dunes[dunenumber] = Dune(dunenumber)
    # Now dune is stored in incomplete_dunes[dunenumber]
    current_dune = incomplete_dunes[dunenumber]
    
    # Add this point to the dune
    if pointtype == 'F':
        for i in range(len(times_on_current_point)):
            current_dune.feet = current_dune.feet.append(
                    {'Step': times_on_current_point[i],
                     'Foot': positions_on_current_point[i]}, ignore_index=True)
    elif pointtype == 'T':
        for i in range(len(times_on_current_point)):
            current_dune.crests = current_dune.crests.append(
                    {'Step': times_on_current_point[i],
                     'Crest': positions_on_current_point[i]}, ignore_index=True)
    elif pointtype == 'W':
        for i in range(len(times_on_current_point)):
            current_dune.dwends = current_dune.dwends.append(
                    {'Step': times_on_current_point[i],
                     'Dwend': positions_on_current_point[i]}, ignore_index=True)
    else:
        print "Trying to add a point with no type: " + pointtype
    
    incomplete_dunes[dunenumber] = current_dune
    return incomplete_dunes
    
def get_n(ser, n):
    """Get the nth value of items from dataframe series and return them in an array"""
    lst = []
    for row in ser:
        lst.append(row[n])
    arr = np.array(lst)
    return arr

def turn_pos_into_xy(pos_df, key):
    """ Take in a dune position dataframe, and a point type e.g. 'Foot', 'Crest',
    and return the x-value (x_or_y = 0) or y-value (x_or_y = 1)
    of that point type"""
    xvec = np.zeros(len(pos_df))
    yvec = np.zeros(len(pos_df))
    for i, item in enumerate(pos_df[key]):
        print 'ITEM: ' + str(item) + ('printed in turn_pos_into_xy')
        (x,y) = item
        xvec[i] = x
        yvec[i] = y
    return (xvec, yvec)
        


# 4. Get heights, widths, slopes, velocities, etc.
def get_dims(dunes, params):
    """Add dimension information to """
    # parameters is an object containing whatever parameters we will need here 
    for name in dunes.keys():
        dune = dunes[name]
        # Get t and each x and y series
        t = dune.position.index.values*params.steps2s
#        xf = get_n(dune.position['Foot'], 0)
#        yf = get_n(dune.position['Foot'], 1)
#        xc = get_n(dune.position['Crest'], 0)
#        yc = get_n(dune.position['Crest'], 1)
#        xd = get_n(dune.position['Dwend'], 0)
#        yd = get_n(dune.position['Dwend'], 1)
        
        # KK version of madonna get_n code
        (xf, yf) = turn_pos_into_xy(dune.position, 'Foot')
        (xc, yc) = turn_pos_into_xy(dune.position, 'Crest')
        (xd, yd) = turn_pos_into_xy(dune.position, 'Dwend')
        
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
def makeplot(var_x, var_y, label, dunes, params):
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
    if params.saving_on:
        plt.savefig(params.test_name+var_x+var_y)
    # Close the plot
    plt.close()


def plot_all(dunes, params):
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
            makeplot(var_x, var_y, label, dunes, params)
            n += 1
        i += 1
    print('All {} plots have been made.'.format(str(n)))

def get_test_parameters(params):
    """ Process data from a given video - store all the associated methods
    and filenames together here
    test_name       : Chelsea or Madonna
    istest          : True or False (use test dataset or full)"""
    # Structure to store test parameters
    test_name = params.test_name
    if test_name in ['240118', 'Chelsea']:
        # Method for reading data
        read_data      = read_data_Chelsea
        # Parameters for timing, distance etc for this test
        params.steps2s = 20
        # File with input data
        if params.istest:  params.fname   = 'testdata_012418.csv'
        else:              params.fname   = 'data012418.csv'
    elif test_name in ['170122', 'Madonna']:
        read_data      = read_data_Madonna
        params.steps2s = 10
        if params.istest:  params.fname   = '170122AA_test.csv'
        else:              params.fname   = '170122AA_full.csv'
    else:
        print "That is not a valid testname."
        print "Valid names: 'Chelsea', 'Madonna', '240118', '170122'."
    return (params, read_data)

def main(params):
    # Read data from file
    (params, read_data) = get_test_parameters(params)
    dunes = read_data(params.fname)
    # Add dimensions and velocities to dune structures
    dunes = get_dims(dunes, params)
    # Plot everything possible
    plot_all(dunes, params)
    return dunes

# Which of the available tests would you like to run?
test_name = 'Chelsea'
# Define parameters for the test.
# Some parameters are set as a function of test name.
params = Parameters(test_name)
# True/False flag : run on test data set or whole data set?
params.istest     = True
# Save the output figures?
params.saving_on  = False

dunes = main(params)
