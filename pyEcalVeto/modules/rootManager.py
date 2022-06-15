import argparse
import glob
import numpy as np
import os
import ROOT as r
import sys


#########################
# Miscellaneous
#########################

# Wrapper for argparse
def parse():

    # Set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_mode', action = 'store_true', dest = 'batch_mode', default = False,
                        help = 'Whether to run in batch mode (Default: False)')
    parser.add_argument('--separate', action = 'store_true', dest = 'separate_categories', default = False,
                        help = 'Whether to separate events by fiducial category (Default: False)')
    parser.add_argument('--input_dirs', nargs = '+', action = 'store', dest = 'input_directories', default = [],
                        help = 'Space-separated list of input file directories')
    parser.add_argument('--out_dirs', '-o', nargs = '+', action = 'store', dest = 'outputs', default = [],
                        help = 'Space-separated list of output files or output file directories')
    parser.add_argument('-i', nargs = '+', action = 'store', dest = 'input_files', default = [],
                        help = 'Space-separated list of input files')
    parser.add_argument('-g', nargs = '+', action = 'store', dest = 'group_labels', default = [],
                        help = 'Space-separated list of labels for each file group')
    parser.add_argument('-s', type = int, action = 'store', dest = 'start_event', default = 0,
                        help = 'Index of first event to process')
    parser.add_argument('-m', type = int, action = 'store', dest = 'max_events', default = -1,
                        help = 'Maximum number of events to run over for each file group')
    args = parser.parse_args()

    # Enforce some conditions

    if not ((len(args.input_files) > 0 and len(args.input_directories) == 0)\
            or (len(args.input_files) == 0 and len(args.input_directories) > 0)):
        print('\n[ ERROR ] - Must provide a list of input files or input file directories!')
        sys.exit(1)

    if len(args.input_files) > 0 and len(args.input_directories) == 0:
        if len(args.input_files) != len(args.group_labels):
            print('\n[ ERROR ] - Number of input file groups does not match number of group labels!')
            sys.exit(1)
    elif len(args.input_files) == 0 and len(args.input_directories) > 0:
        if len(args.input_directories) != len(args.group_labels):
            print('\n[ ERROR ] - Number of input file groups does not match number of group labels!')
            sys.exit(1)

    if len(args.outputs) == 0:
        print('\n[ ERROR ] - Must provide a list of output files or output file directories!')
        sys.exit(1)

    if len(args.outputs) != len(args.group_labels):
        print('\n[ ERROR ] - Number of output file groups does not match number of group labels!')
        sys.exit(1)

    # Parse inputs
    if len(args.input_files) > 0 and len(args.input_directories) == 0:
        inputs = [[f] for f in args.input_files]
    elif len(args.input_files) == 0 and len(args.input_directories) > 0:
        inputs = [glob.glob('{}/*.root'.format(input_dir)) for input_dir in args.input_directories]

    parsing_dict = {'batch_mode': args.batch_mode,
                    'separate_categories': args.separate_categories,
                    'inputs': inputs,
                    'group_labels': args.group_labels,
                    'outputs': args.outputs,
                    'start_event': args.start_event,
                    'max_events': args.max_events}

    return parsing_dict

# Function to load a tree from a list of file names
def load(file_names, tree_name):

    tree = r.TChain(tree_name)
    for fn in file_names:
        tree.Add(fn)

    return tree

# Function to remove the scratch directory
def remove_scratch():

    if os.path.exists('./scratch'):
        print('\n[ INFO ] - Removing scratch directory')
        os.system('rm -rf ./scratch')


###################################
# Class to process events
###################################

class TreeProcess:

    def __init__(self, process_event, file_group = [], tree = None, tree_name = 'LDMX_Events', name_tag = 'tree_process',
                 start_event = 0, max_events = -1, print_frequency = 1000, batch_mode = False, closing_functions = []):

        print('\n[ INFO ] - Preparing tree process with name tag: {}'.format(name_tag))

        self.process_event = process_event
        self.file_group = file_group
        self.tree = tree
        self.tree_name = tree_name
        self.name_tag = name_tag
        self.start_event = start_event
        self.max_events = max_events
        self.max_event = start_event + max_events
        self.event_count = start_event
        self.print_frequency = print_frequency
        self.batch_mode = batch_mode
        self.closing_functions = closing_functions
        self.main_directory = os.getcwd()

        if not ((len(self.file_group) == 0 and not (self.tree is None))\
                or (len(self.file_group) > 0 and self.tree is None)):
            print('\n[ ERROR ] - Must provide a file group or a tree!')
            sys.exit(1)

        # Move to a scratch directory if providing a file group
        self.use_scratch = False
        if len(self.file_group) > 0 and self.tree is None:
            self.use_scratch = True

            # Make the scratch directory
            scratch_dir = '{}/scratch'.format(self.main_directory)
            print('\n[ INFO ] - Using scratch directory: {}'.format(scratch_dir))
            if not os.path.exists(scratch_dir): os.makedirs(scratch_dir)

            # Assign a number label to this process
            num = 0
            check = True
            while check:
                if os.path.exists('{}/tmp_{}'.format(scratch_dir, num)): num += 1
                else: check = False 

            # Create and move to the temporary directory
            self.temporary_directory = '{}/tmp_{}'.format(scratch_dir, num)
            print('\n[ INFO ] - Making temporary directory: {}'.format(self.temporary_directory))
            os.makedirs(self.temporary_directory)
            os.chdir(self.temporary_directory)

            # Copy the file group to the temporary directory
            print('\n[ INFO ] - Copying files to temporary directory')
            for f in self.file_group:
                os.system('cp {} .'.format(f))
            os.system('ls .')

            # Get the file names
            fns = [f.split('/')[-1] for f in self.file_group]

            # Load the files
            self.tree = load(fns, self.tree_name)

            # Move back to the main directory
            os.chdir(self.main_directory)

    # Method to add a new branch and return it for easy access
    def add_branch(self, ldmx_class, branch_name):

        if ldmx_class == 'EventHeader': branch = r.ldmx.EventHeader()
        elif ldmx_class == 'EcalVetoResult': branch = r.ldmx.EcalVetoResult()
        elif ldmx_class == 'HcalVetoResult': branch = r.ldmx.HcalVetoResult()
        elif ldmx_class == 'TriggerResult': branch = r.ldmx.TriggerResult()
        elif ldmx_class == 'SimParticle': branch = r.map(int, 'ldmx::SimParticle')()
        else: branch = r.std.vector('ldmx::{}'.format(ldmx_class))()

        self.tree.SetBranchAddress(branch_name, r.AddressOf(branch))

        return branch

    # Method to process events
    def run(self, start_event = 0, max_events = -1, print_frequency = 1000):

        print('\n[ INFO ] - Running tree process with name tag: {}'.format(self.name_tag))

        # Move to the temporary directory
        os.chdir(self.temporary_directory)

        # Reset some attributes if desired
        if start_event != self.start_event: self.start_event = start_event
        if max_events != self.max_events: self.max_events = max_events
        if print_frequency != self.print_frequency: self.print_frequency = print_frequency

        # start_event should be between 0 and tree->GetEntries()
        if self.start_event < 0: self.start_event = 0
        elif self.start_event > self.tree.GetEntries() - 1: self.start_event = self.tree.GetEntries() - 1

        # max_events should be between 1 and tree->GetEntries() - start_event
        if self.max_events < 1 or self.max_events > self.tree.GetEntries() - self.start_event:
            self.max_events = self.tree.GetEntries() - self.start_event

        self.max_event = self.start_event + self.max_events
        self.event_count = self.start_event

        # Loop to process events
        while self.event_count < self.max_event:
            self.tree.GetEntry(self.event_count)
            if self.event_count%self.print_frequency == 0:
                print('Processing event: {}'.format(self.event_count))
            self.process_event(self)
            self.event_count += 1

        # Execute any closing functions
        if len(self.closing_functions) > 0:
            for fnc in self.closing_functions:
                fnc()

        # Move back to the main directory
        os.chdir(self.main_directory)

        # Remove the temporary directory
        if self.use_scratch:
            print('\n[ INFO ] - Removing temporary directory: {}'.format(self.temporary_directory))
            os.system('rm -rf {}'.format(self.temporary_directory))


#####################################
# Class to write to a TTree
#####################################

class TreeMaker:

    def __init__(self, out_name, tree_name, branch_information = {}, out_directory = ''):

        self.out_name = out_name
        self.tree_name = tree_name
        self.branch_information = branch_information
        self.branches = {}
        self.out_directory = out_directory

        # Create output file and tree
        self.out_file = r.TFile(self.out_name, 'RECREATE')
        self.tree = r.TTree(self.tree_name, self.tree_name)

        # Add branches to the tree if already given
        for branch_name in self.branch_information:
            data_type = self.branch_information[branch_name]['dtype']
            self.branches[branch_name] = np.zeros(1, dtype = data_type)
            if str(data_type) == "<type 'float'>" or str(data_type) == "<class 'float'>":
                self.tree.Branch(branch_name, self.branches[branch_name], branch_name + '/D')
            elif str(data_type) == "<type 'int'>" or str(data_type) == "<class 'int'>":
                self.tree.Branch(branch_name, self.branches[branch_name], branch_name + '/I')

    # Method to add a new branch
    def add_branch(self, data_type, default_value, branch_name):

        self.branch_information[branch_name] = {'dtype': data_type, 'default': default_value}
        self.branches[branch_name] = np.zeros(1, dtype = data_type)
        if str(data_type) == "<type 'float'>" or str(data_type) == "<class 'float'>":
            self.tree.Branch(branch_name, self.branches[branch_name], branch_name + '/D')
        elif str(data_type) == "<type 'int'>" or str(data_type) == "<class 'int'>":
            self.tree.Branch(branch_name, self.branches[branch_name], branch_name + '/I')

    # Method to reset the branch values
    def reset_values(self):

        for branch_name in self.branches:
            self.branches[branch_name][0] = self.branch_information[branch_name]['default']

    # Method to fill the branches with new values
    def fill(self, new_values):

        for branch_name in self.branches:
            self.branches[branch_name][0] = new_values[branch_name]

        self.tree.Fill()

    # Method to write to the tree
    def write(self):

        # Save the tree and close the file
        self.out_file.Write(self.tree_name)
        self.out_file.Close()

        if self.out_directory != '':

            # Make the output directory
            if not os.path.exists(self.out_directory):
                print('\n[ INFO ] - Making output directory: {}'.format(self.out_directory))
                os.makedirs(self.out_directory)

            print('\n[ INFO ] - Copying output file to output directory')
            os.system('cp {} {}'.format(self.out_name, self.out_directory))
