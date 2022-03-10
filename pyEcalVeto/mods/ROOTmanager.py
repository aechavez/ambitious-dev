import argparse
import glob
import numpy as np
import os
import ROOT as r
import sys


###############################################
# Class to help process event samples
###############################################

class TreeProcess:

    def __init__(self, process_event, file_group = [], tree = None, tree_name = None, name_tag = '',
                 start_event = 0, max_events = -1, print_frequency = 1000, batch_mode = False, closing_functions = None):

        print('\n[ INFO ] - Preparing tree with name tag: {}'.format(name_tag))

        self.process_event = process_event
        self.files = file_group
        self.tree = tree
        self.tree_name = tree_name
        self.name_tag = name_tag
        self.start_event = start_event
        self.max_events = max_events
        self.print_frequency = print_frequency
        self.batch_mode = batch_mode
        self.closing_functions = closing_functions
        self.cwd = os.getcwd()

        # Move to a scratch directory if providing a file group
        self.mvd = False
        if self.tree is None:
            self.mvd = True

            # Create the scratch directory if it doesn't already exist
            scratch_dir = self.cwd + '/scratch'
            print('\n[ INFO ] - Using scratch path: {}'.format(scratch_dir))
            if not os.path.exists(scratch_dir):
                os.makedirs(scratch_dir)

            # Assign a temporary number to this process
            num = 0
            check = True
            while check:
                if os.path.exists('{}/tmp_{}'.format(scratch_dir, num)):
                    num += 1
                else:
                    check = False 

            ### LEFT OFF HERE ###
            # Create and move into the temporary directory
            if self.batch:
                self.tmp_dir='%s/%s' % (scratch_dir, os.environ['LSB_JOBID'])
            else:
                self.tmp_dir = '%s/%s' % (scratch_dir, 'tmp_'+str(num))
            if not os.path.exists(self.tmp_dir):
                print( 'Creating tmp directory %s' % self.tmp_dir )
            os.makedirs(self.tmp_dir)
            os.chdir(self.tmp_dir)
    
            # Copy input files to the tmp directory
            print( 'Copying input files into tmp directory' )
            for rfilename in self.group_files:
                os.system("cp %s ." % rfilename )
            os.system("ls .")
    
            # Just get the file names without the full path
            tmpfiles = [f.split('/')[-1] for f in self.group_files]
    
            # Load'em
            if self.tree_name != None:
                self.tree = load(tmpfiles, self.tree_name)
            else:
                self.tree = load(tmpfiles)

            # Move back to cwd in case running multiple procs
            os.chdir(self.cwd)

    def addBranch(self, ldmx_class, branch_name):

        # Add a new branch to read from

        if self.tree == None:
            sys.exit('Set tree')

        if ldmx_class == 'EventHeader': branch = r.ldmx.EventHeader()
        elif ldmx_class == 'EcalVetoResult': branch = r.ldmx.EcalVetoResult()
        elif ldmx_class == 'HcalVetoResult': branch = r.ldmx.HcalVetoResult()
        elif ldmx_class == 'TriggerResult': branch = r.ldmx.TriggerResult()
        elif ldmx_class == 'SimParticle': branch = r.map(int, 'ldmx::'+ldmx_class)()
        else: branch = r.std.vector('ldmx::'+ldmx_class)()

        self.tree.SetBranchAddress(branch_name,r.AddressOf(branch))

        return branch
 
    def run(self, strEvent=0, maxEvents=-1, pfreq=1000):
   
        # Process events

        if strEvent != 0: self.strEvent = strEvent
        if maxEvents != -1: self.maxEvents = maxEvents
        if self.maxEvents == -1 or self.strEvent + self.maxEvents > self.tree.GetEntries():
            self.maxEvents = self.tree.GetEntries() - self.strEvent
        maxEvent = self.strEvent + self.maxEvents
        if pfreq != 1000: self.pfreq = pfreq

        self.event_count = self.strEvent
        while self.event_count < maxEvent:
            self.tree.GetEntry(self.event_count)
            if self.event_count%self.pfreq == 0:
                print('Processing Event: %s'%(self.event_count))
            self.event_process(self)
            self.event_count += 1

        # Execute any closing function(s) (might impliment *args, **kwargs later)
        if self.extrafs != None:
            for extraf in self.extrafs:
                extraf()

        # Move back to cwd in case running multiple procs
        os.chdir(self.cwd)

        # Remove tmp directory if created in move
        if self.mvd:
            print( 'Removing tmp directory %s' % self.tmp_dir )
            os.system('rm -rf %s' % self.tmp_dir)


##########################################
# Class to help write to a TTree
##########################################

class TreeMaker:

    def __init__(self, outfile, tree_name, branches_info = {}, outdir=''):

        self.outfile = outfile
        self.tree_name = tree_name
        self.branches_info = branches_info
        self.branches = {}
        self.outdir = outdir

        # Create output file and tree
        self.tfout = r.TFile(self.outfile,"RECREATE")
        self.tree = r.TTree(tree_name, tree_name)

        # Set up new tree branches if given branches_info
        if len(branches_info) != 0:
            for branch_name in branches_info:
                self.addBranch(self.branches_info[branch_name]['rtype'],\
                               self.branches_info[branch_name]['default'],\
                               branch_name)

    def addBranch(self, rtype, default_value, branch_name):

        # Add a new branch to write to

        self.branches_info[branch_name] = {'rtype': rtype, 'default': default_value}
        self.branches[branch_name] = np.zeros(1, dtype=rtype)
        if str(rtype) == "<type 'float'>" or str(rtype) == "<class 'float'>":
            self.tree.Branch(branch_name, self.branches[branch_name], branch_name + "/D")
        elif str(rtype) == "<type 'int'>" or str(rtype) == "<class 'int'>":
            self.tree.Branch(branch_name, self.branches[branch_name], branch_name + "/I")
        # ^ probably use cases based on rtype to change the /D if needed?

    def resetFeats(self):

        # Reset variables to defaults for new event
        # Return because feats['feat'] looks nicer than self.tfMaker.feats['feat']

        feats = {}
        for branch_name in self.branches_info:
            feats[branch_name] = self.branches_info[branch_name]['default']

        return feats

    def fillEvent(self, feats):

        # Fill the tree with new feature values

        for feat in feats:
            self.branches[feat][0] = feats[feat]
        self.tree.Fill()

    def wq(self):

        # Save the tree and close the file
        self.tfout.Write(self.tree_name)
        self.tfout.Close()

        if self.outdir != '':
            if not os.path.exists(self.outdir):
                print( 'Creating %s' % (self.outdir) )
                os.makedirs(self.outdir)

            print( 'cp %s %s' % (self.outfile,self.outdir) )
            os.system('cp %s %s' % (self.outfile,self.outdir))


###################################
# Functions
###################################

def parse(nolist = False):

    import glob
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store_true', dest='batch', default=False,
            help='Run in batch mode [Default: False]')
    parser.add_argument('--sep', action='store_true', dest='separate', default = False,
            help='separate events into different files [Default: False]')
    parser.add_argument('-i', nargs='+', action='store', dest='infiles', default=[],
            help='input file(s)')
    parser.add_argument('--indirs', nargs='+', action='store', dest='indirs', default=[],
            help='Director(y/ies) of input files')
    parser.add_argument('-g','-groupls', nargs='+', action='store', dest='group_labels',
            default='', help='Human readable sample labels e.g. for legends')
    parser.add_argument('-o','--out', nargs='+', action='store', dest='out', default=[],
            help='output files or director(y/ies) of output files')
            # if inputting directories, it's best to make a system
            # for naming files in main() of main script 
    parser.add_argument('--notlist', action='store_true', dest='nolist',
            help="return things without lists (to make things neater for 1 sample runs")
    parser.add_argument('-s','--start', type=int, action='store', dest='startEvent',
            default=0, help='event to start at')
    parser.add_argument('-m','--max', type=int, action='store', dest='maxEvents',
            default=-1, help='max events to run over for EACH group')
    args = parser.parse_args()

    # Input
    if args.infiles != []:
        inlist = [[f] for f in args.infiles] # Makes general loading easier
        if nolist or args.nolist == True:
            inlist = inlist[0]
    elif args.indirs != []:
        inlist = [glob.glob(indir + '/*.root') for indir in args.indirs]
        if nolist or args.nolist == True:
            inlist = inlist[0]
    else:
        sys.exit('provide input')

    # Output
    if args.out != []:
        outlist = args.out
        if nolist or args.nolist == True:
            outlist = outlist[0]
    else:
        sys.exit('provide output')
    
    pdict = {
            'batch': args.batch,
            'separate': args.separate,
            'inlist': inlist,
            'groupls': args.group_labels,
            'outlist': outlist,
            'startEvent': args.startEvent,
            'maxEvents': args.maxEvents
            }

    return pdict

# Load a tree from a group of input files
def load(group,treeName='LDMX_Events'):

    # Load a group of files into a readable tree

    tree = r.TChain(treeName)
    for f in group:
        tree.Add(f)

    return tree

# Remove scratch dir
def rmScratch():
    if os.path.exists('./scratch'):
        print( '\nRemoving scratch directory' )
        os.system('rm -rf ./scratch')

