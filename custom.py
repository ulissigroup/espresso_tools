'''
Contains a function that returns various Quantum Espresso settings given the
host that you're on.
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

import os
import socket
import getpass
import json


# Find and open the JSON of the Lennard-Jones parameters
__MODULE_LOCATION = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__PARAMETERS_LOCATION = os.path.join(__MODULE_LOCATION, 'lj_params.json')
with open(__PARAMETERS_LOCATION, 'r') as file_handle:
    LJ_PARAMETERS = json.load(file_handle)


def hpc_settings():
    '''
    Returns a dictionary with certain information that Quantum Espresso needs
    to run.

    Returns:
        settings    A dictionary with the following keys/values:
                        qe_executable   The full path of the pw.x executable
                        psp_path        The path of the folder containing all
                                        of the pseudopotentials
                        nodes           An integer indicating the maximum
                                        number of nodes you can run on. Note
                                        that ALL jobs on this HPC will be
                                        submitting with this number of nodes.
                        cores_per_node  The number of cores each node has.
    '''
    node_name = socket.gethostname()

    if 'quartz' in node_name:
        settings = {'manager': 'slurm',
                    'qe_executable': ('/usr/workspace/woodgrp/catalysis/Codes'
                                      '/q-e-modified-pprism_beef/bin/pw.x'),
                    'psp_path': '/usr/workspace/woodgrp/catalysis/pseudo',
                    'scratch_dir': '/p/lscratchh/%s/gaspy/' % getpass.getuser(),
                    'nodes': 4,
                    'cores_per_node': 36,
                    'wall_time': 20}  # in hours

    elif 'lassen' in node_name:
        settings = {'manager': 'lsf',
                    'qe_executable': ('/usr/workspace/woodgrp/catalysis/Codes'
                                      '/Lassen/q-e-modified-pprism_beef_xl/bin/pw.x'),
                    'psp_path': '/usr/workspace/woodgrp/catalysis/pseudo',
                    'scratch_dir': '/p/gpfs1/%s/gaspy/' % getpass.getuser(),
                    'nodes': 1,
                    'cores_per_node': 44,
                    'wall_time': 12}  # in hours

    else:
        raise ValueError('espresso_tools does not recognize the %s node. Please '
                         'add the appropriate HPC settings into '
                         'espresso_tools.custom.hpc_settings.'
                         % node_name)

    # Make the scratch directory if it doesn't already exist
    try:
        os.mkdir(settings['scratch_dir'])
    except FileExistsError:
        pass

    return settings
