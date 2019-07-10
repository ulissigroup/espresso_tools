'''
Contains a function that returns various Quantum Espresso settings given the
host that you're on.
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

import socket


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
                    'nodes': 4,
                    'cores_per_node': 36,
                    'wall_time': 24}  # in hours

    elif 'lassen' in node_name:
        settings = {'manager': 'lsf',
                    'qe_executable': ('/usr/workspace/woodgrp/catalysis/Codes'
                                      '/Lassen/q-e-modified-pprism_beef_xl/bin/pw.x'),
                    'psp_path': '/usr/workspace/woodgrp/catalysis/pseudo',
                    'nodes': 4,
                    'cores_per_node': 44,
                    'wall_time': 12}  # in hours

    else:
        raise ValueError('espresso_tools does not recognize the %s node. Please '
                         'add the appropriate HPC settings into '
                         'espresso_tools.custom.hpc_settings.'
                         % node_name)

    return settings
