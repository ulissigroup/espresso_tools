'''
Contains a function that returns various Quantum Espresso settings given the
host that you're on.
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']


def hpc_settings(host_name):
    '''
    Given the name of the host, returns a dictionary with certain information
    that Quantum Espresso needs to run.

    Arg:
        host_name   A string indicating which host you're using.
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
    if 'quartz' in host_name:
        settings = {'qe_executable': ('/usr/workspace/woodgrp/catalysis/Codes'
                                      '/q-e-modified-pprism_beef/bin/pw.x'),
                    'psp_path': '/usr/workspace/woodgrp/catalysis/pseudo',
                    'nodes': 4,
                    'cores_per_node': 36}

    elif 'lassen' in host_name:
        settings = {'qe_executable': ('/usr/workspace/woodgrp/catalysis/Codes'
                                      '/q-e-modified-pprism_beef/bin/pw.x'),
                    'psp_path': '/usr/workspace/woodgrp/catalysis/pseudo',
                    'nodes': 4,
                    'cores_per_node': 44}

    else:
        raise ValueError('espresso_tools does not recognize the %s host. Please '
                         'add the appropriate HPC settings into '
                         'espresso_tools.custom.hpc_settings.'
                         % host_name)

    return settings
