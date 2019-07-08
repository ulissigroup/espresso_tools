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
        settings    A dictionary whose keys are 'qe_executable' and 'psp_path'.
                    The values are strings showing where the Quantum Espresso
                    executable file and the folder of pseudopotentials are,
                    respectively.
    '''
    if 'quartz' in host_name:
        settings = {'qe_executable': ('/usr/workspace/woodgrp/catalysis/Codes'
                                      '/q-e-modified-pprism_beef/bin/pw.x'),
                    'psp_path': '/usr/workspace/woodgrp/catalysis/pseudo',
                    'nodes': 1,
                    'cores_per_node': 36}

    elif 'lassen' in host_name:
        settings = {'qe_executable': ('/usr/workspace/woodgrp/catalysis/Codes'
                                      '/q-e-modified-pprism_beef/bin/pw.x'),
                    'psp_path': '/usr/workspace/woodgrp/catalysis/pseudo',
                    'nodes': 4,
                    'cores_per_node': 44}

    return settings
