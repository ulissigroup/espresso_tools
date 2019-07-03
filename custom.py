""" Standardization of pseudopotentials in self-contained dictionaries.

Includes PBE and PBEsol quantum espresso (.UPF) pseudopotentials from
 - GBRV high-throughput Vanderbilt USP          [ https://www.physics.rutgers.edu/gbrv/#Li ]
 - Standard Solid State Pseudopotentials (SSSP) [ http://materialscloud.org/sssp/ ]
 - Various user-defined ones

Defaults to GBRV as USP.
"""
import os


def custom_usersettings(parameter):
    """ Get the path of where the standardized pseudopotential directories and files are stored.
    :return: the full path housing these directories.
    """
    homepath = '/usr/workspace/woodgrp/catalysis/Codes/q-e-modified-pprism_beef/bin'
    d_custom_usersettings = {
        'pseudopath': '/usr/workspace/woodgrp/catalysis/pseudo',
        'executablepath_quartz': os.path.join(
            homepath,
            'bin/espresso/quartz/espresso-6.0/cp.x'),
        'resub_executablepath': os.path.join(
            homepath,
            'bin/pylib/espressotools/insert_job_dependency.py'),
    }
    if parameter in d_custom_usersettings:
        customval = d_custom_usersettings[parameter]
    else:
        customval = None
    return customval
