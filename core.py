'''
This module contains light wrapping tools between LLNL's espressotools and
CMU's GASpy. Requires Python 3
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

import subprocess
import socket
from datetime import datetime
import ase.io
from .cpespresso_v3 import espresso
from .pseudopotentials import populate_pseudopotentials
from .qe_pw2traj import write_traj
from .custom import hpc_settings

PSP_DIRS = {'quartz': '/usr/WS1/woodgrp/catalysis/espresso_tool/pseudo/',
            'lassen': '/usr/WS1/woodgrp/catalysis/espresso_tool/pseudo/'}


def run_qe(atom_hex, qe_settings):
    '''
    Figures out what machine you're on and then runs QE appropriately

    Args:
        atom_hex        An `ase.Atoms` object encoded as a hex string.
        qe_settings     A dictionary containing various Quantum Espresso settings.
                        You may find a good set of defaults somewhere in in
                        `gaspy.defaults`
    Returns:
        atoms_name  The `ase.Atoms` object converted into a string
        traj_hex    The entire `ase.Atoms` trajectory converted into a hex
                    string
        energy      The final potential energy of the system [eV]
    '''
    # First, figure out what host we're on. This tells us how to create the
    # input file.
    node_name = socket.gethostname()
    if 'quartz' in node_name:
        host_name = 'quartz'
    elif 'lassen' in node_name:
        host_name = 'lassen'
    else:
        raise RuntimeError('Using node %s, but we do not recognize it. Please '
                           'add it to espresso_tools.custom' % node_name)
    create_input_file(atom_hex, qe_settings, host_name)

    # Get the host name, which tells us how to run the job
    print('Job started on %s at %s' % (host_name, datetime.now()))
    if host_name == 'quartz':
        _run_on_quartz()
    elif host_name == 'lassen':
        _run_on_lassen()
    else:
        raise RuntimeError('espresso_tools does not yet have directions for '
                           'running on this host.')
    print('Job ended on %s' % datetime.now())

    # Parse the output
    images = write_traj()
    atoms_name = str(images[-1])
    with open('all.traj', 'rb') as file_handle:
        traj_hex = file_handle.read().hex()
    energy = images[-1].get_potential_energy()
    return atoms_name, traj_hex, energy


def create_input_file(atom_hex, qe_settings, host_name):
    '''
    This is the main wrapper between GASpy and espressotools. It'll take an
    atoms object in hex form, some Quantum Espresso settings whose defaults can
    be found in GASpy, and then create a Quantum Espresso output file for you
    called 'pw.in'

    Args:
        atom_hex        An `ase.Atoms` object encoded as a hex string.
        qe_settings     A dictionary containing various Quantum Espresso settings.
                        You may find a good set of defaults somewhere in in
                        `gaspy.defaults`
        host_name       A string indicating which host you're using. Helps us
                        decide where to look for the pseudopotentials.
    '''
    # Parse the atoms object
    atoms = decode_trajhex_to_atoms(atom_hex)

    # Get the location of the pseudopotentials
    pspdir, setups = populate_pseudopotentials(qe_settings['psps'])

    # Use espressotools to do the heavy lifting
    calc = espresso(calcmode=qe_settings['mode'],
                    xc=qe_settings['xcf'],
                    # [pw] eV, wave function cutoff, chg density cutoff 'dw'
                    # defaults to 10*pw
                    pw=qe_settings['encut'],
                    kptshift=(0, 0, 0),
                    spinpol=qe_settings['spol'],
                    psppath=PSP_DIRS[host_name],
                    setups=qe_settings['setups'],
                    # [sigma] eV, defaults to 0 smearing fixed-occupations; set to
                    # non-zero for gaussian smearing
                    sigma=qe_settings['sigma'],
                    deuterate=0)
    calc.set(atoms=atoms, kpts=qe_settings['kpts'])
    calc.initialize(atoms)


def decode_trajhex_to_atoms(hex_, index=-1):
    '''
    Decode a trajectory-formatted atoms object from a hex string. Only ensured
    to work with Python 3.

    Arg:
        hex_    A hex-encoded string of a trajectory of atoms objects.
        index   Trajectories can contain multiple atoms objects.
                The `index` is used to specify which atoms object to return.
                -1 corresponds to the last image.
    Output:
        atoms   The decoded ase.Atoms object
    '''
    # Make the trajectory from the hex
    with open('atoms_in.traj', 'wb') as fhandle:
        fhandle.write(bytes.fromhex(hex_))

    # Open up the atoms from the trajectory
    atoms = ase.io.read('atoms_in.traj', index=index)

    return atoms


def _run_on_quartz():
    ''' Runs Quantum Espresso on Quartz '''
    # Get and distribute the HPC settings
    settings = hpc_settings('quartz')
    pw_executable = settings['qe_executable']
    nodes = settings['nodes']
    ntasks = nodes * settings['cores_per_node']

    # Run
    command = ('srun --nodes=%i --ntasks=%i %s -in pw.in'
               % (nodes, ntasks, pw_executable))
    process = subprocess.Popen(command.split())  # noqa: F841


def _run_on_lassen():
    ''' Runs Quantum Espresso on Lassen '''
    # Get and distribute the HPC settings
    settings = hpc_settings('lassen')
    pw_executable = settings['qe_executable']
    nodes = settings['nodes']
    ntasks = nodes * settings['cores_per_node']

    # Run
    command = ('jsrun --nodes=%i --ntasks=%i %s -in pw.in'
               % (nodes, ntasks, pw_executable))
    process = subprocess.Popen(command.split())  # noqa: F841
