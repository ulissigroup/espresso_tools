'''
This module contains light wrapping tools between LLNL's espressotools and
CMU's GASpy. Requires Python 3
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

import subprocess
import socket
import ase.io
from .cpespresso_v3 import espresso
from .pseudopotentials import populate_pseudopotentials
from .qe_pw2traj import write_traj
from .custom import hpc_settings


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
    # First, figure out what host we're on.
    node_name = socket.gethostname()
    if 'quartz' in node_name:
        host_name = 'quartz'
    elif 'lassen' in node_name:
        host_name = 'lassen'
    else:
        raise RuntimeError('Using node %s, but we do not recognize it. Please '
                           'add it to espresso_tools.custom' % node_name)

    # Create the input file, then call the appropriate job manager to actually
    # run QE
    atoms = create_input_file(atom_hex, qe_settings, host_name)
    _call_job_manager(host_name, atoms)

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
    Returns:
        atoms   The `ase.Atoms` object that was decoded from the hex string.
    '''
    # Parse the atoms object
    atoms = decode_trajhex_to_atoms(atom_hex)

    # Get the location of the pseudopotentials
    pspdir, setups = populate_pseudopotentials(qe_settings['psps'])

    # Use espressotools to do the heavy lifting
    calc = espresso(calcmode='relax',
                    xc=qe_settings['xcf'],
                    # [pw] eV, wave function cutoff, chg density cutoff 'dw'
                    # defaults to 10*pw
                    pw=qe_settings['encut'],
                    kptshift=(0, 0, 0),
                    spinpol=qe_settings['spol'],
                    psppath=pspdir,
                    setups=setups,
                    # [sigma] eV, defaults to 0 smearing fixed-occupations; set to
                    # non-zero for gaussian smearing
                    sigma=qe_settings['sigma'],
                    deuterate=0)
    calc.set(atoms=atoms, kpts=qe_settings['kpts'])
    calc.initialize(atoms)

    return atoms


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


def _call_job_manager(host_name, atoms):
    '''
    This function will guesstimate the number of nodes and tasks you'll need to
    run, and then call the job manager accordingly.

    Args:
        host_name   A string indicating which host you're running on
        atoms       `ase.Atoms` object of what you are trying to relax
    '''
    # Get and distribute the HPC settings
    settings = hpc_settings(host_name)
    nodes = settings['nodes']
    cores_per_node = settings['cores_per_node']
    pw_executable = settings['qe_executable']

    # Use heuristics to trim down run conditions for small systems
    if len(atoms) <= 5:
        nodes = 1

    # Call the HPC-specific command to actually run
    if host_name == 'quartz':
        _run_on_slurm(nodes, cores_per_node, pw_executable)
    elif host_name == 'lassen':
        _run_on_lsf(nodes, cores_per_node, pw_executable)
    else:
        raise RuntimeError('espresso_tools does not yet know what job manager '
                           'that %s uses. Please modify '
                           'espresso_tools.core._call_job_manager to specify.'
                           % host_name)


def _run_on_slurm(nodes, cores_per_node, pw_executable):
    '''
    Calls Quantum Espresso on Quartz

    Args:
        nodes           An integer indicating how many nodes you want to run on
        cores_per_node  An integer indicating the total number of cores you
                        want to use per node
        pw_executable   A string indicating the location of the Quantum
                        Espresso executable file you want to use
    '''
    n_tasks = nodes * cores_per_node
    command = ('srun --nodes=%i --n_tasks=%i %s -in pw.in'
               % (nodes, n_tasks, pw_executable))
    _ = subprocess.Popen(command.split()).communicate()  # noqa: F841


def _run_on_lsf(nodes, cores_per_node, pw_executable):
    '''
    Calls Quantum Espresso on Lassen

    Args:
        nodes           An integer indicating how many nodes you want to run on
        cores_per_node  An integer indicating the total number of cores you
                        want to use per node
        pw_executable   A string indicating the location of the Quantum
                        Espresso executable file you want to use
    Arg:
        atoms   `ase.Atoms` object that will be run
    '''
    command = ('jsrun --nrs=%i --cpu_per_rs=%i %s -in pw.in'
               % (nodes, cores_per_node, pw_executable))
    _ = subprocess.Popen(command.split()).communicate()  # noqa: F841
