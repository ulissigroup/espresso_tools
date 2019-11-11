'''
This module contains light wrapping tools between LLNL's espressotools and
CMU's GASpy. Requires Python 3
'''

__authors__ = ['Kevin Tran']
__emails__ = ['ktran@andrew.cmu.edu']

import subprocess
import time
import math
import ase.io
from ..qe_pw2traj import write_traj
from ..custom import hpc_settings


def _run_qe(atom_hex, qe_settings, input_file_creator):
    '''
    Figures out what machine you're on and then runs QE appropriately. Meant to
    be used as an abstract parent function.

    Args:
        atom_hex            An `ase.Atoms` object encoded as a hex string.
        qe_settings         A dictionary containing various Quantum Espresso
                            settings. You may find a good set of defaults
                            somewhere in in `gaspy.defaults`
        input_file_creator  A function that takes the `atom_hex` argument,
                            `qe_settings` argument, and the host name and then
                            creates an input file called `pw.in`.
    Returns:
        atoms_name  The `ase.Atoms` object converted into a string
        traj_hex    The entire `ase.Atoms` trajectory converted into a hex
                    string
        energy      The final potential energy of the system [eV]
    '''
    # Create the input file and then run the job
    input_file_creator(atom_hex, qe_settings)
    run_job()

    # Parse the output
    images = write_traj()
    atoms_name = str(images[-1])
    with open('all.traj', 'rb') as file_handle:
        traj_hex = file_handle.read().hex()
    energy = images[-1].get_potential_energy()
    return atoms_name, traj_hex, energy


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


def run_job():
    '''
    This function will try to run the `pw.in` input file in the current
    directory.
    '''
    # Get and distribute the HPC settings
    settings = hpc_settings()
    manager = settings['manager']
    nodes = settings['nodes']
    cores_per_node = settings['cores_per_node']
    threads_per_core = settings['threads_per_core']
    pw_executable = settings['qe_executable']

    # Use heuristics to trim down run conditions for small systems
    min_atoms = 5
    if _find_n_atoms() <= min_atoms:
        nodes = 1
        cores_per_node = min(math.ceil(cores_per_node / 2), 4)
        print('Less than %i atoms, so we are assuming this is a gas phase '
              'calculation and using %i node and %i cores per node to ensure '
              'that there are more processors than bands.'
              % (min_atoms, nodes, cores_per_node))

    # Call the HPC-specific command to actually run
    if manager == 'slurm':
        _run_on_slurm(nodes, cores_per_node, threads_per_core, pw_executable)
    elif manager == 'lsf':
        _run_on_lsf(nodes, cores_per_node, threads_per_core, pw_executable)
    else:
        raise RuntimeError('espresso_tools does not yet know how to submit '
                           'jobs to the "%s" manager. Please modify '
                           'espresso_tools.gaspy_wrappers.core.run_job '
                           'to specify.' % manager)

    # We think that FireWorks or Quantum Espresso give Python the stdout before
    # it's written to the log file. When this happens, espresso_tools ends up
    # reading an incomplete log file and fails. We hack around this issue by
    # waiting two minutes for Quantum Espresso/FireWorks to finish writing to
    # the output file.
    time.sleep(120)  # accepts units of seconds


def _find_n_atoms():
    '''
    Reads the `pw.in` file and counts the number of atoms by counting the
    number of lines between the 'ATOMIC_POSITIONS' line and the 'K_POINTS'
    line. Yes, this is a big assumption, but we're also assuming that you'll be
    using the espresso_tools framework, which usually creates input files like
    this.
    '''
    counting = False
    n_atoms = 0
    with open('pw.in', 'r') as file_handle:
        for line in file_handle.readlines():
            # Stop counting if we reached the ending flag
            if 'K_POINTS' in line:
                break
            if counting:
                n_atoms += 1
            # Start counting after we've reached the starting flag
            if 'ATOMIC_POSITIONS' in line:
                counting = True
    return n_atoms


def _run_on_slurm(nodes, cores_per_node, threads_per_core, pw_executable):
    '''
    Calls Quantum Espresso on Quartz

    Args:
        nodes               An integer indicating how many nodes you want to
                            run on
        cores_per_node      An integer indicating the total number of cores you
                            want to use per node
        threads_per_core    An integer indicating the total number of threads
                            you want to use per core
        pw_executable       A string indicating the location of the Quantum
                            Espresso executable file you want to use
    '''
    n_tasks = nodes * cores_per_node * threads_per_core
    command = ('srun --nodes=%i --ntasks=%i %s -nk %i -nd %i -in pw.in'
               % (nodes, n_tasks, pw_executable, nodes, cores_per_node))
    print('Executing:  %s' % command)
    _ = subprocess.Popen(command.split()).communicate()  # noqa: F841


def _run_on_lsf(nodes, cores_per_node, threads_per_core, pw_executable):
    '''
    Calls Quantum Espresso on Lassen

    Args:
        nodes               An integer indicating how many nodes you want to
                            run on
        cores_per_node      An integer indicating the total number of cores you
                            want to use per node
        threads_per_core    An integer indicating the total number of threads
                            you want to use per core
        pw_executable       A string indicating the location of the Quantum
                            Espresso executable file you want to use
    '''
    n_tasks = nodes * cores_per_node * threads_per_core
    command = ('lrun -n%i %s -nk%s -nd%i -in pw.in'
               % (n_tasks, pw_executable, nodes, cores_per_node))
    print('Executing:  %s' % command)
    _ = subprocess.Popen(command.split()).communicate()  # noqa: F841
