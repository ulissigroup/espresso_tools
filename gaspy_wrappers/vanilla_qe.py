'''
This module contains light wrapping tools between LLNL's espressotools and
CMU's GASpy. Requires Python 3
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

import json
from .core import _run_qe, decode_trajhex_to_atoms
from ..cpespresso_v3 import espresso
from ..pseudopotentials import populate_pseudopotentials
from ..custom import hpc_settings


def run_qe(atom_hex, qe_settings):
    '''
    Function that runs Quantum Espresso for you given a hex string of an
    `ase.Atoms` object and various run settings.

    In practice, this is a Light wrapper for the parent function
    `.core._run_qe`.

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
    atoms_name, traj_hex, energy = _run_qe(atom_hex, qe_settings,
                                           create_vanilla_input_file)
    return atoms_name, traj_hex, energy


def create_vanilla_input_file(atom_hex, qe_settings):
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
    '''
    # Parse various input parameters/settings into formats accepted by the
    # `espresso` class
    atoms = decode_trajhex_to_atoms(atom_hex)
    pspdir, setups = populate_pseudopotentials(qe_settings['psps'], qe_settings['xcf'])
    calcmode = qe_settings.get('calcmode', 'relax')
    # Get the FireWorks ID, which will be used as the QE prefix
    with open('FW.json', 'r') as file_handle:
        fw_info = json.load(file_handle)
    prefix = fw_info['fw_id']

    # Set the run-time to 2 minutes less than the job manager's wall time
    settings = hpc_settings()
    wall_time = settings['wall_time']
    max_seconds = wall_time * 60 * 60 - 120
    outdir = settings['scratch_dir']

    # Use espressotools to do the heavy lifting
    calc = espresso(calcmode=calcmode,
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
                    deuterate=0,
                    max_seconds=max_seconds,
                    outdir=outdir,
                    prefix=prefix)
    calc.set(atoms=atoms, kpts=qe_settings['kpts'])
    calc.initialize(atoms)
