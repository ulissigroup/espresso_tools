'''
This module contains light wrapping tools between LLNL's espressotools and
CMU's GASpy. Requires Python 3. Tailored to run RISM.

Reference:
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.96.115429
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

from .core import _run_qe, decode_trajhex_to_atoms
from ..cpespresso_v3 import espresso
from ..pseudopotentials import populate_pseudopotentials


def run_rism(atom_hex, qe_settings):
    '''
    Function that runs a RISM version of Quantum Espresso for you given a hex
    string of an `ase.Atoms` object and various run settings.

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
                                           create_rism_input_file)
    return atoms_name, traj_hex, energy


# TODO: This is currently no different from Vanilla QE. Need to modify to use
# rismtools and whatnot
def create_rism_input_file(atom_hex, qe_settings, host_name):
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