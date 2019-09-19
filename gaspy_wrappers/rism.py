'''
This module contains light wrapping tools between LLNL's espressotools and
CMU's GASpy. Requires Python 3. Tailored to run RISM.

Reference:
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.96.115429
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

import json
import numpy as np
from .core import _run_qe, decode_trajhex_to_atoms
from ..cpespresso_v3 import rismespresso
from ..pseudopotentials import populate_pseudopotentials
from ..custom import hpc_settings, LJ_PARAMETERS
from ..rismtools import (populate_solvent_specie,
                         populate_solvents,
                         get_molchg,
                         format_LJ)


def run_rism(atom_hex, rism_settings):
    '''
    Function that runs a RISM version of Quantum Espresso for you given a hex
    string of an `ase.Atoms` object and various run settings.

    In practice, this is a Light wrapper for the parent function
    `.core._run_qe`.

    Args:
        atom_hex        An `ase.Atoms` object encoded as a hex string.
        rism_settings   A dictionary containing various Quantum Espresso settings.
                        You may find a good set of defaults somewhere in in
                        `gaspy.defaults`
    Returns:
        atoms_name  The `ase.Atoms` object converted into a string
        traj_hex    The entire `ase.Atoms` trajectory converted into a hex
                    string
        energy      The final potential energy of the system [eV]
    '''
    atoms_name, traj_hex, energy = _run_qe(atom_hex, rism_settings,
                                           create_rism_input_file)
    return atoms_name, traj_hex, energy


def create_rism_input_file(atom_hex, rism_settings):
    '''
    This is the main wrapper between GASpy and espressotools. It'll take an
    atoms object in hex form, some Quantum Espresso settings whose defaults can
    be found in GASpy, and then create a Quantum Espresso output file for you
    called 'pw.in'

    Args:
        atom_hex        An `ase.Atoms` object encoded as a hex string.
        rism_settings   A dictionary containing various Quantum Espresso-RISM
                        settings. You may find a good set of defaults
                        somewhere in in `gaspy.defaults`
    Returns:
        atoms   The `ase.Atoms` object that was decoded from the hex string.
    '''
    # Parse various input parameters/settings into formats accepted by the
    # `rismespresso` class
    atoms = _parse_atoms(atom_hex)
    pspdir, setups = populate_pseudopotentials(rism_settings['psps'], rism_settings['xcf'])
    solvents, anions, cations = _parse_solvent(rism_settings, pspdir)
    laue_starting_right = _calculate_laue_starting_right(atoms)
    calcmode = rism_settings.get('calcmode', 'relax')
    settings = hpc_settings()
    try:
        nosym = rism_settings['nosym']
    except KeyError:
        nosym = False

    # Get the FireWorks ID, which will be used as the directory for the
    # scratch/outdir files
    with open('FW.json', 'r') as file_handle:
        fw_info = json.load(file_handle)
    fw_id = fw_info['fw_id']
    outdir = settings['scratch_dir'] + '/%s' % fw_id

    # Set the run-time to 2 minutes less than the job manager's wall time
    wall_time = settings['wall_time']
    max_seconds = wall_time * 60 * 60 - 120

    # Use rismespresso to do the heavy lifting
    calc = rismespresso(calcmode=calcmode,
                        printforces=True,
                        xc=rism_settings['xcf'],
                        pw=rism_settings['encut'],
                        kpts=rism_settings['kpts'],
                        kptshift=(0, 0, 0),
                        sigma=rism_settings['sigma'],
                        smearing=rism_settings['smearing'],
                        spinpol=rism_settings['spol'],
                        psppath=pspdir,
                        setups=setups,
                        max_seconds=max_seconds,
                        solvents=solvents,
                        cations=cations,
                        anions=anions,
                        laue_starting_right=laue_starting_right,
                        conv_thr=rism_settings['conv_elec'],
                        laue_expand_right=rism_settings['laue_expand_right'],
                        mdiis1d_step=rism_settings['mdiis1d_step'],
                        rism1d_conv_thr=rism_settings['rism1d_conv_thr'],
                        rism3d_conv_thr=rism_settings['rism3d_conv_thr'],
                        rism3d_conv_level=rism_settings['rism3d_conv_level'],
                        mdiis3d_step=rism_settings['mdiis3d_step'],
                        nosym=nosym,
                        nstep=200,
                        electron_maxstep=1000,
                        mixing_mode='local-TF',
                        laue_reference='right',
                        rism3d_maxstep=int(8e4),
                        rism1d_maxstep=int(1e5),
                        mdiis3d_size=15,
                        mdiis1d_size=20,
                        outdir=outdir,
                        prefix='rism')
    calc.set(atoms=atoms)
    _post_process_rismespresso(calc, atoms, rism_settings)

    # Create the input file
    calc.initialize(atoms)


def _parse_atoms(atom_hex):
    '''
    This function will read the hex string and decode it to an `ase.Atoms`
    object. It will then center the slab at Z = 0 and add either 20 Angstroms
    of vacuum or the minimum vacuum size to let RISM run, whichever is greater.
    The centering reduces unit cell size and therefore calculation speed.

    Arg:
        atom_hex    An `ase.Atoms` object encoded as a hex string.
    Returns:
        parsed_atoms    The decoded `ase.Atoms` object, but also with a vacuum
                        buffer at the top of the unit cell and with the slab
                        centered at Z=0
    '''
    atoms = decode_trajhex_to_atoms(atom_hex)

    # Fix the centering and unit cell of slabs and adsorbate+slabs
    try:
        centered_atoms = _center_slab(atoms)
        parsed_atoms = _set_unit_cell_height(centered_atoms)
        return parsed_atoms

    # If there are no slab atoms (e.g., if we are doing just a molecule), then
    # there's no need to do this processing. The
    # `__update_molecular_parameters` function will take care of that.
    except ValueError:
        return atoms


def _center_slab(atoms):
    '''
    This will center a slab or adslab so that the Cartesian middle-point of the
    slab/adslab (in the Z-direction) will be at Z = 0. Any adsorbates will be
    ignored.

    Arg:
        atoms   `ase.Atoms` object of the [ad]slab. As per the GASpy
                infrastructure, we assume that atoms with the `0` tag are slab
                atoms, and atoms with tags `> 0` are adsorbates.
    Returns:
        atoms   A new instance of the `ase.Atoms` where the [ad]slab is
                centered.
    '''
    # Make a new instance so we don't mess with the old one
    atoms = atoms.copy()

    # Move the adslab down so that its center is at Z = 0
    slab_atom_heights = [atom.position[2] for atom in atoms if atom.tag == 0]
    slab_top = max(slab_atom_heights)
    slab_bottom = min(slab_atom_heights)
    slab_height = slab_top - slab_bottom
    atoms.translate([0, 0, -(slab_top - slab_height/2)])
    return atoms


def _set_unit_cell_height(atoms):
    '''
    This will extend the height of the unit cell until it there is at least 20
    Angstroms of vacuum or the height is twice the height of the [ad]slab,
    whichever is greater. The former is to ensure steady-state of the
    electrolyte in the z-direction, and the latter is to ensure that RISM can
    actually run, since it requires that all atoms are within +/- L/2, where L
    is the height of the unit cell.

    NOTE:  This function assumes that the structure has already been centered
    with the `_center_slab` function.

    Arg:
        atoms   `ase.Atoms` object of the [ad]slab. As per the GASpy
                infrastructure, we assume that atoms with the `0` tag are slab
                atoms, and atoms with tags `> 0` are adsorbates.
    Returns:
        atoms   A new instance of the `ase.Atoms` where the unit cell is set
        appropriately.
    '''
    # Make a new instance so we don't mess with the old one
    atoms = atoms.copy()

    # Find the minimum cell height required to run RISM. This assumes that the
    # slab has already been centered with `_center_slab`.
    max_height = max(atom.position[2] for atom in atoms)
    min_height_for_rism = 2 * max_height + 1  # Add one Angstrom buffer for safety

    # Find the minimum cell height required to yield 20 Angstroms of vacuum
    slab_atom_heights = [atom.position[2] for atom in atoms if atom.tag == 0]
    min_height_for_vacuum = max(slab_atom_heights) + 20.

    # Set the cell height to the greater of the two heights
    cell = atoms.get_cell()
    cell[2, 2] = max(min_height_for_rism, min_height_for_vacuum)
    atoms.set_cell(cell)
    return atoms


def _parse_solvent(rism_settings, pspdir):
    '''
    This function will parse the `rism_settings` dictionary from GASpy into the
    appropriate objects that espresso_tools can use to create a RISM
    calculation.

    Args:
        rism_settings   A dictionary of Quantum Espresso-RISM calculation
                        parameters. You can find templates in `gaspy.defaults`.
        pspdir          A string for the directory of the pseudopotentials.
                        This should probably be obtained from
                        `espresso_tools.pseudopotentials.populate_pseudopotentials`.
    Returns:
        solvents    An output of `espresso_tools.rismtools.populate_solvent_specie`
        anions      The anion output of `espresso_tools.rismtools.populate_solvents`
        cations     The cation output of `espresso_tools.rismtools.populate_solvents`
    '''
    # Stop now if the electrolyte change balance isn't right
    anion_concs = rism_settings['anion_concs']
    cation_concs = rism_settings['cation_concs']
    _check_solvent_balance(anion_concs, cation_concs)

    # Auto-populate H2O tip5 solvent
    solvents = [populate_solvent_specie('H2O', -1, file='H2O.tip5p.MOL')]

    # Parse the electrolyte information out of the input
    anion_names = list(anion_concs.keys())
    anion_concs = list(anion_concs.values())
    cation_names = list(cation_concs.keys())
    cation_concs = list(cation_concs.values())

    # Populate electrolytes in proper format
    anions = populate_solvents(anion_names, anion_concs, pspdir)
    cations = populate_solvents(cation_names, cation_concs, pspdir)
    return solvents, anions, cations


def _check_solvent_balance(anion_concs, cation_concs):
    '''
    This function will raise a RuntimeError if the charge of the electrolytes
    is not balanced.

    Arg:
        anion_concs     A dictionary whose keys are the names of the anions and
                        whose values are their concentrations [M]
        cation_concs    A dictionary whose keys are the names of the cations and
                        whose values are their concentrations [M]
    '''
    # Concatenate the dictionaries of ion concentrations
    ion_concs = dict(anion_concs, **cation_concs)

    # Add the charge contribution of each ion
    qbal = 0
    for ion, conc in ion_concs.items():
        molchg = get_molchg(ion) * float(conc)
        qbal += molchg

    # If the charges do not sum to 0, then stop and yell at the user
    if qbal != 0:
        raise RuntimeError('The change balance is %.4f when it should be 0. '
                           'Please update the electrolytes to balance the '
                           'charge.' % qbal)


def _calculate_laue_starting_right(atoms):
    '''
    This function will calculate the location you should use for
    "laue_starting_right". In other words, it tells you where to start the
    mean-field section in the z-axis. Here, we set it equal to 1 Angstrom below
    the upper-most slab atom.

    Note that we assume that the atoms objects are coming from GASpy, which
    sets the tags of slab atoms to `0` and adsorbate atoms to `> 0`.

    Arg:
        atoms   The `ase.Atoms` object you're trying to relax
    Returns:
        starting_height     A float indicating the location in the z-direction
                            at which to start the laue region (Angstroms).
    '''
    try:
        max_height = max(atom.position[2] for atom in atoms if atom.tag == 0)
        starting_height = max_height - 1.
        return starting_height

    # If there are no atoms with a tag of 0, then we're dealing with a
    # molecule. And if we're dealing with a molecule, then it doesn't matter
    # what we return here. It'll just be overridden by the
    # `__update_molecular_parameters` function.
    except ValueError:
        return 0


def _post_process_rismespresso(calc, atoms, rism_settings):
    '''
    Modifies the `rismespresso` object in various ways depending on the flags
    you supplied to `rism_settings`. This is kind of a catch-all function for
    doing this to the class after instantiation.

    Args:
        calc            The instance of the
                        `espresso_tools.cpespresso_v3.rismespresso` class that
                        you're using
        atoms           The `ase.Atoms` structure you're trying to relax
        rism_settings   A dictionary whose key/values may trigger the
                        post-processing. Possible keys may include:
                            starting_charges
                            esm_only
                            target_fermi
                            LJ_epsilon
                            LJ_sigma
                            molecule
                            charge
    '''
    # Set the initial charges
    starting_charges = rism_settings['starting_charges']
    if starting_charges:
        starting_charge = format_LJ(atoms, starting_charges)
        calc.set(starting_charge=starting_charge)

    # Whether or not you want to do just ESM (True), or ESM-RISM (False)
    esm_only = rism_settings['esm_only']
    if esm_only:
        calc.set(trism=0)
        calc.set(esm_bc='bc3')
        print("SETTING ESM:   TRISM = FALSE ; BC='BC3'!")

    # Set the Fermi level, which is effectively the applied potential. If not
    # specified, then do a constant-charge calculation (instead of constant
    # fermi)
    try:
        target_fermi = rism_settings['target_fermi']
        calc.set(constmu=1,
                 fcp_mu=target_fermi,
                 fcp_conv_thr=rism_settings['fcp_conv_thr'],
                 freeze_all_atoms=rism_settings['freeze_all_atoms'])
        print("SETTING CONSTANT-MU CALCULATION AT E_FERMI = %s" % (target_fermi))
    except KeyError:
        pass

    # Call some helper functions to do fancier post-processing
    __update_LJ_parameters(calc, atoms, rism_settings)
    __update_molecular_parameters(calc, atoms, rism_settings)

    # Set the charge manually. If not specified, then default to 0
    try:
        charge = rism_settings['charge']
    except KeyError:
        charge = 0.
    calc.set(tot_charge=charge)
    print("SETTING CHARGE %s" % (charge))


def __update_LJ_parameters(calc, atoms, rism_settings):
    '''
    This script will identify each of the species in the quantum section of the
    cell and then assign their corresponding Lennard-Jones parameters from our
    default settings.

    Args:
        calc            The instance of the
                        `espresso_tools.cpespresso_v3.rismespresso` class that
                        you're using
        atoms           The `ase.Atoms` structure you're trying to relax
        rism_settings   A dictionary whose with the `LJ_epsilon` and `LJ_sigma`
                        keys. If `None` then espresso_tools will use some
                        defaults. Or you can give it a list with the same
                        length as the number of atoms in the `atoms` object.
    '''
    # Set the epsilon values
    if rism_settings['LJ_epsilons'] is None:
        LJ_epsilons = [LJ_PARAMETERS[atom.symbol]['epsilon'] for atom in atoms]
    else:
        LJ_epsilons = rism_settings['LJ_epsilons']
    solute_epsilons = format_LJ(atoms, LJ_epsilons)
    calc.set(solute_epsilons=solute_epsilons)
    print("SETTING LJ epsilons! %s" % (LJ_epsilons))

    # Set the sigma values
    if rism_settings['LJ_epsilons'] is None:
        LJ_sigmas = [LJ_PARAMETERS[atom.symbol]['sigma'] for atom in atoms]
    else:
        LJ_sigmas = rism_settings['LJ_sigmas']
    solute_sigmas = format_LJ(atoms, LJ_sigmas)
    calc.set(solute_sigmas=solute_sigmas)
    print("SETTING LJ sigmas! %s" % (LJ_sigmas))


def __update_molecular_parameters(calc, atoms, rism_settings):
    '''
    If this is a solvent-phase molecule, then change everything up

    Args:
        calc            The instance of the
                        `espresso_tools.cpespresso_v3.rismespresso` class that
                        you're using
        atoms           The `ase.Atoms` structure you're trying to relax
        rism_settings   A dictionary that may have the key `molecule` with a
                        Boolean value. If `True`, then this function will
                        update the calculator settings. If `False` or not
                        there, then it will do nothing.
    '''
    try:  # If the user does not specify, assume it's not a molecule
        molecule = rism_settings['molecule']
    except KeyError:
        molecule = False

    # Center the molecule
    if molecule:
        atoms.center()
        zcom = np.array([0, 0, atoms.get_center_of_mass()[-1]])
        atoms.translate(-zcom)
        calc.set(atoms=atoms)
        print("DUAL BOUNDARY RISM")

        # Need to ensure that molecule is all oplsaa and no remnant LJ
        # parameters?
        calc.set(laue_starting_right=0,
                 laue_expand_left=90,
                 laue_starting_left=0,
                 laue_reference='average',
                 isolated='esm',
                 esm_bc='bc1',
                 constmu=None)
