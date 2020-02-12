'''
This module contains light wrapping tools between LLNL's espressotools and
CMU's GASpy. Requires Python 3. Tailored to run RISM.

Reference:
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.96.115429
'''

__authors__ = ['Joel Varley', 'Kevin Tran']
__emails__ = ['varley2@llnl.gov', 'ktran@andrew.cmu.edu']

import os
import sys
import warnings
import json
import numpy as np
from ase.data import covalent_radii
from fireworks import LaunchPad
from .core import _run_qe, decode_trajhex_to_atoms
from ..qe_pw2traj import _find_qe_output_name, read_positions_qe, FailedToReadQeOutput
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
        atoms_name      The `ase.Atoms` object converted into a string
        traj_hex        The entire `ase.Atoms` trajectory converted into a hex
                        string
        energy          The final potential energy of the system [eV]
        fermi_energy    The final Fermi energy of the system [eV]
    '''
    atoms_name, traj_hex, energy = _run_qe(atom_hex, rism_settings,
                                           create_rism_input_file)
    fermi_energy = _read_fermi_from_output()
    return atoms_name, traj_hex, energy, fermi_energy


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
    # If we've already tried this calculation on this host and it timed out
    # gracefully, then we instead restart that old calculation.
    try:
        atoms = get_atoms_from_old_run()

    # Otherwise, we need to start from scratch
    except FailedToReadQeOutput:
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
                        rism3d_maxstep=int(5e5),
                        rism1d_maxstep=int(1e5),
                        mdiis3d_size=15,
                        mdiis1d_size=20,
                        startingpot=rism_settings['startingpot'],
                        startingwfc=rism_settings['startingwfc'],
                        outdir=outdir,
                        prefix='rism')
    calc.set(atoms=atoms)
    _post_process_rismespresso(calc, atoms, rism_settings)

    # Create the input file
    calc.initialize(atoms)


def get_atoms_from_old_run():
    '''
    This function will try to look for a FireWorks directory on this host that
    contains the same calculation, but got cut off due to wall time. It will
    then move us into that directory and modify the 'pw.in' file to do a
    restart calculation instead of a calculation from scratch.

    Returns:
        atoms   `ase.Atoms` object of the final image in the trajectory of the
                previous run
    '''
    old_output = _find_previous_output_file()
    images = read_positions_qe(old_output)
    atoms = images[-1]
    return atoms


def _find_previous_output_file(fw_json='FW.json'):
    '''
    Search for fireworks that look like the one we're trying to run now. If
    there are any matching fireworks that have fizzled, ended with a walltime
    error, and also happen to have launch directories on this same host, then
    return the location of the output file.

    Arg:
        fw_json     String indicating the path of a JSON-formatted file that
                    contains the FireWorks info you want to use as a comparison
    Returns:
        out_file    String indicating the path of the last walltimed launch
                    directory that matches the provided firework (if it's on
                    this host). If there is no match, returns an empty string.
    '''
    lpad = _get_launchpad()

    # Get the IDs of the fizzled fireworks that match the one we're trying to
    # run now
    with open(fw_json, 'r') as file_handle:
        fw_info = json.load(file_handle)
    query = {'name.%s' % key: value for key, value in fw_info['name'].items()}
    query['state'] = 'FIZZLED'
    fws = list(lpad.fireworks.find(filter=query, projection={'fw_id': 1, '_id': 0}))
    fwids = [fw['fw_id'] for fw in fws]

    # Get the launch information for each of the matching fireworks
    launches = list(lpad.launches.find(filter={'fw_id': {'$in': fwids}},
                                       projection={'fw_id': 1, 'launch_dir': 1, '_id': 0}))
    launches = sorted(launches, key=lambda launch: launch['fw_id'], reverse=True)

    # Look for the error file(s) of the launches
    for launch in launches:
        launch_dir = launch['launch_dir']
        try:
            for file_name in sorted(os.listdir(launch_dir), reverse=True):
                if file_name.endswith('error'):
                    with open(os.path.join(launch_dir, file_name), 'r') as file_handle:
                        for line in reversed(file_handle.readlines()):

                            # If the error file tells us that we hit the wall
                            # time, then we know to return this output file
                            if 'AssertionError: Calculation hit the wall time' in line:
                                print('Using atomic positions from previous calculation at:  ' + launch_dir)
                                out_file = os.poth.join(launch_dir, file_name)
                                return out_file
                    break

        # Move on if the launch directory doesn't exist on this host
        except FileNotFoundError:
            continue
    return ''


def _get_launchpad(submission_script='FW_submit.script'):
    '''
    This function assumes that you're in a directory where a "FW_submit.script"
    exists and contains the location of your launchpad.yaml file. It then uses
    this yaml file to instantiate a LaunchPad object for you.

    Arg:
        submission_script   String indicating the path of the job submission
                            script used to launch this firework. It should
                            contain an `rlaunch` command in it.
    Returns:
        lpad    A configured and authenticated `fireworks.LaunchPad` object
    '''
    # Look for the line in the submission script that has `rlaunch`
    with open(submission_script, 'r') as file_handle:
        for line in file_handle.readlines():
            if line.startswith('rlaunch'):
                break

    # The line with `rlaunch` should also have the location of the launchpad
    words = line.split(' ')
    for i, word in enumerate(words):
        if word == '-l':
            lpad_file = words[i+1]
            break

    # Instantiate the lpad with the yaml and return it
    lpad = LaunchPad.from_file(lpad_file)
    return lpad


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
    the lowest surface atom.

    Arg:
        atoms   The `ase.Atoms` object you're trying to relax
    Returns:
        starting_height     A float indicating the location in the z-direction
                            at which to start the laue region (Angstroms).
    '''
    surface_atom_indices = find_surface_atom_indices(atoms)
    positions = atoms.get_positions()
    try:
        starting_height = min(positions[i][2] for i in surface_atom_indices) - 1
        return starting_height

    except ValueError:
        warnings.warn('No surface atoms were found. We will assume that this is '
                      'a molecular relaxation and set `laue_starting_right` to 0',
                      RuntimeWarning)
        return 0


def find_surface_atom_indices(atoms, covalent_percent=1.25):
    '''
    If an atom is under-coordinated relative to the maximum coordination of its
    corresponding element, then it is considered a surface atom. This function
    finds these surface atoms and gives you their indices. Note that this
    function assumes that two atoms are "coordinated" if their distance is smaller
    than the sum of their covalent radii.

    Loosely based on a gist from tgmaxson
    (https://gist.github.com/tgmaxson/8b9d8b40dc0ba4395240).

    Args:
        atoms               `ase.Atoms` object of the slab you want to find the
                            surface atoms of
        covalent_percent    Fudge [float] that assigns "bonds" more liberally
                            (or conservatively) based on their interatomic
                            distances. Increase to have more "bonding" and vice
                            versa.
    Returns:
        surface_indices     A list of integers corresponding to the indices of
                            the surface atoms within the provided `ase.Atoms`
                            object.
    '''
    # Get the covalent radius of each atom
    radii = np.take(covalent_radii, atoms.numbers)

    # Get all the distances between each of the atoms
    scaled_distances = np.divide(atoms.get_all_distances(mic=True), covalent_percent)

    # Create the connectivity matrix that has 1's where there is a bond and 0's
    # where there is not a bond. The indices of the matrix correspond to the
    # indices of the atoms in the `ase.Atoms` object.
    connectivity_matrix = np.empty((len(atoms), len(atoms)))
    for i, radius_i in enumerate(radii):
        for j, radius_j in enumerate(radii):
            if i != j and scaled_distances[i, j] <= radius_i + radius_j:
                connectivity_matrix[i, j] = 1
            else:
                connectivity_matrix[i, j] = 0
    coordinations = connectivity_matrix.sum(axis=0)

    # Find the maximum coordination for each element
    elements = {atom.symbol for atom in atoms}
    max_coords = dict.fromkeys(elements, 0)
    for atom, coordination in zip(atoms, coordinations):
        element = atom.symbol
        max_coords[element] = max(max_coords[element], coordination)

    # Find all the surface atoms based on their coordination
    scaled_positions = atoms.get_scaled_positions()
    surface_indices = []
    for i, (atom, scaled_position, coordination) in enumerate(zip(atoms, scaled_positions, coordinations)):
        # Don't want to find atoms on the bottom of the slab
        if scaled_position[2] > 0.5:
            # Reduced the coordination cutoff threshold as a heuristic to
            # compensate for naturally under-coordinated atoms
            if coordination < max_coords[atom.symbol] - 2:
                surface_indices.append(i)

    return surface_indices


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
                            starting_charge
                            esm_only
                            target_fermi
                            LJ_epsilon
                            LJ_sigma
                            molecule
                            charge
    '''
    # Set the initial charges
    starting_charge = rism_settings['starting_charge']
    if starting_charge:
        starting_charge = format_LJ(atoms, starting_charge)
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
                 startingpot=1,
                 startingwfc=1,
                 constmu=None)


def _read_fermi_from_output(qe_output_name=None):
    '''
    This function will try to get the last Fermi energy reported in the RISM-QE
    log file.

    Arg:
        qe_output_name  String indicating the location of the RISM-QE output
                        file
    Returns:
        fermi   The last reported Fermi energy in the log file [eV]
    '''
    # If the log file is not provided, then guess it
    if qe_output_name is None:
        qe_log_name = _find_qe_output_name()

    # We assume the log file will say something like:
    # "the Fermi energy is    -4.6007 ev" so we grep it accordingly
    with open(qe_log_name) as file_handle:
        for line in reversed(file_handle.readlines()):
            if 'the Fermi energy is' in line:
                fermi = float(line.split(' ')[-2])
                break
    try:
        return fermi

    # More detail error handling
    except NameError as error:
        message = ('happened because we could not find the Fermi energy from '
                   'the output file %s' % qe_log_name)
        raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])
