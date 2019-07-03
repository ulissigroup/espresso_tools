import os
from functools import reduce


def format_LJ(atoms, parameterlist):
    """ Create format needed for espressotools to populate LJ epsilon or sigma RISM parameters
    Input atoms object (that is ordered to match the input parameter list)
    e.g. formats a list like 0.5,0,0 to { 'atomtype1' : 0.5 , 'atomtype2': 0, ...}
    """
    atypes = reduce(
        lambda l,
        x: l if x in l else l +
        [x],
        atoms.get_chemical_symbols(),
        [])
    d_parameters = {}
    for i, atype in enumerate(atypes):
        d_parameters[atype] = parameterlist[i]
    return d_parameters

########################################################################
#  SOLVENTS ############################################################
########################################################################


def create_rismdir(cationlist, anionlist, cconclist, aconclist):
    """ Create a directory name of format
    cation_anion_molarity
    """
    fullname = zip(cationlist + anionlist, cconclist + aconclist)
    #rismdir  = string.join(fullname,'_')

    # rismdir  = string.join(['_'.join([str(j) for j in i]) for i in
    # fullname],'_') #python2
    rismdir = '_'.join(['_'.join([str(j) for j in i]) for i in fullname])

    # eliminate + and -
    # rismdir  = string.replace(string.replace(rismdir,'+',''),'-','') #python2
    rismdir = (rismdir.replace('+', '')).replace('-', '')

    #uniqconc = list(set(cconclist+aconclist))
    #uniqconc = string.join(uniqconc,'_')
    # print "unitconc: %s" %(uniqconc)
    #rismdir = string.join(cationlist+anionlist,'_')+'_%s' %(uniqconc)
    #rismdir = string.replace(string.replace(rismdir,'+',''),'-','')
    return rismdir

# def get_molfile(specie, moldir,
# topchoice='aq',secondchoice='oplsua',thirdchoice='oplsaa'):


def get_molfile(
        specie,
        moldir,
        topchoice='oplsua',
        secondchoice='oplsaa',
        thirdchoice='aq'):
    """ Enter a specie and it returns which is the best choice for a given
    anion/cation.

    Currently prefers oplsua if it exists, then .opsaa. then .aq.
    """
    mollist = [i for i in os.listdir(
        moldir) if os.path.splitext(i)[-1] == '.MOL']
    slist = [i for i in mollist if i.split('.')[0] == specie]
    set1 = [i for i in slist if topchoice in i.split('.')]
    set2 = [i for i in slist if secondchoice in i.split('.')]
    set3 = [i for i in slist if thirdchoice in i.split('.')]
    # print "setlist %s and set1 %s" %(slist,set1)
    if set1:
        molfile = set1[0]
        if len(set1) > 1:
            print("MULTIPLE MATCHING %s MOL FILES!" % (topchoice))
    elif set2:
        molfile = set2[0]
        if len(set2) > 1:
            print("MULTIPLE MATCHING %s MOL FILES!" % (secondchoice))
    elif set3:
        molfile = set3[0]
        if len(set3) > 1:
            print("MULTIPLE MATCHING %s MOL FILES!" % (thirdchoice))
    else:
        molfile = None
        print("NO MATCHING %s MOL FILES!" % (specie))
    if molfile:
        print("SET %s FILE!" % (molfile))
    return molfile


def get_molchg(specie):
    """ Enter a specie and it counts the + and - to see what its formal
    charge is
    """
    posq = [i for i in specie if i == '+']
    negq = [i for i in specie if i == '-']
    if posq:
        q = len(posq)
    elif negq:
        q = -1 * len(negq)
    else:
        q = 0
    return q


def populate_solvent_specie(specie=None, density=None, file=None):
    """ Generate solvent dictionary of form
    solvents = { 'type' : 'H2O', 'density' : -1, 'file' :'H2O.spc.MOL'}
    Need to feed a list of these dictionaries to rismespresso as solvents,
    cations, and anions. eg of form
    cations  = [{ 'type' : 'Ca++','density' : 0.5, 'file' : 'Ca++.aq.MOL'}]
    anions   = [{ 'type' : 'SO4--','density' : 0.5, 'file' : 'SO4--.aq.MOL'}]
    """
    if file is None:
        #molfile= '%s.aq.MOL' %(specie)
        molfile = '%s.oplsaa.MOL' % (specie)
    else:
        molfile = file
    d_specie = {'type': '%s' % (specie), 'density': density, 'file': molfile}
    return d_specie


def populate_solvents_v0(
        solvent='H2O',
        cation=None,
        dens_cat=1,
        cationfile=None,
        anion=None,
        dens_an=1,
        anionfile=None):
    """ cleaner population of solvent/cation/anion info"""
    #MOLfiles = [os.path.split(i)[-1] for i in glob.glob(os.path.join(pspdir,'*MOL'))]
    d_solv = [populate_solvent_specie('H2O', -1, file='H2O.spc.MOL')]
    if cation:
        d_cat = [populate_solvent_specie(cation, dens_cat, cationfile)]
    else:
        d_cat = None
    if anion:
        d_an = [populate_solvent_specie(anion, dens_an, anionfile)]
    else:
        d_an = None

    return d_solv, d_cat, d_an


def populate_solvents(solvlist, conclist, pspdir):
    """ Enter a list of solvents and a corresponding list of concentrations and the directory of the PSPs;
    Constructs the final solvent object (need to run for cation and anions separately)
    """
    solvents = []
    if len(solvlist) != len(conclist):
        print("MISMATCHING SOLVENT/CONCENTRATION LIST")
        # sys.exit()
    for solvent, solconc in zip(solvlist, conclist):
        if solvent == 'H3O+':
            solvents.append(
                populate_solvent_specie(
                    solvent,
                    solconc,
                    file='H3O+.chuev.MOL'))
        else:
            solvents.append(
                populate_solvent_specie(
                    solvent,
                    solconc,
                    file=get_molfile(
                        solvent,
                        pspdir)))
    return solvents
    # else:
    #cationlist = ['H3O+']
    #cconclist  = [options.solvent_concentration_cation]
    #cations.append(populate_solvent_specie('H3O+',options.solvent_concentration_cation, file='H3O+.chuev.MOL'))
    #anions.append(populate_solvent_specie('Cl-', options.solvent_concentration_anion, file='Cl-.aq.MOL'))
#############################################################################
