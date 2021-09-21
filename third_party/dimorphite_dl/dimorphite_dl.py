# Copyright 2018 Jacob D. Durrant
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script identifies and enumerates the possible protonation sites of SMILES
strings.
"""

from __future__ import print_function
import copy
import os
import argparse
import sys

try:
    # Python2
    from StringIO import StringIO
except ImportError:
    # Python3
    from io import StringIO

# Always let the user know a help file is available.
# print("\nFor help, use: python dimorphite_dl.py --help")

# And always report citation information.
# print("\nIf you use Dimorphite-DL in your research, please cite:")
# print("Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An")
# print("open-source program for enumerating the ionization states of drug-like small")
# print("molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.\n")

site_sub_str = '''
Nitro	[C,c,N,n,O,o:1]-[NX3:2](=[O:3])-[O:4]-[H]	3	-1000	0			
Sulfate	[SX4:1](=[O:2])(=[O:3])([O:4]-[C,c,N,n:5])-[OX2:6]-[H]	5	-2.36	1.304804309			
Sulfonate	[SX4:1](=[O:2])(=[O:3])(-[C,c,N,n:4])-[OX2:5]-[H]	4	-1.818461538	1.408621348			
Thioic_acid	[C,c,N,n:1](=[O,S:2])-[SX2,OX2:3]-[H]	2	0.678267	1.497048764			
Sulfinic_acid	[SX3:1](=[O:2])-[O:3]-[H]	2	1.793333333	0.437207045			
Phosphonate	[PX4:1](=[O:2])(-[OX2:3]-[H])(-[C,c,N,n:4])-[OX2:5]-[H]	2	1.883571429	0.592599982	5	7.247254902	0.851147645
Phosphonate_ester	[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[C,c,N,n,F,Cl,Br,I:5])-[OX2:6]-[H]	5	2.0868	0.450302861			
Phosphate	[PX4:1](=[O:2])(-[OX2:3]-[H])(-[O+0:4])-[OX2:5]-[H]	2	2.41826087	1.109117799	5	6.5055	0.951278779
*Imide	[F,Cl,Br,S,s,P,p:1][#6:2][CX3:3](=[O,S:4])-[NX3+0:5]([CX3:6]=[O,S:7])-[H]	4	2.466666667	1.484362939			
Phosphate_diester	[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[O+0:5]-[C,c,N,n,F,Cl,Br,I:4])-[OX2:6]-[H]	6	2.728043478	2.543744886			
Phosphinic_acid	[PX4:1](=[O:2])(-[C,c,N,n,F,Cl,Br,I:3])(-[C,c,N,n,F,Cl,Br,I:4])-[OX2:5]-[H]	4	2.9745	0.686788675			
Carboxyl	[C:1](=[O:2])-[O:3]-[H]	2	3.456652972	1.287142089			
Phenyl_carboxyl	[c,n,o:1]-[C:2](=[O:3])-[O:4]-[H]	3	3.463441968	1.251805441			
*Amide_electronegative	[C:1](=[O:2])-[N:3](-[Br,Cl,I,F,S,O,N,P:4])-[H]	2	3.4896	2.688124315			
O=C-C=C-OH	[O:1]=[C;R:2]-[C;R:3]=[C;R:4]-[O:5]-[H]	4	3.554	0.803339459			
Anilines_primary	[c:1]-[NX3+0:2]([H:3])[H:4]	1	3.899298673	2.068768504			
Primary_hydroxyl_amine	[C,c:1]-[O:2]-[NH2:3]	2	4.035714286	0.846381654			
Anilines_tertiary	[c:1]-[NX3+0:2]([!H:3])[!H:4]	1	4.16690685	2.005865736			
Anilines_secondary	[c:1]-[NX3+0:2]([H:3])[!H:4]	1	4.335408163	2.176884202			
Aromatic_nitrogen_unprotonated	[n+0&H0:1]	0	4.353544124	2.071407266			
*Azide	[N+0:1]=[N+:2]=[N+0:3]-[H]	2	4.65	0.070710678			
Phenyl_Thiol	[c,n:1]-[SX2:2]-[H]	1	4.978235294	2.613700048			
*Ringed_imide1	[O,S:1]=[C;R:2]([$([#8]),$([#7]),$([#16]),$([#6][Cl]),$([#6]F),$([#6][Br]):3])-[N;R:4]([C;R:5]=[O,S:6])-[H]	3	6.4525	0.555562778			
Phenol	[c,n,o:1]-[O:2]-[H]	1	7.065359867	3.277356122			
*Aromatic_nitrogen_protonated	[n:1]-[H]	0	7.17	2.946023955			
*Sulfonamide	[SX4:1](=[O:2])(=[O:3])-[NX3+0:4]-[H]	3	7.916032609	1.984212132			
Amines_primary_secondary_tertiary	[C:1]-[NX3+0:2]	1	8.159107682	2.518359745			
*Ringed_imide2	[O,S:1]=[C;R:2]-[N;R:3]([C;R:4]=[O,S:5])-[H]	2	8.681666667	1.865777998			
Peroxide1	[O:1]([$(C=O),$(C[Cl]),$(CF),$(C[Br]),$(CC#N):2])-[O:3]-[H]	2	8.738888889	0.756259284			
Vinyl_alcohol	[C:1]=[C:2]-[O:3]-[H]	2	8.871850714	1.660200255			
Thiol	[C,N:1]-[SX2:2]-[H]	1	9.124482759	1.331796816			
N-hydroxyamide	[C:1](=[O:2])-[N:3]-[O:4]-[H]	3	9.301904762	1.218189719			
AmidineGuanidine2	[C:1](-[N:2])=[NX2+0:3]	2	10.03553846	2.131282647			
*Imide2	[O,S:1]=[CX3:2]-[NX3+0:3]([CX3:4]=[O,S:5])-[H]	2	10.23	1.119821414			
Peroxide2	[C:1]-[O:2]-[O:3]-[H]	2	11.97823529	0.86976459			
*Amide	[C:1](=[O:2])-[N:3]-[H]	2	12.00611111	4.512491341			
AmidineGuanidine1	[N:1]-[C:2](-[N:3])=[NX2:4]-[H:5]	3	12.02533333	1.594104615			
*Indole_pyrrole	[c;R:1]1[c;R:2][c;R:3][c;R:4][n;R:5]1[H]	4	14.52875	4.067024916			
Alcohol	[C:1]-[O:2]-[H]	1	14.78038462	2.546464971			
'''

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
except:
    pass


class UtilFuncs:
    """A namespace to store functions for manipulating mol objects. To keep
    things organized."""

    @staticmethod
    def neutralize_mol(mol):
        """All molecules should be neuralized to the extent possible. The user
        should not be allowed to specify the valence of the atoms in most cases.

        :param rdkit.Chem.rdchem.Mol mol: The rdkit Mol objet to be neutralized.
        :return: The neutralized Mol object.
        """

        # Get the reaction data
        rxn_data = [
            ['[Ov1-1:1]', '[Ov2+0:1]-[H]'],  # To handle O- bonded to only one atom (add hydrogen).
            ['[#7v4+1:1]-[H]', '[#7v3+0:1]'],  # To handle N+ bonded to a hydrogen (remove hydrogen).
            ['[Ov2-:1]', '[Ov2+0:1]'],  # To handle O- bonded to two atoms. Should not be Negative.
            ['[#7v3+1:1]', '[#7v3+0:1]'],  # To handle N+ bonded to three atoms. Should not be positive.
            ['[#7v2-1:1]', '[#7+0:1]-[H]'],  # To handle N- Bonded to two atoms. Add hydrogen.
            # ['[N:1]=[N+0:2]=[N:3]-[H]', '[N:1]=[N+1:2]=[N+0:3]-[H]'],  # To handle bad azide. Must be
            # protonated. (Now handled
            # elsewhere, before SMILES
            # converted to Mol object.)
            ['[H]-[N:1]-[N:2]#[N:3]', '[N:1]=[N+1:2]=[N:3]-[H]']  # To handle bad azide. R-N-N#N should
            # be R-N=[N+]=N
        ]

        # Add substructures and reactions (initially none)
        for i, rxn_datum in enumerate(rxn_data):
            rxn_data[i].append(Chem.MolFromSmarts(rxn_datum[0]))
            rxn_data[i].append(None)

        # Add hydrogens (respects valence, so incomplete).
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)

        while True:  # Keep going until all these issues have been resolved.
            current_rxn = None  # The reaction to perform.
            current_rxn_str = None

            for i, rxn_datum in enumerate(rxn_data):
                reactant_smarts, product_smarts, substruct_match_mol, rxn_placeholder = rxn_datum
                if mol.HasSubstructMatch(substruct_match_mol):
                    if rxn_placeholder is None:
                        current_rxn_str = reactant_smarts + '>>' + product_smarts
                        current_rxn = AllChem.ReactionFromSmarts(current_rxn_str)
                        rxn_data[i][3] = current_rxn  # Update the placeholder.
                    else:
                        current_rxn = rxn_data[i][3]
                    break

            # Perform the reaction if necessary
            if current_rxn is None:  # No reaction left, so break out of while loop.
                break
            else:
                mol = current_rxn.RunReactants((mol,))[0][0]
                mol.UpdatePropertyCache(strict=False)  # Update valences

        # The mols have been altered from the reactions described above, we
        # need to resanitize them. Make sure aromatic rings are shown as such
        # This catches all RDKit Errors. without the catchError and
        # sanitizeOps the Chem.SanitizeMol can crash the program.
        sanitize_string = Chem.SanitizeMol(
            mol,
            sanitizeOps=rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ALL,
            catchErrors=True
        )

        return mol if sanitize_string.name == "SANITIZE_NONE" else None

    @staticmethod
    def convert_smiles_str_to_mol(smiles_str):
        """Given a SMILES string, check that it is actually a string and not a
        None. Then try to convert it to an RDKit Mol Object.

        :param string smiles_str: The SMILES string.
        :return: A rdkit.Chem.rdchem.Mol object, or None if it is the wrong type or
            if it fails to convert to a Mol Obj
        """

        # Check that there are no type errors, ie Nones or non-string A
        # non-string type will cause RDKit to hard crash
        if smiles_str is None or type(smiles_str) is not str:
            return None

        # Try to fix azides here. They are just tricky to deal with.
        smiles_str = smiles_str.replace("N=N=N", "N=[N+]=N")
        smiles_str = smiles_str.replace("NN#N", "N=[N+]=N")

        # Now convert to a mol object. Note the trick that is necessary to
        # capture RDKit error/warning messages. See
        # https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
        stderr_fileno = sys.stderr.fileno()
        stderr_save = os.dup(stderr_fileno)
        stderr_pipe = os.pipe()
        os.dup2(stderr_pipe[1], stderr_fileno)
        os.close(stderr_pipe[1])

        mol = Chem.MolFromSmiles(smiles_str)

        os.close(stderr_fileno)
        os.close(stderr_pipe[0])
        os.dup2(stderr_save, stderr_fileno)
        os.close(stderr_save)

        # Check that there are None type errors Chem.MolFromSmiles has
        # sanitize on which means if there is even a small error in the SMILES
        # (kekulize, nitrogen charge...) then mol=None. ie.
        # Chem.MolFromSmiles("C[N]=[N]=[N]") = None this is an example of an
        # nitrogen charge error. It is cased in a try statement to be overly
        # cautious.
        return None if mol is None else mol

    @staticmethod
    def eprint(*args, **kwargs):
        """Error messages should be printed to STDERR. See
        https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python"""

        print(*args, file=sys.stderr, **kwargs)


class Protonate(object):
    """A generator class for protonating SMILES strings, one at a time."""

    def __init__(self, min_ph, max_ph, pka_precision, label_states):

        self.subs = ProtSubstructFuncs.load_protonation_substructs_calc_state_for_ph(
            min_ph, max_ph, pka_precision
        )
        self.label_states = label_states

    def run(self, smiles):
        orig_smi = smiles
        sites, mol_used_to_idx_sites = ProtSubstructFuncs.get_prot_sites_and_target_states(orig_smi, self.subs)

        new_mols = []
        if (len(sites) > 0):
            for site in sites:
                new_mols.append(ProtSubstructFuncs.protonate_site([mol_used_to_idx_sites], site)[0])
        else:
            # Deprotonate the mols (because protonate_site never called to do
            # it).
            mol_used_to_idx_sites = Chem.RemoveHs(mol_used_to_idx_sites)
            new_mols = [mol_used_to_idx_sites]
            sites = [[]]

        # In some cases, the script might generate redundant molecules.
        # Phosphonates, when the pH is between the two pKa values and the
        # stdev value is big enough, for example, will generate two identical
        # BOTH states. Let's remove this redundancy.
        new_smis = [
            Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) for m in new_mols
        ]

        # Sometimes Dimorphite-DL generates molecules that aren't actually
        # possible. Simply convert these to mol objects to eliminate the bad
        # ones (that are None).
        new_smis_list = []
        states = []
        for idx, s in enumerate(new_smis):
            if UtilFuncs.convert_smiles_str_to_mol(s) is not None:
                new_smis_list.append(s)
                states.append(sites[idx])

        # If there are no smi left, return the input one at the very least.
        # All generated forms have apparently been judged
        # inappropriate/malformed.
        return orig_smi, new_smis, states


class ProtSubstructFuncs:
    """A namespace to store functions for loading the substructures that can
    be protonated. To keep things organized."""

    @staticmethod
    def load_protonation_substructs_calc_state_for_ph(min_ph=6.4, max_ph=8.4, pka_std_range=1):
        """A pre-calculated list of R-groups with protonation sites, with their
        likely pKa bins.

        :param float min_ph:  The lower bound on the pH range, defaults to 6.4.
        :param float max_ph:  The upper bound on the pH range, defaults to 8.4.
        :param pka_std_range: Basically the precision (stdev from predicted pKa to
                            consider), defaults to 1.
        :return: A dict of the protonation substructions for the specified pH
                range.
        """

        subs = []
        pwd = os.path.dirname(os.path.realpath(__file__))

        # site_structures_file = "{}/{}".format(pwd, "site_substructures.smarts")
        # with open(site_structures_file, 'r') as substruct:
        substruct = site_sub_str
        for line in substruct.split('\n'):
            line = line.strip()
            sub = {}
            if line is not "":
                splits = line.split()
                sub["name"] = splits[0]
                sub["smart"] = splits[1]
                sub["mol"] = Chem.MolFromSmarts(sub["smart"])

                pka_ranges = [splits[i:i + 3] for i in range(2, len(splits) - 1, 3)]

                prot = []
                for pka_range in pka_ranges:
                    site = pka_range[0]
                    std = float(pka_range[2]) * pka_std_range
                    mean = float(pka_range[1])
                    protonation_state = ProtSubstructFuncs.define_protonation_state(
                        mean, std, min_ph, max_ph
                    )

                    prot.append([site, protonation_state])

                sub["prot_states_for_pH"] = prot
                subs.append(sub)
        return subs

    @staticmethod
    def define_protonation_state(mean, std, min_ph, max_ph):
        """Updates the substructure definitions to include the protonation state
        based on the user-given pH range. The size of the pKa range is also based
        on the number of standard deviations to be considered by the user param.

        :param float mean:   The mean pKa.
        :param float std:    The precision (stdev).
        :param float min_ph: The min pH of the range.
        :param float max_ph: The max pH of the range.
        :return: A string describing the protonation state.
        """

        min_pka = mean - std
        max_pka = mean + std

        # This needs to be reassigned, and 'ERROR' should never make it past
        # the next set of checks.
        if min_pka <= max_ph and min_ph <= max_pka:
            protonation_state = 'BOTH'
        elif mean > max_ph:
            protonation_state = 'PROTONATED'
        else:
            protonation_state = 'DEPROTONATED'

        return protonation_state

    @staticmethod
    def get_prot_sites_and_target_states(smi, subs):
        """For a single molecule, find all possible matches in the protonation
        R-group list, subs. Items that are higher on the list will be matched
        first, to the exclusion of later items.

        :param string smi: A SMILES string.
        :param list subs: Substructure information.
        :return: A list of protonation sites (atom index), pKa bin.
            ('PROTONATED', 'BOTH', or  'DEPROTONATED'), and reaction name.
            Also, the mol object that was used to generate the atom index.
        """

        # Convert the Smiles string (smi) to an RDKit Mol Obj
        mol_used_to_idx_sites = UtilFuncs.convert_smiles_str_to_mol(smi)

        # Check Conversion worked
        if mol_used_to_idx_sites is None:
            UtilFuncs.eprint("ERROR:   ", smi)
            return []

        # Try to Add hydrogens. if failed return []
        try:
            mol_used_to_idx_sites = Chem.AddHs(mol_used_to_idx_sites)
        except:
            UtilFuncs.eprint("ERROR:   ", smi)
            return []

        # Check adding Hs worked
        if mol_used_to_idx_sites is None:
            UtilFuncs.eprint("ERROR:   ", smi)
            return []

        ProtectUnprotectFuncs.unprotect_molecule(mol_used_to_idx_sites)
        protonation_sites = []

        for item in subs:
            smart = item["mol"]
            if mol_used_to_idx_sites.HasSubstructMatch(smart):
                matches = ProtectUnprotectFuncs.get_unprotected_matches(
                    mol_used_to_idx_sites, smart
                )
                prot = item["prot_states_for_pH"]
                for match in matches:
                    # We want to move the site from being relative to the
                    # substructure, to the index on the main molecule.
                    for site in prot:
                        proton = int(site[0])
                        category = site[1]
                        new_site = (match[proton], category, item["name"])

                        if not new_site in protonation_sites:
                            # Because sites must be unique.
                            protonation_sites.append(new_site)

                    ProtectUnprotectFuncs.protect_molecule(
                        mol_used_to_idx_sites, match
                    )

        return protonation_sites, mol_used_to_idx_sites

    @staticmethod
    def protonate_site(mols, site):
        """Given a list of molecule objects, we protonate the site.

        :param list mols:  The list of molecule objects.
        :param tuple site: Information about the protonation site.
                           (idx, target_prot_state, prot_site_name)
        :return: A list of the appropriately protonated molecule objects.
        """

        # Decouple the atom index and its target protonation state from the
        # site tuple
        idx, target_prot_state, prot_site_name = site

        state_to_charge = {"DEPROTONATED": [-1],
                           "PROTONATED": [0],
                           "BOTH": [-1, 0]}

        charges = state_to_charge[target_prot_state]

        # Now make the actual smiles match the target protonation state.
        output_mols = ProtSubstructFuncs.set_protonation_charge(
            mols, idx, charges, prot_site_name
        )

        return output_mols

    @staticmethod
    def set_protonation_charge(mols, idx, charges, prot_site_name):
        """Sets the atomic charge on a particular site for a set of SMILES.

        :param list mols:                  A list of the input molecule
                                           objects.
        :param int idx:                    The index of the atom to consider.
        :param list charges:               A list of the charges (ints) to
                                           assign at this site.
        :param string prot_site_name:      The name of the protonation site.
        :return: A list of the processed (protonated/deprotonated) molecule
                 objects.
        """

        # Sets up the output list and the Nitrogen charge
        output = []

        for charge in charges:
            # The charge for Nitrogens is 1 higher than others (i.e.,
            # protonated state is positively charged).
            # TODO: why????
            nitro_charge = charge + 1

            # But there are a few nitrogen moieties where the acidic group is
            # the neutral one. Amides are a good example. I gave some thought
            # re. how to best flag these. I decided that those
            # nitrogen-containing moieties where the acidic group is neutral
            # (rather than positively charged) will have "*" in the name.
            if "*" in prot_site_name:
                nitro_charge = nitro_charge - 1  # Undo what was done previously.

            for mol in mols:
                # Make a copy of the molecule.
                mol_copy = copy.deepcopy(mol)

                # Remove hydrogen atoms.
                # print("DDD", Chem.MolToSmiles(mol_copy))
                try:
                    mol_copy = Chem.RemoveHs(mol_copy)
                except:
                    UtilFuncs.eprint("WARNING: Skipping poorly formed SMILES string: " + Chem.MolToSmiles(mol_copy))
                    continue

                atom = mol_copy.GetAtomWithIdx(idx)

                # Assign the protonation charge, with special care for
                # nitrogens
                element = atom.GetAtomicNum()
                if element == 7:
                    atom.SetFormalCharge(nitro_charge)
                else:
                    atom.SetFormalCharge(charge)

                # Deprotonating protonated aromatic nitrogen gives [nH-]. Change this
                # to [n-].
                if "[nH-]" in Chem.MolToSmiles(mol_copy):
                    atom.SetNumExplicitHs(0)

                mol_copy.UpdatePropertyCache()

                output.append(mol_copy)

        return output


class ProtectUnprotectFuncs:
    """A namespace for storing functions that are useful for protecting and
    unprotecting molecules. To keep things organized. We need to identify and
    mark groups that have been matched with a substructure."""

    @staticmethod
    def unprotect_molecule(mol):
        """Sets the protected property on all atoms to 0. This also creates the
        property for new molecules.

        :param rdkit.Chem.rdchem.Mol mol: The rdkit Mol object.
        :type mol: The rdkit Mol object with atoms unprotected.
        """

        for atom in mol.GetAtoms():
            atom.SetProp('_protected', '0')

    @staticmethod
    def protect_molecule(mol, match):
        """Given a 'match', a list of molecules idx's, we set the protected status
        of each atom to 1. This will prevent any matches using that atom in the
        future.

        :param rdkit.Chem.rdchem.Mol mol: The rdkit Mol object to protect.
        :param list match: A list of molecule idx's.
        """

        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            atom.SetProp('_protected', '1')

    @staticmethod
    def get_unprotected_matches(mol, substruct):
        """Finds substructure matches with atoms that have not been protected.
        Returns list of matches, each match a list of atom idxs.

        :param rdkit.Chem.rdchem.Mol mol: The Mol object to consider.
        :param string substruct: The SMARTS string of the substructure ot match.
        :return: A list of the matches. Each match is itself a list of atom idxs.
        """

        matches = mol.GetSubstructMatches(substruct)
        unprotected_matches = []
        for match in matches:
            if ProtectUnprotectFuncs.is_match_unprotected(mol, match):
                unprotected_matches.append(match)
        return unprotected_matches

    @staticmethod
    def is_match_unprotected(mol, match):
        """Checks a molecule to see if the substructure match contains any
        protected atoms.

        :param rdkit.Chem.rdchem.Mol mol: The Mol object to check.
        :param list match: The match to check.
        :return: A boolean, whether the match is present or not.
        """

        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            protected = atom.GetProp("_protected")
            if protected == "1":
                return False
        return True
