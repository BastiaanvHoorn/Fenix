import copy
import typing
import numpy as np

import veloxchem as vlx
import openff.interchange.exceptions as ffexceptions
import openff.toolkit as fftool
import openff.units as ffunits
import openff.interchange.components._packmol as packmol
from openmmforcefields.generators import GAFFTemplateGenerator
import openmm as mm
import openmm.app as mmapp
import openmm.unit as mmunit

from EVB.reaction_space_builder import Reaction_space_builder


solvents: dict = {
    "Benzene": {
        "Smiles": "C1=CC=CC=C1",
        "Density": 876,
    },
    "Water": {"Smiles": "[OH2]", "Density": 997},
    "DMF": {"Smiles": "CN(C)C=O", "Density": 944},
    "Methanol": {"Smiles": "CO", "Density": 791},
    "Chloroform": {"Smiles": "ClC(Cl)Cl", "Density": 1494},
    "Acetonitrile": {"Smiles": "CC#N", "Density": 786},
    "Acetone": {"Smiles": "CC(=O)C", "Density": 784},
    # "Dichloromethane":"ClCCl",
    # "Toluene":"Cc1ccccc1",
    # "Benzene":"c1ccccc1",
}


class System_builder:
    # Generates the OpenMM system and topology for the total system, calls the active system builder

    def __init__(
        self,
        reactant: vlx.ForceFieldGenerator,
        product: vlx.ForceFieldGenerator,
        Lambda: typing.List[float],
        temperature: float = 300,
        pressure: float = 1,
        NPT: bool = False,
    ):
        self.reactant: vlx.ForceFieldGenerator = reactant
        self.product: vlx.ForceFieldGenerator = product
        self.systems: typing.Dict = {}
        self.Lambda: typing.List[float] = Lambda
        self.temperature: float = temperature
        self.pressure: float = pressure
        self.NPT: bool = NPT

    @staticmethod
    def CNT_xyz_to_ff_molecule(filename) -> fftool.Molecule:
        with open(filename, "r") as f:
            lines = f.read().split("\n")[1:-1]
        lines = [line.split() for line in lines]
        index = [int(line[0]) for line in lines]
        element = [line[1] for line in lines]
        coords = [[float(line[2]), float(line[3]), float(line[4])] for line in lines]
        connections = [line[6:] for line in lines]

        atoms = []
        for i in index:
            atoms.append(
                {
                    "name": f"{i}{element[i-1]}",
                    "formal_charge": 0,
                    "is_aromatic": False,
                    "atomic_number": mmapp.element.Element.getBySymbol(
                        element[i - 1]
                    ).atomic_number,
                }
            )

        bonds = []
        bonds_set = set()

        for i, connection in enumerate(connections):
            for j in connection:
                ordered_tuple = tuple(sorted([index[i] - 1, int(j) - 1]))
                if ordered_tuple not in bonds_set:
                    bonds_set.add(ordered_tuple)
                    bonds.append(
                        {
                            "atom1": index[i] - 1,
                            "atom2": int(j) - 1,
                            "bond_order": 1,
                            "is_aromatic": False,
                        }
                    )

        ff_dict = {
            "name": "CNT",
            "atoms": atoms,
            "bonds": bonds,
            "properties": {},
            "conformers": None,
            "partial_charges": None,
            "partial_charges_unit": None,
            "hierarchy_schemes": {},
        }
        ff_molecule = fftool.topology.Molecule.from_dict(ff_dict)
        ff_molecule.add_conformer(ffunits.Quantity(coords, ffunits.unit.angstroms))
        return ff_molecule

    @staticmethod
    def convert_vlx_to_ff_molecule(
        molecule: vlx.ForceFieldGenerator,
    ) -> fftool.Molecule:
        ff_dict = {
            "name": "Reactant",
            "atoms": [
                {
                    "name": atom,
                    "formal_charge": 0,
                    "is_aromatic": False,
                    "atomic_number": mmapp.element.Element.getBySymbol(
                        atom
                    ).atomic_number,
                }
                for atom in molecule.molecule.get_labels()
            ],
            "bonds": [
                {
                    "atom1": key[0],
                    "atom2": key[1],
                    "bond_order": 1,
                    "is_aromatic": False,
                }
                for key in molecule.bonds.keys()
            ],
            "properties": {},
            "conformers": None,
            "partial_charges": None,
            "partial_charges_unit": None,
            "hierarchy_schemes": {},
        }

        no_charge_dict = copy.deepcopy(ff_dict)

        total_charge = 0
        for atom in molecule.atoms.values():
            total_charge += atom["charge"]
        total_charge = round(total_charge)
        formal_charges = [0] * len(molecule.atoms)

        for i, atom in enumerate(molecule.atoms.values()):
            formal_charges[i] = round(atom["charge"])

        try:
            assert sum(formal_charges) == total_charge

            for i, charge in enumerate(formal_charges):
                if charge != 0:
                    print(f"The formal charge for atom {i} is set to {charge}")
                ff_dict["atoms"][i]["formal_charge"] = charge

            ff_molecule = fftool.topology.Molecule.from_dict(ff_dict)
        except AssertionError:
            print(
                f"WARNING: Total charge of the molecule is {total_charge}, but the sum of the formal charges is {sum(formal_charges)}"
            )
            print(
                "WARNING: Solvating molecule with all formal charges set to 0. You can (probably) safely ignore this if your system is not ionic."
            )
            ff_molecule = fftool.topology.Molecule.from_dict(no_charge_dict)

        ff_molecule.add_conformer(
            ffunits.Quantity(
                molecule.molecule.get_coordinates_in_angstrom(), ffunits.unit.angstroms
            )
        )
        return ff_molecule

    # Packs the reaction in a box with the solvent, and sets total_mm_sys and total_mm_top to the OpenMM system and topology of the solvatned system
    # the active space has residue name REA, and is added to the nonbondedforce of the solvent, but has no internal interactions turned on
    def build_solvated_systems(
        self,
        smiles: list[str],
        molecules_count: list[int],
        density,
        nonbonded_cutoff: float = None,  # type:ignore
        nonbonded_switch_factor=0.9,
        neutralise_charge=0,
        periodic_box=True,
        # CNT_xyz: str = None,  # type:ignore #file location of the cnt xyz file
        # CNT_posres_force_k: float = 500000,
        # CNT_centering_force_k: float = 150,
        # CNT_centering_force_rmax: float = 0.5,  # in nm
        # CNT_centering_force_Z_scaling: float = 0.1,
        # CNT_reactant_packing_height: float = None,  # type: ignore  # in nm
        # CNT_size: float = 10,  # in nm
    ):  # -> tuple[mmapp.Topology,mm.System,mmapp.Topology,mm.System]:

        # Generate all the solvent molecules
        if neutralise_charge > 0:
            charge = self.reactant.molecule.get_charge()
            Na_count = 0
            Cl_count = 0
            if charge < 0:
                Na_count = -charge + neutralise_charge - 1
                Cl_count = neutralise_charge - 1
            if charge > 0:
                Na_count = neutralise_charge - 1
                Cl_count = charge + neutralise_charge - 1
            if Na_count > 0:
                smiles.append("[Na+]")
                molecules_count.append(int(Na_count))
            if Cl_count > 0:
                molecules_count.append(int(charge))
                smiles.append("[Cl-]")

        ff_molecules = [fftool.Molecule.from_smiles(smi) for smi in smiles]

        gaff = GAFFTemplateGenerator(
            molecules=ff_molecules
        )  # Use the ff_molecules here so that after that the reactant can be added

        # Convert the vlx reactant to ff reactant and add it to the list of molecules
        ff_reactant = self.convert_vlx_to_ff_molecule(self.reactant)
        ff_reactant.name = "Reaction"
        ff_molecules.insert(0, ff_reactant)
        molecules_count.insert(0, 1)
        # ff_molecules.append(ff_reactant)
        # molecules_count.append(1)

        # packing_vectors = None
        # if not CNT_xyz is None:
        #     CNT_size = CNT_size * 10  # convert to angstrom
        #     ff_cnt = self.CNT_xyz_to_ff_molecule(CNT_xyz)
        #     ff_cnt.name = "CNT"
        #     ff_molecules.insert(0, ff_cnt)
        #     molecules_count.insert(0, 1)
        print("Starting packing system")
        # packing = True
        # scalings = 0
        total_topology: fftool.Topology = None  # type:ignore
        # while packing:
        #     if not CNT_xyz is None:

        #     packing_vectors = [None] * len(ff_molecules)
        #     # packmol inputs are in angstrom
        #     CNT_packing_vector = [0.5, 0.5, 1, CNT_size, CNT_size, 2]
        #     packing_vectors[0] = CNT_packing_vector  # type:ignore
        #     print(f"CNT packing vector = {CNT_packing_vector}")

        #     min_dist = CNT_size * 0.5 - CNT_centering_force_rmax * 9
        #     max_dist = CNT_size * 0.5 + CNT_centering_force_rmax * 9
        #     if CNT_reactant_packing_height is not None:
        #         max_height = CNT_reactant_packing_height * 10
        #     else:
        #         max_height = CNT_centering_force_rmax * 9 + 2
        #     rea_packing_vector = [
        #         min_dist,
        #         min_dist,
        #         2,
        #         max_dist,
        #         max_dist,
        #         max_height,
        #     ]
        #     packing_vectors[1] = rea_packing_vector  # type:ignore
        #     print(f"Reactant packing vector = {rea_packing_vector}")

        # # Pack the box
        # try:
        total_topology: fftool.Topology = packmol.pack_box(
            molecules=ff_molecules,
            number_of_copies=molecules_count,
            mass_density=density * ffunits.unit.kilogram / ffunits.unit.meter**3,
            box_shape=packmol.UNIT_CUBE,
            # packing_vectors=packing_vectors,
        )
        #     packing = False
        # except (
        #     ffexceptions.PACKMOLRuntimeError,
        #     ffexceptions.PACKMOLValueError,
        # ) as e:
        #     if CNT_xyz is None:
        #         raise e
        #     if scalings < 3:
        #         print(
        #             f"Encountered packmol exception, scaling up CNT_size from {CNT_size} to {round(CNT_size*1.2,1)} and retrying packing"
        #         )
        #         CNT_size = round(CNT_size * 1.2, 1)
        #         scalings += 1
        #     else:
        #         print("Encountered too many packmol exceptions, exiting")
        #         raise e

        print("Finished packing system")
        total_topology.to_file("Initial_packing.pdb")

        # Seperate the reactant and solvent
        solvent_dict = total_topology.to_dict()
        solvent_dict["molecules"] = [
            mol
            for mol in solvent_dict["molecules"]
            if not mol["name"] == "Reaction" and not mol["name"] == "CNT"
        ]
        solvent_topology: fftool.Topology = fftool.Topology.from_dict(solvent_dict)
        self.topology = solvent_topology.to_openmm()
        self.positions = (
            solvent_topology.get_positions().to_openmm().value_in_unit(mmunit.nanometer)
        )

        reactant_dict = total_topology.to_dict()
        reactant_dict["molecules"] = [
            mol for mol in reactant_dict["molecules"] if mol["name"] == "Reaction"
        ]
        reactant_topology: fftool.Topology = fftool.Topology.from_dict(reactant_dict)
        reactant_mm_top = reactant_topology.to_openmm()
        reactant_positions = (
            reactant_topology.get_positions()
            .to_openmm()
            .value_in_unit(mmunit.nanometer)
        )

        self.box_size = self.topology.getPeriodicBoxVectors()[0, 0].value_in_unit(
            mmunit.nanometer
        )

        # # Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
        forcefield = mmapp.ForceField("amber14/tip3pfb.xml")
        # Register the GAFF template generator
        forcefield.registerTemplateGenerator(gaff.generator)
        print("Creating system")
        if periodic_box:
            if nonbonded_cutoff is None:
                nonbonded_cutoff = min(1.5, self.box_size / 2)
                print(
                    f"No nonbonded cutoff given, setting to {nonbonded_cutoff} nm (half of the box size with a maximum of 1.5 nm)"
                )
            if nonbonded_cutoff * 2 > self.box_size:
                print(
                    f"WARNING: nonbonded cutoff ({nonbonded_cutoff} nm) is larger than half the box size ({self.box_size} /2 nm). OpenMM will likely complain during the FEP run"
                )
            if nonbonded_switch_factor > 1:
                nonbonded_switch_factor = 1
                print(
                    f"WARNING: nonbonded switch factor ({nonbonded_switch_factor}) is larger than 1. Setting to 1"
                )
            solvent_mm_sys = forcefield.createSystem(
                self.topology,
                nonbondedMethod=mmapp.PME,  # type: ignore
                switchDistance=nonbonded_cutoff * nonbonded_switch_factor,
                nonbondedCutoff=nonbonded_cutoff,
            )
        else:
            solvent_mm_sys = forcefield.createSystem(
                self.topology, nonbondedMethod=mmapp.NoCutoff
            )
        print("Finished creating system")

        # if not CNT_xyz is None:
        #     CNT_dict = total_topology.to_dict()
        #     CNT_dict["molecules"] = [
        #         mol for mol in CNT_dict["molecules"] if mol["name"] == "CNT"
        #     ]
        #     CNT_topology: fftool.Topology = fftool.Topology.from_dict(CNT_dict)
        #     CNT_mm_top = CNT_topology.to_openmm()
        #     CNT_positions = (
        #         CNT_topology.get_positions().to_openmm().value_in_unit(mmunit.nanometer)
        #     )
        #     CNT_atoms = []
        #     CNT_chain = self.topology.addChain()
        #     CNT_residue = self.topology.addResidue(name="CNT", chain=CNT_chain)

        #     solvent_nb_force = [
        #         force
        #         for force in solvent_mm_sys.getForces()
        #         if type(force) == mm.NonbondedForce
        #     ][0]
        #     CNT_posres_force = mm.CustomExternalForce(
        #         f"{CNT_posres_force_k}*periodicdistance(x, y, z, x0, y0, z0)^2"
        #     )
        #     CNT_posres_force.addPerParticleParameter("x0")
        #     CNT_posres_force.addPerParticleParameter("y0")
        #     CNT_posres_force.addPerParticleParameter("z0")
        #     CNT_posres_force.setName("CNT position restraint")
        #     solvent_mm_sys.addForce(CNT_posres_force)

        #     CNT_centering_force = mm.CustomExternalForce(
        #         # todo make sure there is no restraint perpendicular to the CNT
        #         f"{CNT_centering_force_k}*(max(0,periodicdistance(x, y, z*{CNT_centering_force_Z_scaling}, x0, y0, z0*{CNT_centering_force_Z_scaling})-{CNT_centering_force_rmax}))^2"
        #     )
        #     CNT_center = [CNT_size / 2, CNT_size / 2, 2]
        #     CNT_centering_force.addGlobalParameter("x0", CNT_center[0])
        #     CNT_centering_force.addGlobalParameter("y0", CNT_center[1])
        #     CNT_centering_force.addGlobalParameter("z0", CNT_center[2])
        #     CNT_centering_force.setName("CNT centering force")

        #     solvent_mm_sys.addForce(CNT_centering_force)

        #     for i, cnt_atom in enumerate(CNT_mm_top.atoms()):
        #         CNT_atoms.append(
        #             self.topology.addAtom(cnt_atom.name, cnt_atom.element, CNT_residue)
        #         )
        #         solvent_mm_sys.addParticle(cnt_atom.element.mass)
        #         self.positions = np.vstack(
        #             (self.positions, CNT_positions[cnt_atom.index])
        #         )
        #         # add nonbonded force, default gaff2.20 parameters
        #         if cnt_atom.element.symbol == "C":
        #             solvent_nb_force.addParticle(0.0, 0.331521230994383, 0.4133792)
        #         if cnt_atom.element.symbol == "H":
        #             solvent_nb_force.addParticle(0.0, 0.262547852235958, 0.0673624)

        #         # add position restraint
        #         CNT_posres_force.addParticle(CNT_atoms[i].index, list(CNT_positions[i]))
        #         for j in range(len(CNT_atoms) - 1):
        #             solvent_nb_force.addException(
        #                 CNT_atoms[i].index, CNT_atoms[j].index, 0.0, 1.0, 0.0
        #             )

        reaction_atoms = []
        reaction_chain = self.topology.addChain()
        reaction_residue = self.topology.addResidue(name="REA", chain=reaction_chain)
        for i, reactant_atom in enumerate(reactant_mm_top.atoms()):
            reaction_atoms.append(
                self.topology.addAtom(
                    reactant_atom.name, reactant_atom.element, reaction_residue
                )
            )
            solvent_mm_sys.addParticle(reactant_atom.element.mass)
            self.positions = np.vstack((self.positions, reactant_positions[i]))
            # if not CNT_xyz is None:
            #     CNT_centering_force.addParticle(reaction_atoms[-1].index)
        if self.NPT:
            solvent_mm_sys.addForce(
                mm.MonteCarloBarostat(
                    self.pressure * mmunit.bar, self.temperature * mmunit.kelvin
                )
            )

        for l in self.Lambda:
            reaction_particles = []
            system = copy.deepcopy(solvent_mm_sys)

            solvent_nb_force = [
                force
                for force in system.getForces()
                if type(force) == mm.NonbondedForce
            ][0]

            for i, (reactant_atom, product_atom) in enumerate(
                zip(self.reactant.atoms.values(), self.product.atoms.values())
            ):

                charge = (1 - l) * reactant_atom["charge"] + l * product_atom["charge"]
                sigma = (1 - l) * reactant_atom["sigma"] + l * product_atom["sigma"]
                epsilon = (1 - l) * reactant_atom["epsilon"] + l * product_atom[
                    "epsilon"
                ]
                reaction_particles.append(
                    solvent_nb_force.addParticle(charge, sigma, epsilon)
                )

                for j, _ in enumerate(self.reactant.atoms.values()):
                    if j > i:
                        solvent_nb_force.addException(
                            reaction_atoms[i].index,
                            reaction_atoms[j].index,
                            0.0,
                            1.0,
                            0.0,
                        )
            total_charge: float = 0
            for i in reaction_particles:
                charge = solvent_nb_force.getParticleParameters(i)[0]
                total_charge += charge.value_in_unit(mmunit.elementary_charge)
            if not round(total_charge, 5).is_integer():
                print(
                    f"Warning: total charge for lambda {l} is {total_charge} and is not a whole number"
                )
            if l == 0.0:
                self.systems["reactant"] = copy.deepcopy(system)
            if l == 1.0:
                self.systems["product"] = copy.deepcopy(system)
            self.systems[l] = system

        self.reaction_atoms = reaction_atoms

    def build_systems(self):

        system = mm.System()
        self.topology = mmapp.Topology()
        reaction_chain = self.topology.addChain()
        reaction_residue = self.topology.addResidue(name="REA", chain=reaction_chain)
        reaction_atoms = []
        # add all atoms
        elements = self.reactant.molecule.get_labels()
        for i, atom in enumerate(self.reactant.atoms.values()):
            mm_element = mmapp.Element.getBySymbol(elements[i])
            name = f"{elements[i]}{i}"
            reaction_atoms.append(
                self.topology.addAtom(name, mm_element, reaction_residue)
            )
            system.addParticle(mm_element.mass)
        if self.NPT:
            system.addForce(
                mm.MonteCarloBarostat(
                    self.pressure * mmunit.bar, self.temperature * mmunit.kelvin
                )
            )
        self.systems: typing.Dict = {}
        for l in self.Lambda:
            self.systems[l] = copy.deepcopy(system)
            if l == 0.0:
                self.systems["reactant"] = copy.deepcopy(system)
            if l == 1.0:
                self.systems["product"] = copy.deepcopy(system)
        self.reaction_atoms = reaction_atoms
        self.positions = self.reactant.molecule.get_coordinates_in_angstrom() * 0.1
        self.topology: mmapp.Topology = self.topology

    def parameterise_reaction_space(
        self,
        write_xml: bool = False,
        folder: str = "",
        constraints=[],
        soft_core_run=False,
        soft_core_ref=True,
        sc_alpha_lj=0.85,
        sc_alpha_q=0.3,
        sc_sigma_q=1.0,
        sc_power=1 / 6,
        morse_D_default=500,  # kj/mol, default dissociation energy if none is given
        morse_couple=1,  # kj/mol, scaling for the morse potential to emulate a coupling between two overlapping bonded states
        restraint_k=1000,  # kj/mol nm^2, force constant for the position restraints
        restraint_r_default=0.5,  # nm, default position restraint distance if none is given
        restraint_r_offset=0.1,  # nm, distance added to the measured distance in a structure to set the position restraint distance
        coul14=0.833,
        lj14=0.5,
        verbose=True,
    ):
        # create active space builder
        reaction_space_builder = Reaction_space_builder(
            self.reactant,
            self.product,
            self.reaction_atoms,
            constraints,
            sc_alpha_lj,
            sc_alpha_q,
            sc_sigma_q,
            sc_power,
            morse_D_default,
            morse_couple,
            restraint_k,
            restraint_r_default,
            restraint_r_offset,
            coul14,
            lj14,
            verbose,
        )

        for l in self.Lambda:
            reaction_space_builder.add_reaction_forces(
                self.systems[l], l, reference_state=False, soft_core=soft_core_run
            )

            if write_xml:
                with open(
                    f"{folder}/{l:.3f}_sys.xml", mode="w", encoding="utf-8"
                ) as output:
                    output.write(mm.XmlSerializer.serialize(self.systems[l]))

        reaction_space_builder.add_reaction_forces(
            self.systems["reactant"], 0, True, soft_core=soft_core_ref
        )
        for i, force in enumerate(self.systems["reactant"].getForces()):
            if "CNT" in force.getName():
                self.systems["reactant"].removeForce(i)

        reaction_space_builder.add_reaction_forces(
            self.systems["product"], 1, True, soft_core=soft_core_ref
        )
        for i, force in enumerate(self.systems["product"].getForces()):
            if "CNT" in force.getName():
                self.systems["product"].removeForce(i)
        if write_xml:
            with open(f"{folder}/reactant.xml", mode="w", encoding="utf-8") as output:
                output.write(mm.XmlSerializer.serialize(self.systems["reactant"]))
            with open(f"{folder}/product.xml", mode="w", encoding="utf-8") as output:
                output.write(mm.XmlSerializer.serialize(self.systems["product"]))
