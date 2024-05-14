import json
import ast
import os
import shutil
import glob

import numpy as np
import numpy.typing as npt
import veloxchem as vlx
import openmm as mm
import openmm.app as mmapp
from EVB.fep_driver import FEP_driver
from EVB.system_builder import System_builder
from EVB.forcefield_combiner import merge_product_force_fields


class EVB_driver:
    """
    The EVB_driver class represents a driver for performing EVB (Empirical Valence Bond) simulations.

    Usage:
        EVB = EVB_driver(...)
        EVB.build_forcefields(...)
        EVB.systembuilder_setup(EVB.get_lambda(...),...)
        EVB.setup_and_run_FEP(...)

    Attributes:
        data_folder (str): The path to the data folder.
        run_folder (str): The path to the run folder.
        input_folder (str): The path to the input folder.
        forcefield_folder (str): The path to the force field folder.
        reactant (vlx.ForceFieldGenerator): The force field generator for the reactant.
        product (vlx.ForceFieldGenerator): The force field generator for the product.
        FEP (FEP_driver): The FEP (Free Energy Perturbation) driver.
        system_builder (System_builder): The system builder.

    Methods:
        save_charges: Save the given list of charges to a text file.
        load_charges: Load a list of charges from a file.
        str_to_tuple_key: Converts the keys of a dictionary from string to tuple.
        tuple_to_str_key: Converts the keys of a dictionary from tuples to strings.
        save_forcefield: Save the force field data of the force field generator to a JSON file.
        load_forcefield_from_json: Load force field data from a JSON file.
        get_forcefield: Constructs a force field generator from a given xyz file.
        build_forcefields: Builds force fields for the reactant and products.
        get_lambda: Generate an array of lambda values based on the given multiplier.
        systembuilder_setup: Set up the system builder.

    """

    def __init__(
        self,
        input_folder: str = "input_files",
        forcefield_folder: str = "forcefield_data",
    ):
        self.input_folder: str = input_folder
        self.forcefield_folder: str = forcefield_folder

        if not os.path.exists(self.forcefield_folder):
            os.makedirs(self.forcefield_folder)

        self.reactant: vlx.ForceFieldGenerator
        self.product: vlx.ForceFieldGenerator
        self.FEP: FEP_driver
        self.system_builder: System_builder
        self.data_folder: str
        self.run_folder: str

    def save_charges(self, charges: list, filename: str):
        """
        Save the given list of charges to a text file.

        Args:
            charges (list): A list of charges to be saved.
            filename (str): The name of the file to save the charges to.

        Returns:
            None
        """
        with open(
            f"{self.forcefield_folder}/{filename}_charges.txt", "w", encoding="utf-8"
        ) as file:
            for charge in charges:
                file.write(f"{charge}\n")

    def load_charges(self, filename: str) -> list:
        """
        Load a list of charges from a file.

        Args:
            filename (str): The name of the file to load charges from.

        Returns:
            list: A list of charges read from the file.
        """
        with open(
            f"{self.forcefield_folder}/{filename}_charges.txt", "r", encoding="utf-8"
        ) as file:
            charges = []
            for line in file:
                try:
                    charges.append(float(line))
                except ValueError:
                    print(
                        rf"Could not read line {line} from {filename}_charges.txt. Continuing"
                    )
        return charges

    @staticmethod
    def str_to_tuple_key(dictionary: dict) -> dict:
        """
        Converts the keys of a dictionary from string to tuple.

        Args:
            dictionary (dict): The dictionary to convert.

        Returns:
            dict: The dictionary with keys converted to tuple.
        """
        return {ast.literal_eval(key): value for key, value in dictionary.items()}

    @staticmethod
    def tuple_to_str_key(dictionary: dict) -> dict:
        """
        Converts the keys of a dictionary from tuples to strings.

        Args:
            dictionary (dict): The dictionary to be converted.

        Returns:
            dict: The dictionary with string keys.

        """
        return {str(key): value for key, value in dictionary.items()}

    def save_forcefield(self, forcefield: vlx.ForceFieldGenerator, filename: str):
        """
        Save the forcefield data of the forcefieldgenerator to a JSON file, converting all tuples to strings

        Args:
            forcefield (vlx.ForceFieldGenerator): The forcefield object containing the data to be saved.
            filename (str): The name of the file to save the forcefield data to.

        Returns:
            None
        """
        ff_data = {
            "atoms": forcefield.atoms,
            "bonds": EVB_driver.tuple_to_str_key(forcefield.bonds),
            "angles": EVB_driver.tuple_to_str_key(forcefield.angles),
            "dihedrals": EVB_driver.tuple_to_str_key(forcefield.dihedrals),
            "impropers": EVB_driver.tuple_to_str_key(forcefield.impropers),
        }
        with open(
            f"{self.forcefield_folder}/{filename}_ff_data.json", "w", encoding="utf-8"
        ) as file:
            json.dump(ff_data, file, indent=4)

    def load_forcefield_from_json(
        self, forcefield: vlx.ForceFieldGenerator, filename: str
    ) -> vlx.ForceFieldGenerator:
        """
        Load forcefield data from a JSON file.

        Args:
            forcefield (vlx.ForceFieldGenerator): The forcefield object to load the data into.
            filename (str): The name of the JSON file, without _ff_data.json.

        Returns:
            vlx.ForceFieldGenerator: The updated forcefield object with the loaded data.
        """
        with open(
            f"{self.forcefield_folder}/{filename}_ff_data.json", "r", encoding="utf-8"
        ) as file:
            ff_data = json.load(file)

            forcefield.atoms = self.str_to_tuple_key(ff_data["atoms"])
            forcefield.bonds = self.str_to_tuple_key(ff_data["bonds"])
            forcefield.angles = self.str_to_tuple_key(ff_data["angles"])
            forcefield.dihedrals = self.str_to_tuple_key(ff_data["dihedrals"])
            forcefield.impropers = self.str_to_tuple_key(ff_data["impropers"])
        return forcefield

    def get_forcefield(
        self,
        filename: str,
        charge: int,
        reparameterise: bool = True,
        optimise: bool = False,
    ) -> vlx.ForceFieldGenerator:
        """
        Constructs a forcefieldgenerator from a given xyz file. Will load charges and forcefield data if available, otherwise RESP charges are calculated and the forcefield is reparameterised.

        Args:
            filename (str): The name of the input xyz file.
            charge (int): The charge of the molecule.
            reparameterise (bool): Whether to reparameterise the force field if no force field data is found.
            optimize (bool): Whether to optimise the geometry of the molecule.
            hessian (list): The hessian of the molecule.

        Returns:
            vlx.ForceFieldGenerator: The generated or retrieved force field.

        Raises:
            FileNotFoundError: If the force field data file or charges file is not found.

        """
        molecule = vlx.Molecule.read_xyz_file(f"{self.input_folder}/{filename}.xyz")
        if optimise:
            scf_drv = vlx.XtbDriver()
            opt_drv = vlx.OptimizationDriver(scf_drv)
            opt_results = opt_drv.compute(molecule)
            with open(
                f"{self.input_folder}/{filename}_xtb_opt.xyz", "w", encoding="utf-8"
            ) as file:
                file.write(opt_results["final_geometry"])
            molecule = vlx.Molecule.from_xyz_string(opt_results["final_geometry"])
        molecule.set_charge(charge)
        forcefield = vlx.ForceFieldGenerator()
        forcefield.force_field_data = f"./{self.input_folder}/gaff-2.20.dat"
        loading_charges = False
        if os.path.exists(f"{self.forcefield_folder}/{filename}_charges.txt"):
            loading_charges = True
            forcefield.partial_charges = self.load_charges(filename)
            print(
                f"Loading charges from {self.forcefield_folder}/{filename}_charges.txt file, total charge = {sum(forcefield.partial_charges)}"
            )

        forcefield.create_topology(molecule)
        if not loading_charges:
            self.save_charges(forcefield.partial_charges, filename)  # type: ignore

        loading_ff_data = False
        if os.path.exists(f"{self.forcefield_folder}/{filename}_ff_data.json"):
            print(
                f"Loading force field data from {self.forcefield_folder}/{filename}_ff_data.json file"
            )
            loading_ff_data = True
            self.load_forcefield_from_json(forcefield, filename)

        if not loading_ff_data:
            print(
                f"Could not find force field data file {self.forcefield_folder}/{filename}_ff_data.json."
            )
            if reparameterise:
                print("Reparameterising force field.")

                if os.path.exists(f"{self.forcefield_folder}/{filename}_hess.np"):
                    print(
                        f"Found hessian file at {self.forcefield_folder}/{filename}_hess.np, using it to reparameterise."
                    )
                    hessian = np.loadtxt(f"{self.forcefield_folder}/{filename}_hess.np")
                else:
                    print(
                        f"Could not find hessian file at {self.forcefield_folder}/{filename}_hess.np, calculating hessian with xtb and saving it"
                    )
                    scf_drv = vlx.XtbDriver()
                    xtb_hessian_drv = vlx.XtbHessianDriver(scf_drv)
                    xtb_hessian_drv.compute(molecule)
                    hessian = np.copy(xtb_hessian_drv.hessian)  # type: ignore
                    np.savetxt(f"{self.forcefield_folder}/{filename}_hess.np", hessian)
                forcefield.reparameterize(hessian=hessian)

            self.save_forcefield(forcefield, filename)
        else:
            if reparameterise:
                print("Force field data found, not reparameterising.")

        return forcefield

    def summarise_reaction(self):
        """
        Summarises the reaction by printing the bonds that are being broken and formed.

        Returns:
            None
        """
        reactant_bonds = set(self.reactant.bonds)
        product_bonds = set(self.product.bonds)
        formed_bonds = product_bonds - reactant_bonds
        broken_bonds = reactant_bonds - product_bonds
        print(f"{len(broken_bonds)} breaking bonds:")
        if len(broken_bonds) > 0:
            print("ProType, ReaType, ID\t - ProType, ReaType, ID")
        for bond_key in broken_bonds:
            reactant_type0 = self.reactant.atoms[bond_key[0]]["type"]
            product_type0 = self.product.atoms[bond_key[0]]["type"]
            id0 = bond_key[0]
            reactant_type1 = self.reactant.atoms[bond_key[1]]["type"]
            product_type1 = self.product.atoms[bond_key[1]]["type"]
            id1 = bond_key[1]
            print(
                f"{reactant_type0}\t{product_type0}\t{id0} \t - {reactant_type1}\t{product_type1}\t{id1}"
            )

        print(f"{len(formed_bonds)} forming bonds:")
        if len(formed_bonds) > 0:
            print("ProType, ReaType, ID\t - ProType, ReaType, ID")
        for bond_key in formed_bonds:
            reactant_type0 = self.reactant.atoms[bond_key[0]]["type"]
            product_type0 = self.product.atoms[bond_key[0]]["type"]
            id0 = bond_key[0]
            reactant_type1 = self.reactant.atoms[bond_key[1]]["type"]
            product_type1 = self.product.atoms[bond_key[1]]["type"]
            id1 = bond_key[1]
            print(
                f"{reactant_type0}\t{product_type0}\t{id0} \t - {reactant_type1}\t{product_type1}\t{id1}"
            )

    def build_forcefields(
        self,
        reactant: str,
        product: str | list[str],
        charge: int | list[int],  # type: ignore
        reparameterise: bool = True,
        optimise: bool = False,
    ):
        """
        Builds forcefields for the reactant and products.

        Args:
            reactant_filename (str): The filename of the reactant forcefield.
            products_filename (list[str]): A list of filenames for the product forcefields.
            charge (list[int], optional): A list of charges for each product. Defaults to list of zeros.
            optimise (bool, optional): Whether to optimise the geometry of the molecules. Defaults to False.
            hessians (list, optional): A list of hessians for each molecule. The first element corresponds to the reactant, and after that consecutively to the product.
                Should have as many elements as the amount of products + 1, but elements of the list can be None

        Returns:
            None
        """
        if isinstance(product, str):
            product = [product]
        if isinstance(charge, int):
            charge = [charge]

        # if not hessians is None:
        #     assert (
        #         len(hessians) == len(products_filename) + 1
        #     ), "Number of hessians must be 1 larger then the number of product files given"

        assert len(charge) == len(  # type: ignore
            product
        ), "Number of charges must be equal to the number of product files given"
        total_charge = sum(charge)  # type: ignore

        self.reactant = self.get_forcefield(
            reactant,
            total_charge,
            reparameterise=reparameterise,
            optimise=optimise,
        )
        self.reactant.ostream.flush()

        products = []
        for i, filename in enumerate(product):
            products.append(
                self.get_forcefield(
                    filename,
                    charge[i],  # type: ignore
                    reparameterise=reparameterise,
                    optimise=optimise,
                )
            )

        # Get all files and directories in the current directory
        items = os.listdir(".")

        for item in items:
            # If the item starts with 'vlx_'
            if item.startswith("vlx_"):
                # Construct full item path
                item_path = os.path.join(".", item)
                # If it's a file, remove it
                if os.path.isfile(item_path):
                    os.remove(item_path)
                # If it's a directory, remove it including all its content
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        # Never merge the reactant forcefield generators, for these we actually need positions
        self.product = merge_product_force_fields(
            self.reactant,
            products,
        )
        self.product.ostream.flush()
        self.summarise_reaction()
        self.save_forcefield(self.product, "_".join(product) + "_combined")

    @staticmethod
    def get_lambda(multiplier) -> np.ndarray:
        """
        Generate an array of lambda values based on the given multiplier. The ranges 0 - 0.1, 0.45 - 0.55 and 0.8 - 1 are more densely spaced

        Parameters:
        multiplier (float): The multiplier to adjust the step size for each lambda range.

        Returns:
        np.ndarray: An array of lambda values.

        """
        lambda1 = np.arange(0, 0.1, 0.02 * multiplier)
        lambda2 = np.arange(0.1, 0.45, 0.05 * multiplier)
        lambda3 = np.arange(0.45, 0.54, 0.02 * multiplier)
        lambda4 = np.arange(0.55, 0.8, 0.05 * multiplier)
        lambda5 = np.arange(0.8, 1.01, 0.02 * multiplier)

        Total_lambda = np.round(
            np.concatenate((lambda1, lambda2, lambda3, lambda4, lambda5)), 3
        )
        return Total_lambda

    # todo how can these options be moved to a constructor? -> should they?
    def build_systems(
        self,
        Lambda: list[float],
        temperature: float,
        constraints: dict | list[dict] = None,  # type: ignore
        pressure=1,
        solvated=False,
        solvent_smiles: str | list[str] = None,  # type: ignore
        solvent_count: int | list[int] = None,  # type: ignore
        solvent_density=None,
        solvent_nb_cutoff=None,
        solvent_nb_switch_factor: float = 0.9,
        NPT: bool = False,
        sc_alpha_lj: float = 0.85,
        sc_alpha_q: float = 0.3,
        sc_sigma_q: float = 1.0,
        sc_power: float = 1 / 6,
        morse_D_default: float = 500,  # kj/mol, default dissociation energy if none is given
        morse_couple: float = 1,  # kj/mol, scaling for the morse potential to emulate a coupling between two overlapping bonded states
        restraint_k: float = 1000,  # kj/mol nm^2, force constant for the position restraints
        restraint_r_default: float = 0.5,  # nm, default position restraint distance if none is given
        restraint_r_offset: float = 0.1,  # nm, distance added to the measured distance in a structure to set the position restraint distance
        coul14: float = 0.833,
        lj14: float = 0.5,
        data_folder: str = None,  # type: ignore
        run_folder: str = None,  # type: ignore
        # CNT_xyz_file: str = None,  # type: ignore
        # CNT_size: float = None,  # type: ignore # in nm
        # CNT_posres_force_k: float = 500000,
        # CNT_centering_force_k: float = 150,
        # CNT_centering_force_rmax: float = 1.5,  # in nm
        # CNT_reactant_packing_height: float = None,  # type: ignore # in nm
        # CNT_centering_force_Z_scaling: float = 0.1,
    ):
        """
        Set up the system builder with the specified parameters and parameterise the reaction space.

        Args:
            Lambda (float): The  Lambda array.
            temperature (float, optional): The temperature in Kelvin. Defaults to 300.
            pressure (float, optional): The pressure in bar. Defaults to 1.
            solvated (bool, optional): Whether to build solvated systems. Defaults to False.
            solvent_smiles (list[str], optional): A list of SMILES strings representing the solvents. Defaults to ["[2HO]"].
            solvent_count (list[int], optional): A list of the number of solvent molecules for each solvent. Defaults to [100].
            solvent_density (float, optional): The density of the solvent in kg/m^3. Defaults to 997.
            solvent_nb_cutoff (float, optional): The nonbonded cutoff distance for the solvent in nm. Defaults to 1.5.
            solvent_nb_switch_factor (float, optional): The nonbonded switch factor for the solvent. Defaults to 0.9.
            NPT (bool, optional): Whether to perform NPT ensemble simulations. Defaults to True.
            constraints (list, optional): A list of constraints to apply to the system. Defaults to None.
        """
        if solvated:
            if isinstance(solvent_smiles, str):
                solvent_smiles = [solvent_smiles]
            assert isinstance(
                solvent_smiles, list
            ), "solvent_smiles must be a list of SMILES strings"
            if isinstance(solvent_count, int):
                solvent_count = [solvent_count]
            assert isinstance(
                solvent_count, list
            ), "solvent_count must be a list of integers"
            assert len(solvent_smiles) == len(
                solvent_count
            ), "solvent_smiles and solvent_count must have the same length"
            assert isinstance(solvent_density, float) or isinstance(
                solvent_density, int
            ), "solvent_density must be a number"

        if constraints is None:
            constraints = []
        if isinstance(constraints, dict):
            constraints = [constraints]
        self.system_builder = System_builder(
            self.reactant, self.product, Lambda, temperature, pressure, NPT
        )
        assert (
            Lambda[0] == 0 and Lambda[-1] == 1
        ), f"Lambda must start at 0 and end at 1. Lambda = {Lambda}"
        assert np.all(
            np.diff(Lambda) > 0
        ), f"Lambda must be monotonically increasing. Lambda = {Lambda}"

        solvent_str = str(solvent_smiles).replace("'", "").replace(", ", "_")[1:-1]
        if data_folder is None:
            if solvated is False:
                data_folder = "data_vacuum"
            else:
                # if CNT_xyz_file is None:
                data_folder = f"data_{solvent_str}"
                # else:
                #     data_folder = f"data_{solvent_str}_{CNT_xyz_file[:-4]}"
        if run_folder is None:
            if solvated is False:
                run_folder = "run_vacuum"
            else:
                # if CNT_xyz_file is None:
                run_folder = f"run_{solvent_str}"
                # else:
                #     run_folder = f"run_{solvent_str}_{CNT_xyz_file[:-4]}"

        self.data_folder = data_folder
        self.run_folder = run_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)
        else:
            import shutil

            folder = f"./{self.run_folder}"
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))

        if solvated:

            self.system_builder.build_solvated_systems(
                smiles=solvent_smiles,
                molecules_count=solvent_count,
                density=solvent_density,
                nonbonded_cutoff=solvent_nb_cutoff,  # type: ignore
                nonbonded_switch_factor=solvent_nb_switch_factor,
                periodic_box=True,
                neutralise_charge=True,
                # CNT_xyz=CNT_xyz_file,
                # CNT_size=CNT_size,
                # CNT_posres_force_k=CNT_posres_force_k,
                # CNT_centering_force_k=CNT_centering_force_k,
                # CNT_centering_force_rmax=CNT_centering_force_rmax,
                # CNT_reactant_packing_height=CNT_reactant_packing_height,
                # CNT_centering_force_Z_scaling=CNT_centering_force_Z_scaling,
            )
        else:
            self.system_builder.build_systems()

        self.system_builder.parameterise_reaction_space(
            True,
            self.run_folder,
            constraints,
            soft_core_run=False,
            soft_core_ref=True,
            sc_alpha_lj=sc_alpha_lj,
            sc_alpha_q=sc_alpha_q,
            sc_sigma_q=sc_sigma_q,
            sc_power=sc_power,
            morse_D_default=morse_D_default,
            morse_couple=morse_couple,
            restraint_k=restraint_k,
            restraint_r_default=restraint_r_default,
            restraint_r_offset=restraint_r_offset,
            coul14=coul14,
            lj14=lj14,
            verbose=False,
        )

        np.savetxt(f"{self.data_folder}/Lambda.dat", Lambda)
        mmapp.PDBFile.writeFile(
            self.system_builder.topology,
            self.system_builder.positions
            * 10,  # positions are handled in nanometers, but pdb's should be in angstroms
            open(f"{data_folder}/topology.pdb", "w"),
        )
        option_dict = {
            "temperature": temperature,
            "constraints": str(constraints),
            "pressure": pressure,
            "solvated": solvated,
            "solvent_smiles": str(solvent_smiles),
            "solvent_count": str(solvent_count),
            "solvent_density": solvent_density,
            "solvent_nb_cutoff": solvent_nb_cutoff,
            "solvent_nb_switch_factor": solvent_nb_switch_factor,
            "NPT": NPT,
            "sc_alpha_lj": sc_alpha_lj,
            "sc_alpha_q": sc_alpha_q,
            "sc_sigma_q": sc_sigma_q,
            "sc_power": sc_power,
            "morse_D_default": morse_D_default,
            "morse_couple": morse_couple,
            "restraint_k": restraint_k,
            "restraint_r_default": restraint_r_default,
            "restraint_r_offset": restraint_r_offset,
            "coul14": coul14,
            "lj14": lj14,
            "data_folder": data_folder,
            "run_folder": run_folder,
            # "CNT_xyz_file": CNT_xyz_file,
            # "CNT_size": CNT_size,
            # "CNT_posres_force_k": CNT_posres_force_k,
            # "CNT_centering_force_k": CNT_centering_force_k,
            # "CNT_centering_force_rmax": CNT_centering_force_rmax,
            # "CNT_reactant_packing_height": CNT_reactant_packing_height,
            # "CNT_centering_force_Z_scaling": CNT_centering_force_Z_scaling,
        }
        json.dump(option_dict, open(f"{data_folder}/options.json", "w"), indent=4)

    def load_systems(self, data_folder: str, run_folder: str):
        self.data_folder = data_folder
        self.run_folder = run_folder

        assert os.path.exists(
            self.data_folder
        ), f"Data folder {self.data_folder} not found"
        assert os.path.exists(
            self.run_folder
        ), f"Run folder {self.run_folder} not found"

        options = json.load(open(f"{self.data_folder}/options.json", "r"))
        pdb = mmapp.PDBFile(f"{self.data_folder}/topology.pdb")
        topology = pdb.getTopology()
        positions = pdb.getPositions()
        Lambda = list(np.loadtxt(f"{self.data_folder}/Lambda.dat"))

        self.system_builder = System_builder(
            reactant=vlx.ForceFieldGenerator(),
            product=vlx.ForceFieldGenerator(),
            Lambda=Lambda,
            temperature=options["temperature"],
            pressure=options["pressure"],
            NPT=options["NPT"],
        )
        self.system_builder.topology = topology
        self.system_builder.positions = positions
        systems = {}
        with open(f"{self.run_folder}/reactant.xml", "r") as xml:
            systems.update({"reactant": mm.XmlSerializer.deserialize(xml.read())})
        with open(f"{self.run_folder}/product.xml", "r") as xml:
            systems.update({"product": mm.XmlSerializer.deserialize(xml.read())})
        for l in Lambda:
            with open(f"{self.run_folder}/{l:.3f}_sys.xml", "r") as xml:
                systems.update({l: mm.XmlSerializer.deserialize(xml.read())})
        self.system_builder.systems = systems

    def write_step_data(
        self,
        equil_steps,
        total_sample_steps,
        write_step,
        initial_equil_steps,
        step_size,
    ):
        data = {
            "equil_steps": equil_steps,
            "total_sample_steps": total_sample_steps,
            "write_step": write_step,
            "initial_equil_steps": initial_equil_steps,
            "step_size": step_size,
        }
        with open(f"{self.data_folder}/step_parameters.txt", "w") as file:
            for key, value in data.items():
                file.write(f"{key}: {value}\n")

    def run_FEP(
        self,
        equil_steps,
        total_sample_steps,
        write_step,
        initial_equil_steps=0,
        step_size=0.001,
        platform="CPU",
    ):
        """
        Sets up and runs the Free Energy Perturbation (FEP) simulation.

        Args:
            equil_steps (int): Number of equilibration steps.
            total_sample_steps (int): Total number of sampling steps.
            write_step (int): Frequency of writing output files.
            initial_equil_steps (int, optional): Number of initial equilibration steps. Defaults to 0.
            step_size (float, optional): Step size for the simulation. Defaults to 0.001.
            platform (str, optional): Platform for running the simulation. Defaults to "CPU".
        """

        self.write_step_data(
            equil_steps,
            total_sample_steps,
            write_step,
            initial_equil_steps=0,
            step_size=0.001,
        )
        self.FEP = FEP_driver(
            self.run_folder,
            self.data_folder,
            self.system_builder,
            mm.Platform.getPlatformByName(platform),
        )

        self.FEP.run_FEP(
            equil_steps,
            total_sample_steps,
            write_step,
            initial_equil_steps,
            step_size,
        )

        self.FEP.recalculate(interpolated_potential=True, force_contributions=True)
