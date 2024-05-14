import copy
import veloxchem as vlx


# Applies a mapping to agiven dictionary of parameters(bonds, angles, etc.)


def apply_mapping_to_parameters(
    old_parameters: dict[tuple, dict], mapping: dict[int, int]
) -> dict[tuple, dict]:
    new_parameters = {}
    for old_key in old_parameters:
        new_key = tuple([mapping[atom_key] for atom_key in old_key])
        val = old_parameters[old_key]
        new_parameters.update({new_key: val})
    return new_parameters


def apply_mapping_to_forcefield(
    forcefield: vlx.ForceFieldGenerator, mapping: dict[int, int]
) -> vlx.ForceFieldGenerator:
    new_product_atoms = {}
    for atom_key in forcefield.atoms:
        key = mapping[atom_key]
        val = forcefield.atoms[atom_key]
        new_product_atoms.update({key: val})
    forcefield.atoms = new_product_atoms
    forcefield.bonds = apply_mapping_to_parameters(forcefield.bonds, mapping)
    forcefield.angles = apply_mapping_to_parameters(forcefield.angles, mapping)
    forcefield.dihedrals = apply_mapping_to_parameters(forcefield.dihedrals, mapping)
    forcefield.impropers = apply_mapping_to_parameters(forcefield.impropers, mapping)
    return forcefield


# Applies a shift to a given dictionary of parameters(bonds, angles, etc.)


def apply_shift_to_parameters(
    shift: int, old_parameters: dict[tuple, dict]
) -> dict[tuple, dict]:
    new_parameters = {}
    for old_key in old_parameters:
        new_key = tuple([atom_key + shift for atom_key in old_key])
        val = old_parameters[old_key]
        new_parameters.update({new_key: val})
    return new_parameters

    # Shifts all indices in a forcefield_generator by a given amount


def shift_atom_indices(
    forcefield: vlx.ForceFieldGenerator, shift: int
) -> vlx.ForceFieldGenerator:
    new_atoms = {}
    for atom in forcefield.atoms:
        new_key = atom + shift
        new_atoms.update({new_key: forcefield.atoms[atom]})
    forcefield.atoms = new_atoms
    forcefield.bonds = apply_shift_to_parameters(shift, forcefield.bonds)
    forcefield.angles = apply_shift_to_parameters(shift, forcefield.angles)
    forcefield.dihedrals = apply_shift_to_parameters(shift, forcefield.dihedrals)
    forcefield.impropers = apply_shift_to_parameters(shift, forcefield.impropers)
    return forcefield


def get_atoms_mapping(
    reactant: vlx.ForceFieldGenerator, products: list[vlx.ForceFieldGenerator]
) -> dict:
    reactant_atoms: dict[int, dict] = reactant.atoms
    # make a copy of the product atoms so they can be modified and removed
    products_atoms: list[dict[int, dict]] = [
        copy.deepcopy(product.atoms) for product in products
    ]
    product_i = 0
    product_amount = len(products)
    mapping = {}

    # loop over all reactant atoms
    for reactant_atom in reactant_atoms:
        searching = True
        reactant_mass = reactant_atoms[reactant_atom]["mass"]
        while searching:
            # find per reactant atom a corresponding product atom in the first product
            match_found = False
            match_atom = None
            for product_atom in products_atoms[product_i]:
                product_mass = products_atoms[product_i][product_atom]["mass"]
                # for a match, add it to the mapping and remove both atoms
                if reactant_mass == product_mass:
                    match_found = True
                    mapping.update({product_atom: reactant_atom})
                    # todo remove product_atom in a nicer way
                    match_atom = product_atom
                    break
            # if no product is found, continue searching in the next product
            if match_found == False:
                product_i += 1
                product_i = product_i % product_amount
            # if a match is found, continue searching in the same product afterwards
            else:
                searching = False
        # go on till there is nothing left
        del products_atoms[product_i][match_atom]
    return mapping


def create_combined_forcefield(
    forcefields: list[vlx.ForceFieldGenerator],
) -> vlx.ForceFieldGenerator:
    forcefield = vlx.ForceFieldGenerator()
    forcefield.atoms = {}
    forcefield.bonds = {}
    forcefield.angles = {}
    forcefield.dihedrals = {}
    forcefield.impropers = {}
    atom_count = 0
    for product_ffgen in forcefields:
        # The shifting guarantees unique id's for all atoms
        # todo make this more robust, check if shifting is necessary
        shift_atom_indices(product_ffgen, atom_count)
        atom_count += len(product_ffgen.atoms)
        forcefield.atoms.update(product_ffgen.atoms)
        forcefield.bonds.update(product_ffgen.bonds)
        forcefield.angles.update(product_ffgen.angles)
        forcefield.dihedrals.update(product_ffgen.dihedrals)
        forcefield.impropers.update(product_ffgen.impropers)
    return forcefield


def merge_product_force_fields(
    reactant: vlx.ForceFieldGenerator,
    products: list[vlx.ForceFieldGenerator],
    atoms_mapping: dict = None,  # type: ignore
) -> vlx.ForceFieldGenerator:

    # merge all of the products into one large forcefield generator, creating new unique ids if necessary
    product = create_combined_forcefield(products)
    assert len(reactant.atoms) == len(
        product.atoms
    ), "The number of atoms in the reactant and combined product do not match"

    # Generate the mapping if not provided
    if atoms_mapping is None:
        atoms_mapping = {}
        # Check if we need a mapping at all
        matching = True
        for reactant_atom, product_atom in zip(
            reactant.atoms.values(), product.atoms.values()
        ):
            if reactant_atom["mass"] != product_atom["mass"]:
                print(
                    "Reactant and product atom id's have nonmatching elements, attempting to construct mapping. Check the output for correctness."
                )
                atoms_mapping = get_atoms_mapping(reactant, products)
                matching = False
                break

        # If we need no mapping, generate a trivial one
        if matching:
            for reactant_key, product_key in zip(
                reactant.atoms.keys(), product.atoms.keys()
            ):
                atoms_mapping.update({reactant_key: product_key})

    # Apply the mapping
    # product = apply_mapping_to_forcefield(product, atoms_mapping)

    return product
