import math
import typing
import veloxchem as vlx
import openmm as mm


class Reaction_space_builder:

    def __init__(
        self,
        reactant: vlx.ForceFieldGenerator,
        product: vlx.ForceFieldGenerator,
        reaction_atoms,
        constraints,
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
        self.reactant: vlx.ForceFieldGenerator = reactant
        self.product: vlx.ForceFieldGenerator = product
        self.reaction_atoms = reaction_atoms
        self.constraints = constraints

        # soft-core parameters
        self.alpha_lj = sc_alpha_lj
        self.alpha_q = sc_alpha_q
        self.sigma_q = sc_sigma_q
        self.sc_power = sc_power

        self.morse_default_D = morse_D_default
        self.morse_couple = morse_couple
        self.restraint_k = restraint_k
        self.restraint_r_default = restraint_r_default

        self.restraint_r_offset = restraint_r_offset  # nm

        # Scaling constants for 1-4 interactions
        self.coul14 = coul14
        self.lj14 = lj14

        self.verbose = verbose

        # 1/(4 pi epsilon) = 138.93..., https://manual.gromacs.org/current/reference-manual/definitions.html
        self.k = 4.184 * vlx.hartree_in_kcalpermol() * 0.1 * vlx.bohr_in_angstrom()

        self.deg_to_rad: float = vlx.mathconst_pi() / 180

    # if reference state is true, don't add any constraints and restraints, and don't add any bonds to the topology that are scaled to 0
    def add_reaction_forces(
        self, system, l, reference_state=False, soft_core=False
    ) -> mm.System:
        self.reference_state = reference_state

        # todos bond forces

        # add morse_couple

        forces: list[mm.Force] = []
        forces = forces + self.create_bond_forces(l)
        forces = forces + self.create_angle_forces(l)
        forces = forces + self.create_proper_torsion_forces(l)
        forces = forces + self.create_improper_torsion_forces(l)
        forces = forces + self.create_nonbonded_forces(l, soft_core)
        forces = forces + self.create_constraint_forces(l)

        for i, force in enumerate(forces):
            force.setForceGroup(i + 1)  # The 0th force group are the solvent forces
            system.addForce(force)
        return system

    def reaction_to_total_atomid(self, reaction_atom_id: int) -> int:
        # print(
        #     f"Converted reaction id {reaction_atom_id} to total id {self.reaction_atoms[reaction_atom_id].index}"
        # )
        return self.reaction_atoms[reaction_atom_id].index

    def create_bond_forces(self, l) -> list[mm.Force]:

        harmonic_force = mm.HarmonicBondForce()
        harmonic_force.setName("Reaction harmonic bond")

        morse_expr = "D*(1-exp(-a*(r-re)))^2;"
        morse_force = mm.CustomBondForce(morse_expr)
        morse_force.setName("Reaction morse bond")
        morse_force.addPerBondParameter("D")
        morse_force.addPerBondParameter("a")
        morse_force.addPerBondParameter("re")

        max_dist_expr = "k*step(r-rmax)*(r-rmax)^2"
        max_distance = mm.CustomBondForce(max_dist_expr)
        max_distance.setName("Reaction distance restraint")
        max_distance.addPerBondParameter("rmax")
        max_distance.addPerBondParameter("k")

        bond_keys = list(set(self.reactant.bonds) | set(self.product.bonds))
        static_bond_keys = list(set(self.reactant.bonds) & set(self.product.bonds))
        broken_bond_keys = list(set(self.reactant.bonds) - set(self.product.bonds))
        formed_bond_keys = list(set(self.product.bonds) - set(self.reactant.bonds))

        for key in bond_keys:
            # mm_top.addBond(mm_atoms[key[0]], mm_atoms[key[1]])
            # if in a and b: interpolate harmonic

            if key in static_bond_keys:

                bondA = self.reactant.bonds[key]
                bondB = self.product.bonds[key]

                if bondA["equilibrium"] == bondB["equilibrium"]:
                    harmonic_force.addBond(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        bondA["equilibrium"],
                        (1 - l) * bondA["force_constant"] + l * bondB["force_constant"],
                    )
                else:
                    harmonic_force.addBond(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        bondA["equilibrium"],
                        (1 - l) * bondA["force_constant"],
                    )
                    harmonic_force.addBond(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        bondB["equilibrium"],
                        l * bondB["force_constant"],
                    )
            # if in a or b:
            else:
                if key in broken_bond_keys:
                    scale = 1 - l
                    bond = self.reactant.bonds[key]
                    if hasattr(self.product, "molecule"):
                        coords = self.product.molecule.get_coordinates_in_angstrom()
                        r = (
                            vlx.AtomTypeIdentifier.measure_length(
                                coords[key[0]], coords[key[1]]
                            )
                            * 0.1
                        )
                    else:
                        r = self.restraint_r_default
                        if self.verbose:
                            print(
                                f"INFO: No product geometry given, defaulting position restraint to {r} nm"
                            )
                elif key in formed_bond_keys:
                    scale = l
                    bond = self.product.bonds[key]
                    coords = self.reactant.molecule.get_coordinates_in_angstrom()
                    r = (
                        vlx.AtomTypeIdentifier.measure_length(
                            coords[key[0]], coords[key[1]]
                        )
                        * 0.1
                    )
                else:

                    assert (
                        False
                    ), "A bond can either be static, or dynamic, in  which case it can be broken or formed"
                if scale > 0:
                    # scale morse
                    if not "D" in bond.keys():
                        D = self.morse_default_D
                        if self.verbose:
                            print(
                                f"INFO: no D value associated with bond {key[0]} {key[1]}. Setting to default value {self.morse_default_D}"
                            )
                    else:
                        D = bond["D"]
                    a = math.sqrt(bond["force_constant"] / (2 * D))
                    re = bond["equilibrium"]
                    morse_force.addBond(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        [scale * D, a, re],
                    )
                    # harm_bond_force.addBond(key[0],key[1],bond['equilibrium'],0.5*scale*bond['force_constant'])

                if not self.reference_state:
                    k = self.restraint_k
                    rmax = r + self.restraint_r_offset
                    max_distance.addBond(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        [rmax, (1 - scale) * k],
                    )
                    if self.verbose:
                        print(
                            f"INFO: Adding maximum distance {rmax} with k {(1-scale)*k} to atoms {key[0]} and {key[1]} of for lambda {l}"
                        )
        return [harmonic_force, morse_force, max_distance]

    def create_angle_forces(self, l: float) -> typing.List[mm.Force]:
        harmonic_force = mm.HarmonicAngleForce()
        harmonic_force.setName("Reaction harmonic angle")
        angle_keys = list(set(self.reactant.angles) | set(self.product.angles))
        for key in angle_keys:
            if key in self.reactant.angles.keys() and key in self.product.angles.keys():
                angleA = self.reactant.angles[key]
                angleB = self.product.angles[key]
                if angleA["equilibrium"] == angleB["equilibrium"]:
                    harmonic_force.addAngle(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        self.reaction_to_total_atomid(key[2]),
                        angleA["equilibrium"] * self.deg_to_rad,
                        (1 - l) * angleA["force_constant"]
                        + l * angleB["force_constant"],
                    )
                else:
                    harmonic_force.addAngle(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        self.reaction_to_total_atomid(key[2]),
                        angleA["equilibrium"] * self.deg_to_rad,
                        (1 - l) * angleA["force_constant"],
                    )
                    harmonic_force.addAngle(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        self.reaction_to_total_atomid(key[2]),
                        angleB["equilibrium"] * self.deg_to_rad,
                        l * angleB["force_constant"],
                    )
            else:
                if key in self.reactant.angles.keys():
                    scale = 1 - l
                    angle = self.reactant.angles[key]
                else:
                    scale = l
                    angle = self.product.angles[key]
                if scale > 0:
                    harmonic_force.addAngle(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        self.reaction_to_total_atomid(key[2]),
                        angle["equilibrium"] * self.deg_to_rad,
                        scale * angle["force_constant"],
                    )
        return [harmonic_force]

    def create_proper_torsion_forces(self, l) -> typing.List[mm.Force]:
        fourier_force = mm.PeriodicTorsionForce()
        fourier_force.setName("Reaction proper fourier torsion")
        RB_force = mm.RBTorsionForce()
        RB_force.setName("Reaction proper RB torsion")

        dihedral_keys = list(set(self.reactant.dihedrals) | set(self.product.dihedrals))
        for key in dihedral_keys:
            total_atom_id = [
                self.reaction_to_total_atomid(reaction_atomid)
                for reaction_atomid in key
            ]
            if (
                key in self.reactant.dihedrals.keys()
                and key in self.product.dihedrals.keys()
            ):
                dihedA = self.reactant.dihedrals[key]
                if dihedA["type"] == "RB":
                    RB_force.addTorsion(
                        total_atom_id[0],
                        total_atom_id[1],
                        total_atom_id[2],
                        total_atom_id[3],
                        (1 - l) * dihedA["RB_coefficients"][0],
                        (1 - l) * dihedA["RB_coefficients"][1],
                        (1 - l) * dihedA["RB_coefficients"][2],
                        (1 - l) * dihedA["RB_coefficients"][3],
                        (1 - l) * dihedA["RB_coefficients"][4],
                        (1 - l) * dihedA["RB_coefficients"][5],
                    )
                elif dihedA["type"] == "Fourier":
                    fourier_force.addTorsion(
                        total_atom_id[0],
                        total_atom_id[1],
                        total_atom_id[2],
                        total_atom_id[3],
                        dihedA["periodicity"],
                        dihedA["phase"],
                        (1 - l) * dihedA["barrier"],
                    )
                else:
                    assert False, "Unknown dihedral type"
                dihedB = self.product.dihedrals[key]
                if dihedB["type"] == "RB":
                    RB_force.addTorsion(
                        total_atom_id[0],
                        total_atom_id[1],
                        total_atom_id[2],
                        total_atom_id[3],
                        l * dihedB["RB_coefficients"][0],
                        l * dihedB["RB_coefficients"][1],
                        l * dihedB["RB_coefficients"][2],
                        l * dihedB["RB_coefficients"][3],
                        l * dihedB["RB_coefficients"][4],
                        l * dihedB["RB_coefficients"][5],
                    )
                elif dihedB["type"] == "Fourier":
                    fourier_force.addTorsion(
                        total_atom_id[0],
                        total_atom_id[1],
                        total_atom_id[2],
                        total_atom_id[3],
                        dihedB["periodicity"],
                        dihedB["phase"],
                        l * dihedB["barrier"],
                    )
                else:
                    assert False, "Unknown dihedral type"
            else:
                if key in self.reactant.dihedrals.keys():
                    scale = 1 - l
                    dihed = self.reactant.dihedrals[key]
                else:
                    scale = l
                    dihed = self.product.dihedrals[key]
                if scale > 0:
                    if dihed["type"] == "RB":
                        RB_force.addTorsion(
                            total_atom_id[0],
                            total_atom_id[1],
                            total_atom_id[2],
                            total_atom_id[3],
                            scale * dihed["RB_coefficients"][0],
                            scale * dihed["RB_coefficients"][1],
                            scale * dihed["RB_coefficients"][2],
                            scale * dihed["RB_coefficients"][3],
                            scale * dihed["RB_coefficients"][4],
                            scale * dihed["RB_coefficients"][5],
                        )
                    elif dihed["type"] == "Fourier":
                        fourier_force.addTorsion(
                            total_atom_id[0],
                            total_atom_id[1],
                            total_atom_id[2],
                            total_atom_id[3],
                            dihed["periodicity"],
                            dihed["phase"],
                            scale * dihed["barrier"],
                        )
        return [fourier_force, RB_force]

    def create_improper_torsion_forces(self, l) -> typing.List[mm.Force]:

        # harm_force = mm.CustomTorsionForce("0.5*k*(theta-theta0)^2")
        # harm_force.setName("Reaction harmonic improper torsion")
        # harm_force.addPerTorsionParameter("theta0")
        # harm_force.addPerTorsionParameter("k")

        fourier_force = mm.PeriodicTorsionForce()
        fourier_force.setName("Reaction improper fourier torsion")

        dihedral_keys = list(set(self.reactant.impropers) | set(self.product.impropers))
        for key in dihedral_keys:
            total_atom_id = [
                self.reaction_to_total_atomid(reaction_atomid)
                for reaction_atomid in key
            ]
            if (
                key in self.reactant.impropers.keys()
                and key in self.product.impropers.keys()
            ):
                dihedA = self.reactant.impropers[key]
                if dihedA["type"] == "Fourier":
                    fourier_force.addTorsion(
                        total_atom_id[0],
                        total_atom_id[1],
                        total_atom_id[2],
                        total_atom_id[3],
                        dihedA["periodicity"],
                        dihedA["phase"],
                        (1 - l) * dihedA["barrier"],
                    )
                else:
                    assert False, "Unknown dihedral type"
                dihedB = self.product.impropers[key]
                if dihedB["type"] == "Fourier":
                    fourier_force.addTorsion(
                        total_atom_id[0],
                        total_atom_id[1],
                        total_atom_id[2],
                        total_atom_id[3],
                        dihedB["periodicity"],
                        dihedB["phase"],
                        l * dihedB["barrier"],
                    )
                else:
                    assert False, "Unknown dihedral type"
            else:
                if key in self.reactant.impropers.keys():
                    scale = 1 - l
                    dihed = self.reactant.impropers[key]
                else:
                    scale = l
                    dihed = self.product.impropers[key]
                if scale > 0:
                    if dihed["type"] == "Fourier":
                        fourier_force.addTorsion(
                            total_atom_id[0],
                            total_atom_id[1],
                            total_atom_id[2],
                            total_atom_id[3],
                            dihed["periodicity"],
                            dihed["phase"],
                            scale * dihed["barrier"],
                        )
        return [fourier_force]

    def get_long_range_expression(self, l, soft_core: bool):
        soft_core_expression = (
            " (1-l) * (LjtotA + CoultotA) "
            "  + l  * (LjtotB + CoultotB); "
            ""
            "LjtotA     = (step(r - rljA) * LjA + step(rljA - r) * LjlinA);"
            "LjtotB     = (step(r - rljB) * LjB + step(rljB - r) * LjlinB);"
            "LjlinA     = ( (78*A12) / (rljAdiv^14) - (21*A6) / (rljAdiv^8) )*r^2 - ( (168*A12) / (rljAdiv^13) - (48*A6) / (rljAdiv^7) )*r + ( (91*A12) / (rljAdiv^12) - (28*A6) / (rljAdiv^6) );"
            "LjlinB     = ( (78*B12) / (rljBdiv^14) - (21*B6) / (rljBdiv^8) )*r^2 - ( (168*B12) / (rljBdiv^13) - (48*B6) / (rljBdiv^7) )*r + ( (91*B12) / (rljBdiv^12) - (28*B6) / (rljBdiv^6) );"
            # if rljA = 0, returns 1, otherwise returns rljA. Prevents division by 0 while the step factor is already 0
            "rljAdiv    = select(rljA,rljA,1);"
            "rljBdiv    = select(rljB,rljB,1);"
            "rljA       = alphalj * ( (26/7 ) * A6  * 1 ) ^ pow;"
            "rljB       = alphalj * ( (26/7 ) * B6  * 1 ) ^ pow;"
            "LjA        = A12 / r^12 - A6 / r^6;"
            "LjB        = B12 / r^12 - B6 / r^6;"
            "A12        = 4 * epsilonA * sigmaA ^ 12; "
            "B12        = 4 * epsilonB * sigmaB ^ 12; "
            "A6         = 4 * epsilonA * sigmaA ^ 6; "
            "B6         = 4 * epsilonB * sigmaB ^ 6; "
            ""
            "CoultotA   = step(r-rqA) * CoulA + step(rqA-r) * CoullinA;"
            "CoultotB   = step(r-rqB) * CoulB + step(rqB-r) * CoullinB;"
            "CoullinA   = k * ( ( qqA / rqAdiv^3 ) * r^2 - 3 * ( qqA / rqAdiv^2 ) * r + 3 * ( qqA / rqAdiv ) );"
            "CoullinB   = k * ( ( qqB / rqBdiv^3 ) * r^2 - 3 * ( qqB / rqBdiv^2 ) * r + 3 * ( qqB / rqBdiv ) );"
            "rqAdiv     = select(rqA,rqA,1);"
            "rqBdiv     = select(rqB,rqB,1);"
            "rqA        = alphaq * ( 1 + sigmaq * qqA )  * 1 ^ pow;"
            "rqB        = alphaq * ( 1 + sigmaq * qqB )  * 1 ^ pow;"
            "CoulA      = k*qqA/r; "
            "CoulB      = k*qqB/r; "
            ""
            f"k         = {self.k};"
            f"l         = {l};"
            ""
            f"alphalj   = {self.alpha_lj};"
            f"alphaq    = {self.alpha_q};"
            f"sigmaq    = {self.sigma_q};"
            f"pow       = {self.sc_power};"
        )

        hard_core_expression = (
            " (1-l) * (LjtotA + CoultotA) "
            "  + l  * (LjtotB + CoultotB); "
            ""
            "LjtotA     = A12 / r^12 - A6 / r^6;"
            "LjtotB     = B12 / r^12 - B6 / r^6;"
            "A12        = 4 * epsilonA * sigmaA ^ 12; "
            "B12        = 4 * epsilonB * sigmaB ^ 12; "
            "A6         = 4 * epsilonA * sigmaA ^ 6; "
            "B6         = 4 * epsilonB * sigmaB ^ 6; "
            ""
            "CoultotA   = k*qqA/r; "
            "CoultotB   = k*qqB/r; "
            ""
            f"k         = {self.k};"
            f"l         = {l};"
        )

        if soft_core:
            return soft_core_expression
        else:
            return hard_core_expression

    def create_nonbonded_forces(self, l, soft_core=False) -> typing.List[mm.Force]:
        nonbonded_force = mm.CustomBondForce(
            self.get_long_range_expression(l, soft_core)
        )
        if soft_core:
            nonbonded_force.setName("Reaction internal nonbonded soft-core")
        else:
            nonbonded_force.setName("Reaction internal nonbonded hard-core")
        nonbonded_force.addPerBondParameter("sigmaA")
        nonbonded_force.addPerBondParameter("sigmaB")
        nonbonded_force.addPerBondParameter("epsilonA")
        nonbonded_force.addPerBondParameter("epsilonB")
        nonbonded_force.addPerBondParameter("qqA")
        nonbonded_force.addPerBondParameter("qqB")

        reactant_exceptions = self.create_exceptions_from_bonds(
            self.reactant, self.coul14, self.lj14
        )
        product_exceptions = self.create_exceptions_from_bonds(
            self.product, self.coul14, self.lj14
        )

        for i in self.reactant.atoms.keys():
            for j in self.reactant.atoms.keys():
                if i < j:
                    key = (i, j)
                    # Remove any exception from the nonbondedforce
                    # and add it instead to the exception bond force
                    if (
                        key in reactant_exceptions.keys()
                        and key in product_exceptions.keys()
                    ):
                        epsilonA = reactant_exceptions[key]["epsilon"]
                        epsilonB = product_exceptions[key]["epsilon"]
                        sigmaA = reactant_exceptions[key]["sigma"]
                        sigmaB = product_exceptions[key]["sigma"]
                        qqA = reactant_exceptions[key]["qq"]
                        qqB = product_exceptions[key]["qq"]
                    elif key in reactant_exceptions.keys():
                        epsilonA = reactant_exceptions[key]["epsilon"]
                        sigmaA = reactant_exceptions[key]["sigma"]
                        qqA = reactant_exceptions[key]["qq"]

                        atomB1 = self.product.atoms[key[0]]
                        atomB2 = self.product.atoms[key[1]]

                        epsilonB = math.sqrt(atomB1["epsilon"] * atomB2["epsilon"])
                        sigmaB = 0.5 * (atomB1["sigma"] + atomB2["sigma"])
                        qqB = atomB1["charge"] * atomB2["charge"]
                    elif key in product_exceptions.keys():
                        atomA1 = self.reactant.atoms[key[0]]
                        atomA2 = self.reactant.atoms[key[1]]

                        epsilonA = math.sqrt(atomA1["epsilon"] * atomA2["epsilon"])
                        sigmaA = 0.5 * (atomA1["sigma"] + atomA2["sigma"])
                        qqA = atomA1["charge"] * atomA2["charge"]

                        epsilonB = product_exceptions[key]["epsilon"]
                        sigmaB = product_exceptions[key]["sigma"]
                        qqB = product_exceptions[key]["qq"]
                    else:
                        atomA1 = self.reactant.atoms[key[0]]
                        atomA2 = self.reactant.atoms[key[1]]

                        epsilonA = math.sqrt(atomA1["epsilon"] * atomA2["epsilon"])
                        sigmaA = 0.5 * (atomA1["sigma"] + atomA2["sigma"])
                        qqA = atomA1["charge"] * atomA2["charge"]

                        atomB1 = self.product.atoms[key[0]]
                        atomB2 = self.product.atoms[key[1]]

                        epsilonB = math.sqrt(atomB1["epsilon"] * atomB2["epsilon"])
                        sigmaB = 0.5 * (atomB1["sigma"] + atomB2["sigma"])
                        qqB = atomB1["charge"] * atomB2["charge"]

                    if sigmaA == 1.0:
                        sigmaA = sigmaB
                    elif sigmaB == 1.0:
                        sigmaB = sigmaA
                    if not (
                        qqA == 0.0
                        and qqB == 0.0
                        and epsilonA == 0.0
                        and epsilonB == 0.0
                    ):
                        nonbonded_force.addBond(
                            self.reaction_to_total_atomid(key[0]),
                            self.reaction_to_total_atomid(key[1]),
                            [sigmaA, sigmaB, epsilonA, epsilonB, qqA, qqB],
                        )

        return [nonbonded_force]

    def create_exceptions_from_bonds(
        self, molecule: vlx.ForceFieldGenerator, coulomb14_scale, lj14_scale
    ):
        particles = molecule.atoms

        exclusions = [set() for _ in range(len(particles))]
        bonded12 = [set() for _ in range(len(particles))]
        exceptions = {}

        # Populate bonded12 with bonds.
        for bond in molecule.bonds.keys():
            bonded12[bond[0]].add(bond[1])
            bonded12[bond[1]].add(bond[0])

        # Find particles separated by 1, 2, or 3 bonds.
        for i in range(len(particles)):
            self.add_exclusions_to_set(bonded12, exclusions[i], i, i, 2)

        # Find particles separated by 1 or 2 bonds and create exceptions.
        for i in range(len(exclusions)):
            bonded13 = set()
            self.add_exclusions_to_set(bonded12, bonded13, i, i, 1)
            for j in exclusions[i]:
                if j < i:
                    if j not in bonded13:
                        # This is a 1-4 interaction.
                        particle1 = particles[j]
                        particle2 = particles[i]
                        charge_prod = (
                            coulomb14_scale * particle1["charge"] * particle2["charge"]
                        )
                        sigma = 0.5 * (particle1["sigma"] + particle2["sigma"])
                        epsilon = (
                            lj14_scale
                            * (particle1["epsilon"] * particle2["epsilon"]) ** 0.5
                        )
                        exceptions[tuple(sorted((i, j)))] = {
                            "qq": charge_prod,
                            "sigma": sigma,
                            "epsilon": epsilon,
                        }
                    else:
                        # This interaction should be completely excluded.
                        exceptions[tuple(sorted((i, j)))] = {
                            "qq": 0.0,
                            "sigma": 1.0,
                            "epsilon": 0.0,
                        }
        return exceptions

    def add_exclusions_to_set(
        self, bonded12, exclusions, base_particle, from_particle, current_level
    ):
        for i in bonded12[from_particle]:
            if i != base_particle:
                exclusions.add(i)
            if current_level > 0:
                self.add_exclusions_to_set(
                    bonded12, exclusions, base_particle, i, current_level - 1
                )

    def create_constraint_forces(self, l) -> typing.List[mm.Force]:
        bond_constraint = mm.HarmonicBondForce()
        bond_constraint.setName("Bond constraint")
        angle_constraint = mm.HarmonicAngleForce()
        angle_constraint.setName("Angle constraint")
        torsion_constraint = mm.CustomTorsionForce("0.5*k*(theta-theta0)^2")
        torsion_constraint.setName("Harmonic torsion constraint")
        torsion_constraint.addPerTorsionParameter("theta0")
        torsion_constraint.addPerTorsionParameter("k")
        if (
            not self.reference_state
        ):  # Return the reference state forces empty so that we have the same number of forces in the reference state and run state
            for constraint in self.constraints:
                key = list(constraint.keys())[0]
                if "lambda_couple" in constraint[key].keys():
                    if constraint[key]["lambda_couple"] == 1:
                        scale = l
                    elif constraint[key]["lambda_couple"] == -1:
                        scale = 1 - l
                    else:
                        scale = 1
                else:
                    scale = 1

                if len(key) == 2:
                    bond_constraint.addBond(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        constraint[key]["equilibrium"],
                        constraint[key]["force_constant"] * scale,
                    )
                if len(key) == 3:
                    angle_constraint.addAngle(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        self.reaction_to_total_atomid(key[2]),
                        constraint[key]["equilibrium"] * self.deg_to_rad,
                        constraint[key]["force_constant"] * scale,
                    )
                if len(key) == 4:
                    torsion_constraint.addTorsion(
                        self.reaction_to_total_atomid(key[0]),
                        self.reaction_to_total_atomid(key[1]),
                        self.reaction_to_total_atomid(key[2]),
                        self.reaction_to_total_atomid(key[3]),
                        [
                            constraint[key]["equilibrium"] * self.deg_to_rad,
                            constraint[key]["force_constant"] * scale,
                        ],
                    )
        return [bond_constraint, angle_constraint, torsion_constraint]
