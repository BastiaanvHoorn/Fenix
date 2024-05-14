from EVB.timer import Timer
from EVB.system_builder import System_builder

import pickle
import numpy as np
import openmm as mm
import openmm.app as mmapp
import openmm.unit as mmunit


class FEP_driver:

    # import openmm.app as mmapp
    # import openmm.unit as unit
    # import openmm as mm
    # import math
    # import os

    def __init__(
        self,
        run_folder,
        data_folder,
        system_builder: System_builder,
        platform=mm.Platform.getPlatformByName("Reference"),
    ):

        self.run_folder = run_folder
        self.data_folder = data_folder
        self.system_builder: System_builder = system_builder
        self.Lambda = system_builder.Lambda
        self.platform = platform

    def run_FEP(
        self,
        equilliberation_steps,
        total_sample_steps,
        write_step,
        lambda_0_equilliberation_steps,
        step_size,
    ):
        assert (
            total_sample_steps % write_step == 0
        ), "write_step must be a factor of total_sample_steps"
        assert (
            total_sample_steps >= 2 * write_step
        ), "total_sample_steps must be at least 2*write_step"

        self.total_snapshots = total_sample_steps / write_step * len(self.Lambda)
        print("Lambda: ", self.Lambda)
        print("Total lambda points: ", len(self.Lambda))
        print("Snapshots per lambda: ", total_sample_steps / write_step)
        print(
            "Snapshots to be recorded: ",
            self.total_snapshots,
        )
        print(
            "Total simulation steps: ",
            (total_sample_steps + equilliberation_steps) * len(self.Lambda)
            + lambda_0_equilliberation_steps,
        )
        print("System time per snapshot: ", step_size * write_step, " ps")
        print("System time per frame: ", step_size * total_sample_steps, " ps")
        print(
            "Total system time: ",
            step_size * total_sample_steps * len(self.Lambda),
            " ps",
        )
        integrator_temperature = self.system_builder.temperature * mmunit.kelvin
        integrator_friction_coeff = 1 / mmunit.picosecond
        # integrator_step_size = 0.001 * unit.picoseconds

        # if self.init_positions == None:
        #     self.init_positions = rea_ff_gen.molecule.get_coordinates_in_angstrom()*0.1
        positions = self.system_builder.positions

        timer = Timer(len(self.Lambda))
        estimated_time_remaining = None
        step_size_factor = 1
        value_error_count = 0
        for i, l in enumerate(self.Lambda):
            succeeded_step = False
            while not succeeded_step:
                try:
                    timer.start()
                    # print(f"lambda = {l}",end='\r')
                    integrator = mm.LangevinMiddleIntegrator(
                        integrator_temperature,
                        integrator_friction_coeff,
                        step_size_factor * step_size * mmunit.picoseconds,
                    )
                    # integrator = mm.VerletIntegrator(integrator_step_size)
                    # todo do I need to care about the topology
                    simulation = mmapp.Simulation(
                        self.system_builder.topology,
                        self.system_builder.systems[l],
                        integrator,
                        platform=self.platform,
                    )

                    simulation.context.setPositions(positions)

                    if estimated_time_remaining:
                        time_estimate_str = ", " + timer.get_time_str(
                            estimated_time_remaining
                        )
                    else:
                        time_estimate_str = ""

                    print(f"lambda = {l}" + time_estimate_str)
                    print("Minimizing energy")
                    simulation.reporters.append(
                        mmapp.PDBReporter(
                            f"{self.run_folder}/minim_{l:.3f}.pdb",
                            write_step / step_size_factor,
                        )
                    )

                    simulation.minimizeEnergy()

                    # Equiliberate
                    # todo write these reporters on my own
                    # todo add lambda value to the reporter
                    print("Running equilliberation")
                    if l == 0:
                        simulation.step(
                            lambda_0_equilliberation_steps
                            + equilliberation_steps / step_size_factor
                        )
                    else:
                        simulation.step(equilliberation_steps / step_size_factor)
                    print("Running sampling")
                    simulation.reporters.append(
                        mmapp.PDBReporter(
                            f"{self.run_folder}/traj{l:.3f}.pdb",
                            write_step / step_size_factor,
                        )
                    )
                    simulation.reporters.append(
                        mmapp.StateDataReporter(
                            f"{self.run_folder}/ETV{l:.3f}.dat",
                            write_step / step_size_factor,
                            step=True,
                            potentialEnergy=True,
                            temperature=True,
                            volume=True,
                        )
                    )
                    simulation.step(total_sample_steps)
                    state = simulation.context.getState(getPositions=True)
                    positions = state.getPositions()
                    estimated_time_remaining = timer.stop_and_calculate(i + 1)
                    succeeded_step = True
                except ValueError:
                    value_error_count += 1
                    if value_error_count > 5:
                        raise ValueError(
                            "Value_error encountered too many times in the same lambda, aborting"
                        )
                    step_size_factor = step_size_factor / 2
                    print(
                        f"Encountered value_error, retrying with smaller step size {step_size*step_size_factor}ps"
                    )

            step_size_factor = 1
            value_error_count = 1

        print("Merging output files")
        self.merge_traj_pdb()
        self.merge_state()

    # Utility functions for merging output
    def merge_state(self):
        print("merging ETV files")

        data = np.array([]).reshape(0, 5)
        np.savetxt(
            f"{self.data_folder}/ETV_combined.dat", data, header="step,E,T,V,lambda"
        )
        step = 0
        for l in self.Lambda:
            _filename = f"{self.run_folder}/ETV{l:.3f}.dat"
            data = np.genfromtxt(_filename, delimiter=",")

            # discard the equilliberation steps
            data[:, 0] -= data[0, 0]
            data[:, 0] += step
            step = data[-1, 0]
            data = np.column_stack((data, np.full(data.shape[0], l)))

            with open(f"{self.data_folder}/ETV_combined.dat", "ab") as f:
                np.savetxt(f, data)

    def merge_traj_pdb(self):
        print("merging pdb files")
        output = ""
        with open(
            f"{self.data_folder}/traj_combined.pdb", "w", encoding="utf-8"
        ) as file:
            file.write(output)
        frame = 1
        crystline = None
        for l in self.Lambda:
            print("Lambda = ", l)
            filename = f"{self.run_folder}/traj{l:.3f}.pdb"

            with open(filename, "r", encoding="utf-8") as file:
                file_contents = file.read()

            # print(file_contents)
            for line in file_contents.split("\n"):
                parts = line.split()
                if len(parts) == 0:
                    continue
                if parts[0] == "REMARK":
                    continue
                if parts[0] == "MODEL":
                    line = f"MODEL{frame: >9}\n"
                    frame += 1
                if parts[0] == "CONECT":
                    continue
                if parts[0] == "END":
                    continue
                if parts[0] == "CRYST1":
                    crystline = line
                    continue

                output += line + "\n"
            # Write every frame seperately already to the file and empty the output string, otherwise the output string will become too large to handle nicely
            with open(
                f"{self.data_folder}/traj_combined.pdb", "a", encoding="utf-8"
            ) as file:
                file.write(output)
            output = ""
        if crystline:
            output += crystline + "\n"
        output += "END"
        with open(
            f"{self.data_folder}/traj_combined.pdb", "a", encoding="utf-8"
        ) as file:
            file.write(output)

    # The variable names need documentation and/or renaming
    # interpolated_potential: recalculate also with all interpolated topologies
    # all_potentials: at every step, recalculate every interpolated potential (you should be able to do cool stuff with BAR with this)
    # force_contributions: calculate the contributions of every force
    def recalculate(self, interpolated_potential=False, force_contributions=False):
        integrator_step_size = 1

        # reference simulations are from the reference systems, run simulations are using the lambda = 0 and lambda = 1, shouldn't be different but used for verification #todo scrap this
        simulation_reactant_reference = mmapp.Simulation(
            self.system_builder.topology,
            self.system_builder.systems["reactant"],
            mm.VerletIntegrator(integrator_step_size),
            platform=self.platform,
        )
        simulation_product_reference = mmapp.Simulation(
            self.system_builder.topology,
            self.system_builder.systems["product"],
            mm.VerletIntegrator(integrator_step_size),
            platform=self.platform,
        )
        simulation_reactant_run = mmapp.Simulation(
            self.system_builder.topology,
            self.system_builder.systems[0],
            mm.VerletIntegrator(integrator_step_size),
            platform=self.platform,
        )
        simulation_product_run = mmapp.Simulation(
            self.system_builder.topology,
            self.system_builder.systems[1],
            mm.VerletIntegrator(integrator_step_size),
            platform=self.platform,
        )

        lsims = []
        if interpolated_potential:
            # Make all other simulations again
            for l in self.Lambda:
                lsims.append(
                    mmapp.Simulation(
                        self.system_builder.topology,
                        self.system_builder.systems[l],
                        mm.VerletIntegrator(integrator_step_size),
                        platform=self.platform,
                    )
                )

        # loop over lambda instead
        estimated_time_remaining = None

        calculated_groups = set()
        force_names = []
        for force in self.system_builder.systems["reactant"].getForces():
            group = force.getForceGroup()
            if group in calculated_groups:
                continue
            calculated_groups.add(group)
            if not group == 0:
                force_names.append(force.getName())
        Energies = []
        E1f_ref = []
        E2f_ref = []
        E1f_run = []
        E2f_run = []
        Efm = []
        np.savetxt(
            f"{self.data_folder}/E1f_ref.dat", E1f_ref, header=",".join(force_names)
        )
        np.savetxt(
            f"{self.data_folder}/E2f_ref.dat", E2f_ref, header=",".join(force_names)
        )
        np.savetxt(
            f"{self.data_folder}/E1f_run.dat", E1f_run, header=",".join(force_names)
        )
        np.savetxt(
            f"{self.data_folder}/E2f_run.dat", E2f_run, header=",".join(force_names)
        )
        if interpolated_potential:
            np.savetxt(f"{self.data_folder}/Efm.dat", Efm)
            header = "E1_ref,E2_ref,E1_run,E2_run,E_m"
        else:
            header = "E1_ref,E2_ref,E1_run,E2_run"
        np.savetxt(
            f"{self.data_folder}/Energies.dat",
            Energies,
            header=header,
        )
        timer = Timer(len(self.Lambda))
        for l in self.Lambda:
            timer.start()

            if estimated_time_remaining:
                time_estimate_str = ", " + timer.get_time_str(estimated_time_remaining)
            else:
                time_estimate_str = ""

            print(f"Recalculating Lambda {l}" + time_estimate_str)
            samples = mmapp.PDBFile(f"{self.run_folder}/traj{l:.3f}.pdb")
            for i in range(samples.getNumFrames()):
                positions = samples.getPositions(True, i)
                # Gets the right frame of the calculation, to pick out the right lambda and the right simulation

                e1_ref, e1f_ref = self.calculate_energy(
                    simulation_reactant_reference, positions, force_contributions
                )
                e2_ref, e2f_ref = self.calculate_energy(
                    simulation_product_reference, positions, force_contributions
                )

                e1_run, e1f_run = self.calculate_energy(
                    simulation_reactant_run, positions, force_contributions
                )
                e2_run, e2f_run = self.calculate_energy(
                    simulation_product_run, positions, force_contributions
                )

                em = 0
                efm = []
                if interpolated_potential:

                    em, efm = self.calculate_energy(
                        lsims[np.where(np.array(self.Lambda) == l)[0][0]],
                        positions,
                        force_contributions,
                    )

                if interpolated_potential:
                    Energies.append([e1_ref, e2_ref, e1_run, e2_run, em])
                else:
                    Energies.append([e1_ref, e2_ref, e1_run, e2_run])
                if force_contributions:
                    E1f_ref.append(e1f_ref)
                    E2f_ref.append(e2f_ref)
                    E1f_run.append(e1f_run)
                    E2f_run.append(e2f_run)
                    if interpolated_potential:
                        Efm.append(efm)

            with open(f"{self.data_folder}/Energies.dat", "ab") as f:
                np.savetxt(f, Energies)
            if force_contributions:
                with open(f"{self.data_folder}/E1f_ref.dat", "ab") as f:
                    np.savetxt(f, E1f_ref)
                with open(f"{self.data_folder}/E2f_ref.dat", "ab") as f:
                    np.savetxt(f, E2f_ref)
                with open(f"{self.data_folder}/E1f_run.dat", "ab") as f:
                    np.savetxt(f, E1f_run)
                with open(f"{self.data_folder}/E2f_run.dat", "ab") as f:
                    np.savetxt(f, E2f_run)
                if interpolated_potential:
                    with open(f"{self.data_folder}/Efm.dat", "ab") as f:
                        np.savetxt(f, Efm)
            Energies = []
            E1f_ref = []
            E2f_ref = []
            E1f_run = []
            E2f_run = []
            Efm = []
            estimated_time_remaining = timer.stop_and_calculate(l + 1)

        data = {}
        return data

    def calculate_energy(self, sim, positions, force_contributions):
        sim.context.setPositions(positions)
        context = sim.context
        E = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(mmunit.kilojoule_per_mole)
        )
        Ef = []
        if force_contributions:
            calculated_groups = set()
            for force in context.getSystem().getForces():
                group = force.getForceGroup()
                if group in calculated_groups:
                    continue
                calculated_groups.add(group)
                if not group == 0:
                    Ef.append(
                        context.getState(getEnergy=True, groups={group})
                        .getPotentialEnergy()
                        .value_in_unit(mmunit.kilojoule_per_mole)
                    )
        return E, Ef
