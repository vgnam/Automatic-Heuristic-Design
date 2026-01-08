import numpy as np
import json
import random
import time

from .eoh_interface_EC import InterfaceEC
# main class for eoh
import numpy as np
import json
import random
import time

from .eoh_interface_EC import InterfaceEC


class EOH_Island:
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage

        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations
        self.n_island = 2
        self.migration_interval = 3
        self.migration_size = 2

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc

        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    def add2pop(self, population, offspring):
        for off in offspring:
            duplicated = any(ind['objective'] == off['objective'] for ind in population)
            if not duplicated:
                population.append(off)
            elif self.debug_mode:
                print("duplicated result, skipping...")

    def initialize_islands(self, interface_ec):
        """Initialize multiple islands"""
        islands = []
        for i in range(self.n_island):
            print(f"Initializing island {i + 1}/{self.n_island}")
            population = []

            if self.use_seed:
                with open(self.seed_path) as file:
                    data = json.load(file)
                population = interface_ec.population_generation_seed(data)
            elif self.load_pop:
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                population = data[:self.pop_size]
                print("initial population has been loaded!")
            else:
                print(f"creating initial population for island {i + 1}:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)
                print(f"Island {i + 1} initial population created!")

            islands.append(population)

        return islands

    def migrate(self, islands):
        """Perform migration between islands"""
        print("🔄 Migration between islands")
        # Collect best individuals from each island
        migrants = []
        for island in islands:
            # Sort by objective (assuming minimization)
            sorted_island = sorted(island, key=lambda x: x['objective'])
            migrants.extend(sorted_island[:self.migration_size])

        # Shuffle and redistribute
        random.shuffle(migrants)
        num_migrants_per_island = len(migrants) // len(islands)

        for i, island in enumerate(islands):
            start = i * num_migrants_per_island
            end = start + num_migrants_per_island
            incoming = migrants[start:end]
            # Remove worst individuals to make space
            island.sort(key=lambda x: x['objective'], reverse=True)
            del island[:len(incoming)]
            island.extend(incoming)

    def run(self):
        print("- Evolution ready for", self.prob.problem, "-")
        time_start = time.time()

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url,
                                   select=self.select, n_p=self.exp_n_proc,
                                   timeout=self.timeout, use_numba=self.use_numba
                                   )

        # Initialize islands
        islands = self.initialize_islands(interface_ec)

        n_op = len(self.operators)

        # Main evolution loop
        for pop in range(self.n_pop):
            print(f"\n🧬 Generation {pop + 1}/{self.n_pop}")

            # Evolve each island
            for island_idx, population in enumerate(islands):
                print(f"🌍 Evolving island {island_idx + 1}/{self.n_island}")

                for i in range(n_op):
                    op = self.operators[i]
                    print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                    op_w = self.operator_weights[i]

                    if (np.random.rand() < op_w):
                        parents, offsprings = interface_ec.get_algorithm(population, op)
                        self.add2pop(population, offsprings)  # Check duplication, and add the new offspring

                        for off in offsprings:
                            print(" Obj: ", off['objective'], end="|")

                    # Population management
                    size_act = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)

                islands[island_idx] = population  # Update island
                print()

            # Perform migration
            if self.n_island > 1 and (pop + 1) % self.migration_interval == 0 and pop != 0:
                self.migrate(islands)

            # Save each island separately
            for island_idx, population in enumerate(islands):
                if population:  # Check if population is not empty
                    # Save full island population
                    filename = self.output_path + f"island_{island_idx}_population_generation_" + str(pop + 1) + ".json"
                    with open(filename, 'w') as f:
                        json.dump(population, f, indent=5)

                    # Save the best individual from this island
                    best_in_island = min(population, key=lambda x: x['objective'])
                    filename = self.output_path + f"island_{island_idx}_best_population_generation_" + str(
                        pop + 1) + ".json"
                    with open(filename, 'w') as f:
                        json.dump(best_in_island, f, indent=5)

            print(
                f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} m")

            # Print objectives for each island
            for island_idx, population in enumerate(islands):
                print(f"Island {island_idx + 1} Objs: ", end=" ")
                for ind in population:
                    print(str(ind['objective']) + " ", end="")
                print()

        # Final result: return best from all islands
        merged_population = []
        for island in islands:
            merged_population.extend(island)

        if merged_population:
            merged_population = self.manage.population_management(merged_population, len(merged_population))
            best_individual = merged_population[0]

            final_filename = self.output_path + "best_individual_final.json"
            with open(final_filename, 'w') as f:
                json.dump(best_individual, f, indent=5)

            print(f"\n🏁 Evolution completed. Best objective: {best_individual['objective']}")
            return best_individual["code"], final_filename
        else:
            print("❌ No individuals found in any island")
            return None, None


