import numpy as np
import json
import random
import time
# from population_encode import get_embedding, compute_cosine_similarity, similarity
from utils.utils import multi_chat_completion

from .eoh_interface_EC import InterfaceEC


# main class for eoh
class EOH_Novelty:

    def __init__(self, paras, problem, select, manage, **kwargs):

        self.best_fe_record = []

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
        self.init_pop_size = paras.init_pop_size

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

        self.problem_name = problem.problem_description
        self.root_dir = problem.root_dir

        self.max_fe = paras.max_fe
        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)

    # run eoh
    def run(self):
        print("- Evolution ready for", self.prob.problem, "-")

        time_start = time.time()

        interface_prob = self.prob
        interface_ec = InterfaceEC(
            self.pop_size, self.init_pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
            self.debug_mode, interface_prob,
            use_local_llm=self.use_local_llm, url=self.url,
            select=self.select, n_p=self.exp_n_proc,
            timeout=self.timeout, use_numba=self.use_numba
        )

        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data)
            filename = self.output_path + "population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop:
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                population = data
                n_start = self.load_pop_id
            else:
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)
                filename = self.output_path + "population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # Main loop
        n_op = len(self.operators)
        novelty_pulsation_interval = 2  # Apply novelty every 5 generations

        best_so_far = None  # best individual across generations
        best_fe_record = []  # store (FE, best_obj)

        pop = 0
        while interface_ec.get_fe() < self.max_fe:
            # Apply novelty pulsation periodically
            # if pop % novelty_pulsation_interval == 0:
            #     print("\n--- Applying novelty pulsation ---")
            #
            #     if np.random.rand() < 1:
            #         print("\n--- Using novelty operator---")
            #
            #         parents, offsprings = interface_ec.get_algorithm(population, 'n1')
            #
            #         self.add2pop(population, offsprings)
            #
            #         for off in offsprings:
            #             print(" Obj: ", off['objective'], end="|")


            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w = self.operator_weights[i]

                if np.random.rand() < op_w:
                    parents, offsprings = interface_ec.get_algorithm(population, op)

                    self.add2pop(population, offsprings)

            print("\n--- Applying Iterated Local Search (ILS) ---")

            ils_offsprings = self.iterated_deepening_search(
                population=population,
                interface_ec=interface_ec,
                interface_prob=interface_prob,
                max_depth=3
            )
            # Thêm các offspring từ ILS vào population
            for offspring in ils_offsprings:
                self.add2pop(population, [offspring])

            # Quản lý lại population
            size_act = min(len(population), self.pop_size)
            population = self.manage.population_management(population, size_act)

            print(f"  ILS added {len(ils_offsprings)} improved individuals.")

            for off in population:
                print(" Obj: ", off['objective'], end="|")

            # Save current population
            filename = self.output_path + f"population_generation_{pop + 1}.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save best individual
            best_filename = self.output_path + f"best_population_generation_{pop + 1}.json"
            with open(best_filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            print(
                f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} m")
            print("Pop Objs: ", end=" ")
            for ind in population:
                print(ind['objective'], end=" ")
            print()

            pop += 1

        return population[0]["code"], filename

    def iterated_deepening_search(self, population, interface_ec, interface_prob, max_depth=5):

        best_individual = max(population, key=lambda x: x['objective'])
        current = best_individual
        offspring_list = []

        print(f"--- Starting Iterated Deepening Search (max depth = {max_depth}) ----")

        for depth in range(1, max_depth + 1):
            print(f"  Depth {depth}/{max_depth}...")
            depth_best = current  # Best tại độ sâu hiện tại

            for step in range(depth):
                operator = random.choice(['ls1', 'ls2', 'ls3'])

                parents, offspring = interface_ec.get_algorithm([current], operator)

                # Accept nếu tốt hơn
                if offspring[0]['objective'] >= current['objective']:
                    current = offspring[0]
                    if offspring[0]['objective'] > depth_best['objective']:
                        depth_best = offspring[0]


            # Lưu kết quả tốt nhất tại độ sâu này
            if depth_best != best_individual:
                offspring_list.append(depth_best)

            # Reset về best overall để bắt đầu độ sâu mới
            current = best_individual

        return offspring_list







