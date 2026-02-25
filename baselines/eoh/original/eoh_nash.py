import numpy as np
import json
import random
import time

from .eoh_interface_EC import InterfaceEC
from population_encode import get_embedding, compute_cosine_similarity


# main class for eoh
class EOH:

    # initilization
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

        # Game Theory settings
        self.use_game_theory = getattr(paras, 'ec_use_game_theory', True)
        self.game_theory_interval = getattr(paras, 'ec_game_theory_interval', 5)
        self.use_nash_selection = getattr(paras, 'ec_use_nash_selection', True)

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            is_duplicate = False
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
                    is_duplicate = True
                    break
            if not is_duplicate:
                population.append(off)

    # Game Theory methods
    def compute_behavioral_similarity_matrix(self, population):
        """Tính behavioral similarity matrix dựa trên code embeddings"""
        n = len(population)
        if n <= 1:
            return np.array([[1.0]]) if n == 1 else np.array([])

        # Lấy embeddings cho tất cả heuristic
        embeddings = [get_embedding(ind['code']) for ind in population]

        # Compute pairwise cosine similarities
        similarity_matrix = compute_cosine_similarity(embeddings)

        return similarity_matrix, embeddings

    def compute_behavioral_similarity_payoff_matrix(self, population):
        """Tính payoff matrix dựa trên behavioral similarity"""

        # Tính behavioral similarity matrix
        similarity_matrix, embeddings = self.compute_behavioral_similarity_matrix(population)

        n = len(population)
        payoff_matrix = np.zeros((n, n))

        # Payoff dựa trên behavioral relationships
        for i in range(n):
            for j in range(n):
                # Nếu 2 heuristic tương tự → cạnh tranh trực tiếp → payoff thấp
                # Nếu 2 heuristic khác biệt → bổ sung nhau → payoff cao
                behavioral_difference = 1.0 - similarity_matrix[i][j]
                payoff_matrix[i][j] = behavioral_difference

        # Điều chỉnh dựa trên performance thực tế
        performances = np.array([ind['objective'] for ind in population])
        for i in range(n):
            # Heuristic tốt hơn có payoff cao hơn khi đối đầu với heuristic khác
            payoff_matrix[i] = payoff_matrix[i] * performances[i]

        return payoff_matrix, similarity_matrix

    def is_nash_equilibrium(self, payoff_matrix, index, threshold=0.1):
        """Kiểm tra heuristic có phải Nash equilibrium không"""
        if len(payoff_matrix) <= 1:
            return True

        own_payoff = np.mean(payoff_matrix[index])

        max_alt_payoff = 0
        for i in range(len(payoff_matrix)):
            if i != index:
                alt_payoff = np.mean(payoff_matrix[i])
                max_alt_payoff = max(max_alt_payoff, alt_payoff)

        regret = max_alt_payoff - own_payoff
        return regret < threshold

    def select_nash_equilibrium_heuristics(self, population):
        """Chọn heuristic gần Nash equilibrium"""

        if len(population) <= 1:
            return population

        payoff_matrix, _ = self.compute_behavioral_similarity_payoff_matrix(population)
        n = len(population)

        nash_heuristics = []
        nash_info = []

        for i in range(n):
            is_nash = self.is_nash_equilibrium(payoff_matrix, i)
            if is_nash:
                nash_heuristics.append(population[i])
                nash_info.append((is_nash, population[i]['objective'], i))

        # Sắp xếp Nash heuristics theo performance
        if nash_heuristics:
            nash_heuristics.sort(key=lambda x: x['objective'], reverse=True)
            if self.debug_mode:
                print(f"  Found {len(nash_heuristics)} Nash equilibria")
            return nash_heuristics
        else:
            # Nếu không có Nash, chọn heuristic ổn định nhất
            stability_scores = []
            for i in range(n):
                behavioral_variance = np.var(payoff_matrix[i])
                stability = 1.0 / (1.0 + behavioral_variance + 1e-8)
                stability_scores.append((stability, population[i]))

            stability_scores.sort(reverse=True)
            if self.debug_mode:
                print("  No Nash equilibria found, using stability-based selection")
            return [heuristic for _, heuristic in stability_scores[:max(1, len(population) // 3)]]


    def apply_game_theory_selection(self, population, generation):
        """Áp dụng Game Theory selection"""

        if not self.use_game_theory or len(population) <= 1:
            return population

        if generation % self.game_theory_interval == 0:
            print("  🎮 Applying Game Theory Nash Selection...")

            # Chọn Nash equilibrium heuristics
            if self.use_nash_selection:
                nash_heuristics = self.select_nash_equilibrium_heuristics(population)

                if nash_heuristics:
                    # Giữ Nash heuristics + một số heuristic tốt khác
                    remaining = [ind for ind in population if ind not in nash_heuristics]
                    remaining_sorted = sorted(remaining, key=lambda x: x['objective'], reverse=True)

                    # Kết hợp: 70% Nash + 30% tốt nhất còn lại
                    nash_count = min(len(nash_heuristics), len(population) * 7 // 10)
                    remaining_count = len(population) - nash_count

                    selected_population = nash_heuristics[:nash_count]
                    selected_population.extend(remaining_sorted[:remaining_count])

                    if self.debug_mode:
                        nash_count_actual = len([h for h in selected_population if h in nash_heuristics])
                        print(f"    Selected {nash_count_actual}/{len(selected_population)} Nash equilibria")

                    return selected_population

        return population

    # run eoh
    def run(self):

        print("- Evolution ready for", self.prob.problem, "-")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url,
                                   select=self.select, n_p=self.exp_n_proc,
                                   timeout=self.timeout, use_numba=self.use_numba
                                   )

        # initialization
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
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)

                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):
            print(f"\n--- Generation {pop + 1}/{self.n_pop} ---")

            # Apply Game Theory selection periodically
            population = self.apply_game_theory_selection(population, pop)

            # Evolutionary operators
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
                print()

            # Save population to a file
            filename = self.output_path + "population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "best_population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            print(
                f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()

        # Final Game Theory analysis
        if self.use_game_theory and len(population) > 1:
            print("\n🎯 Final Game Theory Analysis:")
            final_nash = self.select_nash_equilibrium_heuristics(population)
            if final_nash:
                print(f"Found {len(final_nash)} robust heuristics (Nash equilibria)")
                # Save Nash heuristics
                nash_filename = self.output_path + "nash_equilibrium_heuristics.json"
                with open(nash_filename, 'w') as f:
                    json.dump([h for h in population if h in final_nash], f, indent=5)
                print(f"Nash heuristics saved to {nash_filename}")

        return population[0]["code"], filename