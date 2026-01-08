import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
# from joblib import Parallel, delayed
import re
import concurrent.futures

class InterfaceEC():
    def __init__(self, pop_size, init_pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.pop_size = pop_size
        self.init_pop_size = init_pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba

    def get_completion_tokens(self):
        return self.evol.get_completion_tokens()


    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        
        n_create = 2
        
        population = []


        _,pop = self.get_algorithm([],'i1')
        for p in pop:
            population.append(p)
             
        return population
    
    def population_generation_seed(self,seeds):

        population = []

        fitness = self.interface_eval.batch_evaluate([seed['code'] for seed in seeds])

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population

    # def _get_alg(self, pop, operator):
    #     offspring = {
    #         'algorithm': None,
    #         'code': None,
    #         'objective': None,
    #         'other_inf': None,
    #         'novelty': None,
    #         'complexity': None
    #     }
    #
    #     parents = None  # Default parents value
    #
    #     if operator == "i1":
    #         [offspring['code'], offspring['algorithm']] = self.evol.i1()
    #     elif operator == "e1":
    #         parents = self.select.parent_selection(pop, self.m)
    #         [offspring['code'], offspring['algorithm']] = self.evol.e1(parents)
    #     elif operator == "e2":
    #         parents = self.select.parent_selection(pop, self.m)
    #         [offspring['code'], offspring['algorithm']] = self.evol.e2(parents)
    #     elif operator == "m1":
    #         parents = self.select.parent_selection(pop, 1)
    #         [offspring['code'], offspring['algorithm']] = self.evol.m1(parents[0])
    #     elif operator == "m2":
    #         parents = self.select.parent_selection(pop, 1)
    #         [offspring['code'], offspring['algorithm']] = self.evol.m2(parents[0])
    #     elif operator == "m3":
    #         parents = self.select.parent_selection(pop, 1)
    #         [offspring['code'], offspring['algorithm']] = self.evol.m3(parents[0])
    #     elif operator == 'c1':
    #         parents = self.select.parent_selection(pop, 2)
    #         [offspring['code'], offspring['algorithm']] = self.evol.c1(parents)
    #     elif operator == "n1":
    #         [offspring['code'], offspring['algorithm']] = self.evol.n1(pop)
    #     elif operator == "n2":
    #         [offspring['code'], offspring['algorithm']] = self.evol.n2(pop)
    #     elif operator == "ls1":
    #         best_individual = max(pop, key=lambda x: x['objective'])
    #         [offspring['code'], offspring['algorithm']] = self.evol.ls1(best_individual)
    #     elif operator == "ls2":
    #         best_individual = max(pop, key=lambda x: x['objective'])
    #         performance_feedback = f"Current performance: {best_individual['objective']}"
    #         [offspring['code'], offspring['algorithm']] = self.evol.ls2(best_individual, performance_feedback)
    #     elif operator == "ls3":
    #         best_individual = max(pop, key=lambda x: x['objective'])
    #         tabu_list = self.select.parent_selection(pop, 5)
    #         [offspring['code'], offspring['algorithm']] = self.evol.ls3(best_individual, tabu_list)
    #     else:
    #         print(f"Evolution operator [{operator}] has not been implemented ! \n")
    #
    #     return parents, offspring

    def _get_alg(self, pop, operator):
        """
        Generate offspring individual based on operator
        Returns: parents, offspring (complete individual with code, algorithm, prompt_template, etc.)
        """
        parents = None  # Default parents value

        # Get best individual for local search operators
        best_individual = max(pop, key=lambda x: x['objective']) if pop else None


        if operator == "i1":
            offspring = self.evol.i1()
        elif operator == "e1":
            parents = self.select.parent_selection(pop, self.m)
            offspring = self.evol.e1(parents)
        elif operator == "e2":
            parents = self.select.parent_selection(pop, self.m)
            offspring = self.evol.e2(parents)
        elif operator == "m1":
            parents = self.select.parent_selection(pop, 1)
            offspring = self.evol.m1(parents[0])
        elif operator == "m2":
            parents = self.select.parent_selection(pop, 1)
            offspring = self.evol.m2(parents[0])
        elif operator == "m3":
            parents = self.select.parent_selection(pop, 1)
            offspring = self.evol.m3(parents[0])
        elif operator == 'c1':
            parents = self.select.parent_selection(pop, 2)
            offspring = self.evol.c1(parents)
        elif operator == "n1":
            offspring = self.evol.n1(pop)
        elif operator == "n2":
            offspring = self.evol.n2(pop)
        elif operator == "ls1":
            offspring = self.evol.ls1(best_individual)
        elif operator == "ls2":
            performance_feedback = f"Current performance: {best_individual['objective']}"
            offspring = self.evol.ls2(best_individual, performance_feedback)
        elif operator == "ls3":
            tabu_list = self.select.parent_selection(pop, 5)
            offspring = self.evol.ls3(best_individual, tabu_list)
        elif operator == "acga":
            parents = self.select.parent_selection(pop, 1)
            offspring = self.evol.acga(parents[0])
        elif operator == "meta_acga":
            parents = self.select.parent_selection(pop, 1)
            offspring = self.evol.self_construct_with_prompt_evolution(parents[0], pop)
        elif operator == "recombination":
            offspring = self.evol.recombination(pop)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")
            # Return default individual
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None,
                'novelty': None,
                'complexity': None,
                'prompt_template': "",
                'generation': 0,
                'response_id': 0,
                'stdout_filepath': "",
                'code_path': ""
            }
            return parents, offspring


        return parents, offspring

    def get_offspring(self, pop, operator):

        p = None

        try:
            p, offspring = self._get_alg(pop, operator)
            
            code = offspring['code']

            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                    
                p, offspring = self._get_alg(pop, operator)

                code = offspring['code']
                    
                if n_retry > 1:
                    break
                
            # obj = self.interface_eval.batch_evaluate([code],0)[0]
            # offspring['objective'] = np.round(obj, 5)
                
        except Exception as e:
            print(e)

        return p, offspring

    
    def get_algorithm(self, pop, operator):
        offspring_list = []
        if operator == 'i1':
            for _ in range(self.init_pop_size):
                offspring = self.get_offspring(pop, operator)
                offspring_list.append(offspring)
        else:
            for _ in range(5):
                offspring = self.get_offspring(pop, operator)
                offspring_list.append(offspring)
            
        # objs, complexities, novelties, instances = self.interface_eval.batch_evaluate([offspring['code'] for _, offspring in offspring_list], 0)
        objs, complexities = self.interface_eval.batch_evaluate([offspring['code'] for _, offspring in offspring_list], 0)

        for i, (p, offspring) in enumerate(offspring_list):
            offspring['objective'] = np.round(objs[i], 5)
            offspring['complexity'] = complexities[i]
            # offspring['novelty'] = novelties[i]
            # offspring['instance'] = instances[i]

        results = offspring_list


        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    def get_fe(self):
        return self.interface_eval.get_fe()





