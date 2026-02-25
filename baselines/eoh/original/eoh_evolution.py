import re
import time
from .interface_LLM import InterfaceAPI as InterfaceLLM
import re
from utils.utils import multi_chat_completion
import os
import numpy as np
import logging

input = lambda: ...

# class Evolution():
#
#     def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
#         # -------------------- RZ: use local LLM --------------------
#         assert 'use_local_llm' in kwargs
#         assert 'url' in kwargs
#         self._use_local_llm = kwargs.get('use_local_llm')
#         self._url = kwargs.get('url')
#         # -----------------------------------------------------------
#
#         # set prompt interface
#         #getprompts = GetPrompts()
#         self.prompt_task         = prompts.get_task()
#         self.prompt_func_name    = prompts.get_func_name()
#         # self.prompt_func_signature = prompts.get_func_signature()
#         # self.prompt_fun_desc = prompts.get_func_desc()
#         self.prompt_func_inputs  = prompts.get_func_inputs()
#         self.prompt_func_outputs = prompts.get_func_outputs()
#         self.prompt_inout_inf    = prompts.get_inout_inf()
#         self.prompt_other_inf    = prompts.get_other_inf()
#         self.prompt_knowledge    = prompts.get_knowledge()
#
#         self.instances_dir = f'{prompts.root_dir}/problems/{prompts.problem}/dataset'
#         self.data_file = [f for f in os.listdir(self.instances_dir) if f.startswith("train")][0]
#         self.data = np.load(os.path.join(self.instances_dir, self.data_file), allow_pickle=True)
#
#         if len(self.prompt_func_inputs) > 1:
#             self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
#         else:
#             self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"
#
#         if len(self.prompt_func_outputs) > 1:
#             self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
#         else:
#             self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"
#
#         # set LLMs
#         self.api_endpoint = api_endpoint
#         self.api_key = api_key
#         self.model_LLM = model_LLM
#         self.debug_mode = debug_mode # close prompt checking
#
#         # -------------------- RZ: use local LLM --------------------
#
#         self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)
#
#     def get_prompt_i1(self):
#
#         prompt_content = self.prompt_task+"\n"\
# "First, describe your new algorithm and main steps in one sentence. \
# The description must be inside a brace. Next, implement it in Python as a function named \
# "+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
# +self.joined_inputs+" `. The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
# +self.joined_outputs+". "+self.prompt_inout_inf+" "\
# +self.prompt_other_inf+"\n"+"Do not give additional explanations."
#         return prompt_content
#
#
#     def get_prompt_e1(self,indivs):
#         prompt_indiv = ""
#         for i in range(len(indivs)):
#             prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"
#
#         prompt_content = self.prompt_task+"\n"\
# "I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
# +prompt_indiv+\
# "Please help me create a new algorithm that has a totally different form from the given ones. \n"\
# "First, describe your new algorithm and main steps in one sentence. \
# The description must be inside a brace. Next, implement it in Python as a function named \
# "+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
# +self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
# +self.joined_outputs+". "+self.prompt_inout_inf+" "\
# +self.prompt_other_inf+"\n"+"Do not give additional explanations."
#         return prompt_content
#
#     def get_prompt_e2(self,indivs):
#         prompt_indiv = ""
#         for i in range(len(indivs)):
#             prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"
#
#         prompt_content = self.prompt_task+"\n"\
# "I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
# +prompt_indiv+\
# "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
# "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
# The description must be inside a brace. Thirdly, implement it in Python as a function named \
# "+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
# +self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
# +self.joined_outputs+". "+self.prompt_inout_inf+" "\
# +self.prompt_other_inf+"\n"+"Do not give additional explanations."
#         return prompt_content
#
#     def get_prompt_m1(self,indiv1):
#         prompt_content = self.prompt_task+"\n"\
# "I have one algorithm with its code as follows. \
# Algorithm description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# Please assist me in creating a new algorithm which is a mutated version of the algorithm provided and better than the current algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments. \n"\
# "First, describe your new algorithm and main steps in one sentence. \
# The description must be inside a brace. Next, implement it in Python as a function named \
# "+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
# +self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
# +self.joined_outputs+". "+self.prompt_inout_inf+" "\
# +self.prompt_other_inf+"\n"+"Do not give additional explanations."
#         return prompt_content
#
#     def get_prompt_m2(self,indiv1):
#         prompt_content = self.prompt_task+"\n"\
# "I have one algorithm with its code as follows. \
# Algorithm description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"\
# "First, describe your new algorithm and main steps in one sentence. \
# The description must be inside a brace. Next, implement it in Python as a function named \
# "+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
# +self.joined_inputs+". "+self.prompt_inout_inf+" "\
# +self.prompt_other_inf+"\n"+"Do not give additional explanations."
#         return prompt_content
#
#
#     def _get_alg(self,prompt_content):
#
#         response = self.interface_llm.get_response(prompt_content)
#
#         algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
#         if len(algorithm) == 0:
#             if 'python' in response:
#                 algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
#             elif 'import' in response:
#                 algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
#             else:
#                 algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)
#
#         code = re.findall(r"import.*return", response, re.DOTALL)
#         if len(code) == 0:
#             code = re.findall(r"def.*return", response, re.DOTALL)
#
#         n_retry = 1
#         while (len(algorithm) == 0 or len(code) == 0):
#             if self.debug_mode:
#                 print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")
#
#             response = self.interface_llm.get_response(prompt_content)
#
#             algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
#             if len(algorithm) == 0:
#                 if 'python' in response:
#                     algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
#                 elif 'import' in response:
#                     algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
#                 else:
#                     algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)
#
#             code = re.findall(r"import.*return", response, re.DOTALL)
#             if len(code) == 0:
#                 code = re.findall(r"def.*return", response, re.DOTALL)
#
#             if n_retry > 3:
#                 break
#             n_retry +=1
#
#         algorithm = algorithm[0]
#         code = code[0]
#
#         code_all = code+" "+", ".join(s for s in self.prompt_func_outputs)
#
#         if "import" not in code_all:
#             code_all = "import numpy as np\nimport random\nimport math\nimport scipy\nimport torch\n" + code_all
#
#         return [code_all, algorithm]
#
#
#     def i1(self):
#
#         prompt_content = self.get_prompt_i1()
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def e1(self,parents):
#
#         prompt_content = self.get_prompt_e1(parents)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def e2(self,parents):
#
#         prompt_content = self.get_prompt_e2(parents)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def m1(self,parents):
#
#         prompt_content = self.get_prompt_m1(parents)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def m2(self,parents):
#
#         prompt_content = self.get_prompt_m2(parents)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def get_prompt_n1(self, population, k=3):
#
#         if len(population) > k:
#             sorted_pop = sorted(population,
#                                 key=lambda x: x.get('novelty_score', 0),
#                                 reverse=True)
#             selected_indivs = sorted_pop[:k]
#         else:
#             selected_indivs = population
#
#         prompt_indiv = ""
#         for i, ind in enumerate(selected_indivs):
#             prompt_indiv += f"No.{i + 1} algorithm and the corresponding code are: \n"
#             prompt_indiv += ind['algorithm'] + "\n" + ind['code'] + "\n"
#
#         prompt_content = self.prompt_task + "\n" \
#                                             f"I have {len(selected_indivs)} existing algorithms that are known to be highly novel and diverse. Their codes are as follows: \n" \
#                          + prompt_indiv + \
#                          "Please create a new algorithm that is fundamentally different in approach and structure from all of these existing algorithms. " \
#                          "The new algorithm should explore a completely different problem-solving paradigm. \n" \
#                          "First, describe your new algorithm and main steps in one sentence without special character or letter. \
#                          The description must be inside a brace. Next, implement it in Python as a function named " \
#                          + self.prompt_func_name + ". This function should accept " + str(
#             len(self.prompt_func_inputs)) + " input(s): " \
#                          + self.joined_inputs + ". The function should return " + str(
#             len(self.prompt_func_outputs)) + " output(s): " \
#                          + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
#                          + self.prompt_other_inf + "\n" \
#                                                    "Focus on maximum novelty and avoid similar approaches to the provided algorithms. " \
#                                                    "Do not give additional explanations. Output code only"
#
#         return prompt_content
#
#     def n1(self, population):
#
#         prompt_content = self.get_prompt_n1(population)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [n1] : \n", prompt_content)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def get_prompt_ls1(self, individual):
#         """Neighborhood Search - Tìm giải pháp lân cận tốt hơn"""
#
#         prompt_content = self.prompt_task + "\n" \
#                                             "I have the following algorithm that needs local improvement:\n" \
#                                             "Algorithm description: " + individual['algorithm'] + "\n" \
#                                                                                                   "Code:\n" + \
#                          individual['code'] + "\n\n" \
#                                               "Please help me create a NEIGHBORING algorithm that makes small modifications " \
#                                               "to explore the local search space.\n" \
#                                               "First, identify one or two components that can be slightly modified.\n" \
#                                               "Second, make small changes such as:\n" \
#                                               "- Adjusting weight coefficients\n" \
#                                               "- Changing exploration vs exploitation balance\n" \
#                                               "- Modifying threshold values\n" \
#                                               "- Slightly altering the decision logic\n" \
#                                               "Third, describe your new algorithm and main steps in one sentence. " \
#                                               "The description must be inside a brace. \n" \
#                                               "Fourth, implement the improved version in Python as a function named " \
#                          + self.prompt_func_name + ". This function should accept " + str(
#             len(self.prompt_func_inputs)) + " input(s): " \
#                          + self.joined_inputs + ". The function should return " + str(
#             len(self.prompt_func_outputs)) + " output(s): " \
#                          + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
#                          + self.prompt_other_inf + "\n" \
#                                                    "IMPORTANT: Make only small, focused changes to explore local neighborhood.\n" \
#                                                    "Do not give additional explanations. "
#
#         return prompt_content
#
#     def get_prompt_ls2(self, individual, performance_feedback=""):
#         """Hill Climbing - Cải thiện theo hướng gradient"""
#
#         feedback_section = ""
#         if performance_feedback:
#             feedback_section = "Performance feedback: " + performance_feedback + "\n\n"
#
#         prompt_content = self.prompt_task + "\n" \
#                                             "I have the following algorithm that needs improvement through hill climbing:\n" \
#                                             "Algorithm description: " + individual['algorithm'] + "\n" \
#                                                                                                   "Code:\n" + \
#                          individual['code'] + "\n\n" \
#                          + feedback_section + \
#                          "Please help me perform one step of hill climbing local search:\n" \
#                          "Analyze the current algorithm's weaknesses or suboptimal components.\n" \
#                          "Make a single improvement that addresses the main weakness.\n" \
#                          "Describe your improved algorithm and main steps in one sentence. " \
#                          "The description must be inside a brace. \n" \
#                          "Then, implement the improved version in Python as a function named " \
#                          + self.prompt_func_name + ". This function should accept " + str(
#             len(self.prompt_func_inputs)) + " input(s): " \
#                          + self.joined_inputs + ". The function should return " + str(
#             len(self.prompt_func_outputs)) + " output(s): " \
#                          + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
#                          + self.prompt_other_inf + "\n" \
#                                                    "Focus on making ONE significant improvement rather than many small changes.\n" \
#                                                    "Do not give additional explanations."
#
#         return prompt_content
#
#     def get_prompt_ls3(self, individual, tabu_list=[]):
#
#         tabu_section = ""
#         if tabu_list:
#             tabu_section = "Recently tried modifications (avoid these):\n"
#             for i, tabu_item in enumerate(tabu_list[-3:]):  # Last 3 tabu items
#                 tabu_section += f"{i + 1}. {tabu_item}\n"
#             tabu_section += "\n"
#
#         prompt_content = self.prompt_task + "\n" \
#                                             "I have the following algorithm that needs diversification through tabu search:\n" \
#                                             "Algorithm description: " + individual['algorithm'] + "\n" \
#                                                                                                   "Code:\n" + \
#                          individual['code'] + "\n\n" \
#                          + tabu_section + \
#                          "Please help me create a diversified version that explores new areas:\n" \
#                          "First, identify what has been recently tried (tabu) and avoid similar approaches.\n" \
#                          "Second, make changes that are significantly different from previous attempts.\n" \
#                          "Third, describe your new algorithm and main steps in one sentence. " \
#                          "The description must be inside a brace. \n" \
#                          "Fourth, implement the diversified version in Python as a function named " \
#                          + self.prompt_func_name + ". This function should accept " + str(
#             len(self.prompt_func_inputs)) + " input(s): " \
#                          + self.joined_inputs + ". The function should return " + str(
#             len(self.prompt_func_outputs)) + " output(s): " \
#                          + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
#                          + self.prompt_other_inf + "\n" \
#                                                    "Focus on DIVERSIFICATION and AVOIDING recently tried modifications.\n" \
#                                                    "Do not give additional explanations. "
#
#         return prompt_content
#
#     def ls1(self, individual):
#         """Neighborhood Search Local Search"""
#         prompt_content = self.get_prompt_ls1(individual)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for local search [ ls1 - neighborhood ] : \n", prompt_content)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def ls2(self, individual, performance_feedback=""):
#         """Hill Climbing Local Search"""
#         prompt_content = self.get_prompt_ls2(individual, performance_feedback)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for local search [ ls2 - hill climbing ] : \n", prompt_content)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def ls3(self, individual, tabu_list=[]):
#         """Tabu Search Local Search"""
#         prompt_content = self.get_prompt_ls3(individual, tabu_list)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for local search [ ls3 - tabu search ] : \n", prompt_content)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def get_prompt_n2(self, population, k=3):
#         """
#         Tạo prompt cho Surprise Search - tạo heuristic bất ngờ, khác biệt hoàn toàn
#         """
#         # Chọn k heuristic có novelty score cao nhất (mới lạ nhất)
#         if len(population) > k:
#             # Sắp xếp theo novelty score giảm dần (nếu có)
#             sorted_pop = sorted(population,
#                                 key=lambda x: x.get('novelty_score', 0),
#                                 reverse=True)
#             selected_indivs = sorted_pop[:k]
#         else:
#             selected_indivs = population
#
#         prompt_indiv = ""
#         for i, ind in enumerate(selected_indivs):
#             prompt_indiv += f"No.{i + 1} algorithm (highly novel) and the corresponding code are: \n"
#             prompt_indiv += ind['algorithm'] + "\n" + ind['code'] + "\n"
#
#         prompt_content = self.prompt_task + "\n" \
#                                             f"I have {len(selected_indivs)} existing algorithms that are known to be highly novel and diverse. Their codes are as follows: \n" \
#                          + prompt_indiv + \
#                          "However, I want you to create a COMPLETELY SURPRISING algorithm that is fundamentally different in approach and structure from ALL of these existing algorithms. " \
#                          "The new algorithm should explore a TOTALLY UNEXPECTED problem-solving paradigm that no one would anticipate. \n" \
#                          "First, describe your surprising algorithm and main steps in one sentence but do NOT use special characters here." \
#                          "The description must be inside a brace {}. Next, implement it in Python as a function named" \
#                          + self.prompt_func_name + ". This function should accept " + str(
#             len(self.prompt_func_inputs)) + " input(s): " \
#                          + self.joined_inputs + ". The function should return " + str(
#             len(self.prompt_func_outputs)) + " output(s): " \
#                          + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
#                          + self.prompt_other_inf + "\n" \
#                                                    "IMPORTANT: Focus on MAXIMUM SURPRISE - create something that would be considered 'out of the box' thinking. " \
#                                                    "Avoid any approaches similar to the provided algorithms. Be creative and unexpected!\n" \
#                                                    "Do not give additional explanations."
#
#         return prompt_content
#
#     def n2(self, population):
#
#         prompt_content = self.get_prompt_n2(population)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [ surprise_search n2 ] : \n", prompt_content)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def get_completion_tokens(self):
#         return self.interface_llm.get_completion_tokens()
#
#     def get_prompt_c1(self, parents):
#         """
#         Prompt cho crossover: kết hợp hai thuật toán cha để tạo con
#         """
#         prompt_content = self.prompt_task + "\n" \
#                                             """
#                                             I have two high-performing algorithms that solve this problem effectively:
#
#                                             Parent Algorithm 1:
#                                             Description: {parent1_algo}
#                                             Code:
#                                             {parent1_code}
#
#                                             Parent Algorithm 2:
#                                             Description: {parent2_algo}
#                                             Code:
#                                             {parent2_code}
#
#                                             Please create a new algorithm by COMBINING elements between these two parents.
#
#                                             The new algorithm should be:
#                                             - Functionally complete and coherent
#                                             - Superior to or at least competitive with both parents
#                                             - Not just a simple mixture but a thoughtful integration
#
#                                             First, describe your hybrid algorithm and main steps in one sentence.
#                                             The description must be inside a brace.
#                                             Second, implement it in Python as a function named {func_name}.
#                                             This function should accept {num_inputs} input(s): {inputs}.
#                                             The function should return {num_outputs} output(s): {outputs}.
#                                             {inout_inf} {other_inf}
#
#                                             Do not give additional explanations. Output code only
#                                             """.format(
#             parent1_algo=parents[0]['algorithm'],
#             parent1_code=parents[0]['code'],
#             parent2_algo=parents[1]['algorithm'],
#             parent2_code=parents[1]['code'],
#             func_name=self.prompt_func_name,
#             num_inputs=len(self.prompt_func_inputs),
#             inputs=self.joined_inputs,
#             num_outputs=len(self.prompt_func_outputs),
#             outputs=self.joined_outputs,
#             inout_inf=self.prompt_inout_inf,
#             other_inf=self.prompt_other_inf
#         )
#
#         return prompt_content
#
#     def c1(self, parents):
#
#         prompt_content = self.get_prompt_c1(parents)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for crossover [ c1 ] : \n", prompt_content)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def get_prompt_m3(self, indiv1):
#         prompt_content = "First, you need to identify the main components in the function below. \
#     Next, analyze whether any of these components can be overfit to the in-distribution instances. \
#     Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
#     Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n" + indiv1[
#             'code'] + "\n" \
#                          + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
#         return prompt_content
#
#     def m3(self, parents):
#
#         prompt_content = self.get_prompt_m3(parents)
#
#         if self.debug_mode:
#             print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         [code_all, algorithm] = self._get_alg(prompt_content)
#
#         if self.debug_mode:
#             print("\n >>> check designed algorithm: \n", algorithm)
#             print("\n >>> check designed code: \n", code_all)
#             print(">>> Press 'Enter' to continue")
#             input()
#
#         return [code_all, algorithm]
#
#     def response_to_individual(self, code, algorithm, response_id, file_name=None) -> dict:
#         """
#         Convert LLM response to individual dictionary
#         """
#         # Create output directory
#         outdir = './evaluations/'
#         if not os.path.isdir(outdir):
#             os.mkdir(outdir)
#
#         # Generate unique ID
#         runid = hash(code)
#
#         # Set file names
#         file_name = outdir + f"problem_eval{runid}.txt" if file_name is None else file_name + ".txt"
#         with open(file_name, 'w') as file:
#             file.writelines(code + '\n')
#
#         std_out_filepath = outdir + f"problem_eval{runid}_stdout.txt" if file_name is None else file_name.rstrip(
#             ".txt") + "_stdout.txt"
#
#         # Initialize with default prompt template for Meta-ACGA
#         default_prompt_template = f"""
#     You are a self-improving heuristic for {self.prompt_task}.
#     Your task is to generate a variant of yourself that can potentially perform better.
#     Parent algorithm description: {algorithm}
#     Parent code:
#     {code}
#
#     Instructions:
#     - Make a small but meaningful modification.
#     - Keep the function name and interface unchanged:
#       Function name: {self.prompt_func_name}
#       Inputs: {self.joined_inputs}
#       Outputs: {self.joined_outputs}
#       {self.prompt_inout_inf}
#       {self.prompt_other_inf}
#
#     First, describe your new algorithm and main steps in one sentence. The description must be inside a brace {{}}.
#     Next, implement the improved version in Python.
#
#     Do not give additional explanations.
#     """
#
#         # Create individual
#         individual = {
#             "stdout_filepath": std_out_filepath,
#             "code_path": outdir + f"problem_eval{runid}_code.py",
#             "code": code,
#             "algorithm": algorithm,
#             "prompt_template": default_prompt_template,  # For Meta-ACGA
#             "response_id": response_id,
#             "generation": 0,
#             "objective": None
#         }
#         return individual


import re
import time
from .interface_LLM import InterfaceAPI as InterfaceLLM
import os
import numpy as np

input = lambda: ...


class Evolution():
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        # -----------------------------------------------------------

        # set prompt interface
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        self.prompt_knowledge = prompts.get_knowledge()
        self.prompt_seed_func = prompts.get_seed_func()


        self.instances_dir = f'{prompts.root_dir}/problems/{prompts.problem}/dataset'
        # self.data_file = [f for f in os.listdir(self.instances_dir) if f.startswith("train")][0]
        # self.data = np.load(os.path.join(self.instances_dir, self.data_file), allow_pickle=True)

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        # -------------------- RZ: use local LLM --------------------
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def response_to_individual(self, code, algorithm, response_id, file_name=None) -> dict:
        """
        Convert LLM response to individual dictionary
        """
        # Create output directory
        outdir = './evaluations/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # Generate unique ID
        runid = hash(code)

        # Set file names
        file_name = outdir + f"problem_eval{runid}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(code + '\n')

        std_out_filepath = outdir + f"problem_eval{runid}_stdout.txt" if file_name is None else file_name.rstrip(
            ".txt") + "_stdout.txt"

        # Initialize with default prompt template for Meta-ACGA
        default_prompt_template = f"""
You are a self-improving heuristic for {self.prompt_task}.
Your task is to generate a variant of yourself that can potentially perform better.
Parent algorithm description: {algorithm}
Parent code:
{code}

Instructions:
- Make a small but meaningful modification.
- Keep the function name and interface unchanged:
  Function name: {self.prompt_func_name}
  Inputs: {self.joined_inputs}
  Outputs: {self.joined_outputs}
  {self.prompt_inout_inf}
  {self.prompt_other_inf}

First, describe your new algorithm and main steps in one sentence. The description must be inside a brace {{}}.
Next, implement the improved version in Python.

Do not give additional explanations.
"""

        # Create individual
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": outdir + f"problem_eval{runid}_code.py",
            "code": code,
            "algorithm": algorithm,
            "prompt_template": default_prompt_template,  # For Meta-ACGA
            "response_id": response_id,
            "generation": 0,
            "objective": None
        }
        return individual

    def get_prompt_i1(self):
        prompt_content = (
            f"{self.prompt_task}\n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            f"{self.prompt_func_name}. This function should accept {len(self.prompt_func_inputs)} input(s): "
            f"{self.joined_inputs}. The function should return {len(self.prompt_func_outputs)} output(s): "
            f"{self.joined_outputs}. {self.prompt_inout_inf} {self.prompt_other_inf}\n"
            "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i]['code'] + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n" \
                         + prompt_indiv + \
                         "Please help me create a new algorithm that has a totally different form from the given ones. \n" \
                         "First, describe your new algorithm and main steps in one sentence. \
                         The description must be inside a brace. Next, implement it in Python as a function named \
                         " + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."

        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i]['code'] + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n" \
                         + prompt_indiv + \
                         "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n" \
                         "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
                         The description must be inside a brace. Thirdly, implement it in Python as a function named \
                         " + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m1(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \
                                          Algorithm description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please assist me in creating a new algorithm which is a mutated version of the algorithm provided and better than the current algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments. \n" \
                     "First, describe your new algorithm and main steps in one sentence. \
                 The description must be inside a brace. Next, implement it in Python as a function named \
                 " + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \
                                          Algorithm description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n" \
                     "First, describe your new algorithm and main steps in one sentence. \
                 The description must be inside a brace. Next, implement it in Python as a function named \
                 " + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def _get_alg(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        algorithm = algorithm[0]
        code = code[0]

        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        if "import" not in code_all:
            code_all = "import numpy as np\nimport random\nimport math\nimport scipy\nimport torch\n" + code_all

        return [code_all, algorithm]

    def i1(self):
        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def e1(self, parents):
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def e2(self, parents):
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def m1(self, parents):
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def m2(self, parents):
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def get_prompt_n1(self, population, k=3):
        if len(population) > k:
            sorted_pop = sorted(population,
                                key=lambda x: x.get('novelty_score', 0),
                                reverse=True)
            selected_indivs = sorted_pop[:k]
        else:
            selected_indivs = population

        prompt_indiv = ""
        for i, ind in enumerate(selected_indivs):
            prompt_indiv += f"No.{i + 1} algorithm and the corresponding code are: \n"
            prompt_indiv += ind['algorithm'] + "\n" + ind['code'] + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            f"I have {len(selected_indivs)} existing algorithms that are known to be highly novel and diverse. Their codes are as follows: \n" \
                         + prompt_indiv + \
                         "Please create a new algorithm that is fundamentally different in approach and structure from all of these existing algorithms. " \
                         "The new algorithm should explore a completely different problem-solving paradigm. \n" \
                         "First, describe your new algorithm and main steps in one sentence without special character or letter. \
                         The description must be inside a brace. Next, implement it in Python as a function named " \
                         + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" \
                                                   "Focus on maximum novelty and avoid similar approaches to the provided algorithms. " \
                                                   "Do not give additional explanations. Output code only"

        return prompt_content

    def n1(self, population):
        prompt_content = self.get_prompt_n1(population)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [n1] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def get_prompt_ls1(self, individual):
        """Neighborhood Search - Find better neighboring solution"""
        prompt_content = self.prompt_task + "\n" \
                                            "I have the following algorithm that needs local improvement:\n" \
                                            "Algorithm description: " + individual['algorithm'] + "\n" \
                                                                                                  "Code:\n" + \
                         individual['code'] + "\n\n" \
                                              "Please help me create a NEIGHBORING algorithm that makes small modifications " \
                                              "to explore the local search space.\n" \
                                              "First, identify one or two components that can be slightly modified.\n" \
                                              "Second, make small changes such as:\n" \
                                              "- Adjusting weight coefficients\n" \
                                              "- Changing exploration vs exploitation balance\n" \
                                              "- Modifying threshold values\n" \
                                              "- Slightly altering the decision logic\n" \
                                              "Third, describe your new algorithm and main steps in one sentence. " \
                                              "The description must be inside a brace. \n" \
                                              "Fourth, implement the improved version in Python as a function named " \
                         + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" \
                                                   "IMPORTANT: Make only small, focused changes to explore local neighborhood.\n" \
                                                   "Do not give additional explanations. "

        return prompt_content

    def get_prompt_ls2(self, individual, performance_feedback=""):
        """Hill Climbing - Improve via gradient"""
        feedback_section = ""
        if performance_feedback:
            feedback_section = "Performance feedback: " + performance_feedback + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have the following algorithm that needs improvement through hill climbing:\n" \
                                            "Algorithm description: " + individual['algorithm'] + "\n" \
                                                                                                  "Code:\n" + \
                         individual['code'] + "\n\n" \
                         + feedback_section + \
                         "Please help me perform one step of hill climbing local search:\n" \
                         "Analyze the current algorithm's weaknesses or suboptimal components.\n" \
                         "Make a single improvement that addresses the main weakness.\n" \
                         "Describe your improved algorithm and main steps in one sentence. " \
                         "The description must be inside a brace. \n" \
                         "Then, implement the improved version in Python as a function named " \
                         + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" \
                                                   "Focus on making ONE significant improvement rather than many small changes.\n" \
                                                   "Do not give additional explanations."

        return prompt_content

    def get_prompt_ls3(self, individual, tabu_list=[]):
        tabu_section = ""
        if tabu_list:
            tabu_section = "Recently tried modifications (avoid these):\n"
            for i, tabu_item in enumerate(tabu_list[-3:]):  # Last 3 tabu items
                tabu_section += f"{i + 1}. {tabu_item}\n"
            tabu_section += "\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have the following algorithm that needs diversification through tabu search:\n" \
                                            "Algorithm description: " + individual['algorithm'] + "\n" \
                                                                                                  "Code:\n" + \
                         individual['code'] + "\n\n" \
                         + tabu_section + \
                         "Please help me create a diversified version that explores new areas:\n" \
                         "First, identify what has been recently tried (tabu) and avoid similar approaches.\n" \
                         "Second, make changes that are significantly different from previous attempts.\n" \
                         "Third, describe your new algorithm and main steps in one sentence. " \
                         "The description must be inside a brace. \n" \
                         "Fourth, implement the diversified version in Python as a function named " \
                         + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" \
                                                   "Focus on DIVERSIFICATION and AVOIDING recently tried modifications.\n" \
                                                   "Do not give additional explanations. "

        return prompt_content

    def ls1(self, individual):
        """Neighborhood Search Local Search"""
        prompt_content = self.get_prompt_ls1(individual)

        if self.debug_mode:
            print("\n >>> check prompt for local search [ ls1 - neighborhood ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def ls2(self, individual, performance_feedback=""):
        """Hill Climbing Local Search"""
        prompt_content = self.get_prompt_ls2(individual, performance_feedback)

        if self.debug_mode:
            print("\n >>> check prompt for local search [ ls2 - hill climbing ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def ls3(self, individual, tabu_list=[]):
        """Tabu Search Local Search"""
        prompt_content = self.get_prompt_ls3(individual, tabu_list)

        if self.debug_mode:
            print("\n >>> check prompt for local search [ ls3 - tabu search ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def get_prompt_n2(self, population, k=3):
        """
        Generate prompt for Surprise Search - create surprising heuristic
        """
        if len(population) > k:
            sorted_pop = sorted(population,
                                key=lambda x: x.get('novelty_score', 0),
                                reverse=True)
            selected_indivs = sorted_pop[:k]
        else:
            selected_indivs = population

        prompt_indiv = ""
        for i, ind in enumerate(selected_indivs):
            prompt_indiv += f"No.{i + 1} algorithm (highly novel) and the corresponding code are: \n"
            prompt_indiv += ind['algorithm'] + "\n" + ind['code'] + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            f"I have {len(selected_indivs)} existing algorithms that are known to be highly novel and diverse. Their codes are as follows: \n" \
                         + prompt_indiv + \
                         "However, I want you to create a COMPLETELY SURPRISING algorithm that is fundamentally different in approach and structure from ALL of these existing algorithms. " \
                         "The new algorithm should explore a TOTALLY UNEXPECTED problem-solving paradigm that no one would anticipate. \n" \
                         "First, describe your surprising algorithm and main steps in one sentence but do NOT use special characters here." \
                         "The description must be inside a brace {}. Next, implement it in Python as a function named" \
                         + self.prompt_func_name + ". This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" \
                                                   "IMPORTANT: Focus on MAXIMUM SURPRISE - create something that would be considered 'out of the box' thinking. " \
                                                   "Avoid any approaches similar to the provided algorithms. Be creative and unexpected!\n" \
                                                   "Do not give additional explanations."

        return prompt_content

    def n2(self, population):
        prompt_content = self.get_prompt_n2(population)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ surprise_search n2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def get_completion_tokens(self):
        return self.interface_llm.get_completion_tokens()

    def get_prompt_c1(self, parents):
        """
        Prompt for crossover: combine two parent algorithms
        """
        prompt_content = self.prompt_task + "\n" \
                                            """
                                            I have two high-performing algorithms that solve this problem effectively:

                                            Parent Algorithm 1:
                                            Description: {parent1_algo}
                                            Code:
                                            {parent1_code}

                                            Parent Algorithm 2:
                                            Description: {parent2_algo}
                                            Code:
                                            {parent2_code}

                                            Please create a new algorithm by COMBINING elements between these two parents.

                                            The new algorithm should be:
                                            - Functionally complete and coherent
                                            - Superior to or at least competitive with both parents
                                            - Not just a simple mixture but a thoughtful integration

                                            First, describe your hybrid algorithm and main steps in one sentence. 
                                            The description must be inside a brace.
                                            Second, implement it in Python as a function named {func_name}.
                                            This function should accept {num_inputs} input(s): {inputs}.
                                            The function should return {num_outputs} output(s): {outputs}.
                                            {inout_inf} {other_inf}

                                            Do not give additional explanations. Output code only
                                            """.format(
            parent1_algo=parents[0]['algorithm'],
            parent1_code=parents[0]['code'],
            parent2_algo=parents[1]['algorithm'],
            parent2_code=parents[1]['code'],
            func_name=self.prompt_func_name,
            num_inputs=len(self.prompt_func_inputs),
            inputs=self.joined_inputs,
            num_outputs=len(self.prompt_func_outputs),
            outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf
        )

        return prompt_content

    def c1(self, parents):
        prompt_content = self.get_prompt_c1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for crossover [ c1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def get_prompt_m3(self, indiv1):
        prompt_content = "First, you need to identify the main components in the function below. \
    Next, analyze whether any of these components can be overfit to the in-distribution instances. \
    Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
    Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n" + indiv1[
            'code'] + "\n" \
                         + self.prompt_inout_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def m3(self, parents):
        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)

    def get_prompt_to_evolve_prompt(self, parent_prompt, reflection):
        """
        Generate prompt to evolve the prompt template
        """
        return f"""
You are an expert in prompt engineering for LLM-driven algorithm design.

Current prompt template:
{parent_prompt}

Performance feedback: {reflection}

Reflection:

Your task: Refine and evolve this prompt template to:
- Generate heuristics that improve objective performance across diverse instances
- Explicitly discourage trivial, repetitive, or degenerate solutions (e.g., infinite loops, hard-coded constants, or purely random behavior)
- Encourage exploration of novel yet practically implementable strategies
- Maintain balance between innovation and robustness, ensuring outputs generalize beyond training instances

Requirements:
- Keep the same output format (code with description in braces)
- Do not add external libraries or change function signature
- Focus on improving the instruction clarity and strategic guidance

Return only the evolved prompt template. Do not explain. Do not use sepcial character.
"""

    def self_construct_with_prompt_evolution(self, parent, pop):
        """
        Auto-Constructive with Prompt Evolution:
        1. Evolve the prompt template
        2. Use the evolved prompt to generate offspring code
        """
        # Step 1: Evolve prompt
        reflection = self.reflection(pop)
        evolve_prompt_content = self.get_prompt_to_evolve_prompt(
            parent_prompt=['prompt_template'],
            reflection=reflection
        )

        # Call LLM to generate new prompt
        new_prompt_template = self.interface_llm.get_response(evolve_prompt_content)

        # Step 2: Use evolved prompt to generate offspring code
        offspring_code_prompt = new_prompt_template + "\n\n" + f"""
Parent algorithm description: {parent['algorithm']}
Parent code:
{parent['code']}

Generate a variant of yourself.
"""

        # Call LLM to generate code
        [code_all, algorithm] = self._get_alg(offspring_code_prompt)

        # Create offspring
        offspring = {
            "code": code_all,
            "algorithm": algorithm,
            "prompt_template": new_prompt_template,  # ← EVOLVED PROMPT
            "generation": parent.get("generation", 0) + 1,
            "objective": None,
            "response_id": parent.get("response_id", 0),
            "stdout_filepath": parent.get("stdout_filepath", ""),
            "code_path": parent.get("code_path", "")
        }

        return offspring

    def get_prompt_acga(self, parent):
        """
        Generate prompt for Auto-Constructive Genetic Algorithm
        """
        return f"""
You are a self-replicating heuristic for {self.prompt_task}.
Your task is to generate a child heuristic that inherits and improves upon your capabilities.

Parent algorithm description: {parent['algorithm']}
Parent code:
{parent['code']}

Instructions:
- Make at least one focused improvement which could make the child heuristic outperform its parent.
- Preserve the core strategy while enhancing robustness or efficiency
- Keep function signature unchanged: {self.prompt_func_name}({self.joined_inputs}) -> {self.joined_outputs}

First, describe the child algorithm in one sentence inside braces {{}}.
Then, implement the child heuristic in Python.

Do not give additional explanations. Do not use special character.
"""

    def acga(self, parents):
        """
        Auto-Constructive Genetic Algorithm: each individual generates its own offspring
        """
        if not isinstance(parents, list):
            parents = [parents]

        parent = parents[0]  # Use first parent
        prompt_content = self.get_prompt_acga(parent)

        if self.debug_mode:
            print("\n >>> check prompt for [ acga ] : \n", prompt_content)
            input(">>> Press 'Enter' to continue")

        [code_all, algorithm] = self._get_alg(prompt_content)

        # Create offspring with same prompt template (will evolve later)
        offspring = {
            "code": code_all,
            "algorithm": algorithm,
            "prompt_template": parent.get('prompt_template', ""),  # Inherit prompt
            "generation": parent.get("generation", 0) + 1,
            "objective": None,
            "response_id": parent.get("response_id", 0),
            "stdout_filepath": parent.get("stdout_filepath", ""),
            "code_path": parent.get("code_path", "")
        }

        if self.debug_mode:
            print("\n >>> ACGA Offspring Algorithm: \n", algorithm)
            print("\n >>> ACGA Offspring Code: \n", code_all)
            input(">>> Press 'Enter' to continue")

        return offspring

    def get_prompt_reflection(self, pop):
        # Sắp xếp quần thể theo objective (tốt nhất đến tệ nhất)
        sorted_pop = sorted(pop, key=lambda x: x.get('objective', float('inf')))

        # Tạo danh sách heuristic theo thứ tự
        lst_method = ""
        for i, ind in enumerate(sorted_pop):
            lst_method += f"**[{i + 1}]** Objective: {ind['objective']:.2f}\n"
            lst_method += f"```python\n{ind['code']}\n```\n\n"

        prompt = f'''
You are an expert in the domain of optimization heuristics. Your task is to provide useful advice based on analysis to design better heuristics.

### List heuristics
Below is a list of design heuristics ranked from best to worst.
{lst_method}

### Guide
- Keep in mind, list of design heuristics ranked from best to worst. Meaning the first function in the list is the best and the last function in the list is the worst.
- The response in Markdown style and nothing else has the following structure:
"**Analysis:**  
**Experience:**"  
In there:
+ Meticulously analyze comments, docstrings and source code of several pairs (Better code - Worse code) in List heuristics to fill values for **Analysis:**.  
Example: "Comparing (best) vs (worst), we see ...; (second best) vs (second worst) ...; Comparing (1st) vs (2nd), we see ...; (3rd) vs (4th) ...; Comparing (second worst) vs (worst), we see ...; Overall:"

+ Self-reflect to extract useful experience for design better heuristics and fill to **Experience:** (<60 words).

I'm going to tip $999K for a better heuristics! Let's think step by step.
    '''
        return prompt

    def reflection(self, pop):
        prompt = self.get_prompt_reflection(pop)

        if self.debug_mode:
            print("\n >>> [REFLECTION] Prompt for self-reflection: \n", prompt)
            print(">>> Press 'Enter' to continue")
            input()

        try:
            response = self.interface_llm.get_response(prompt)
        except Exception as e:
            print(f"Error in reflection: {e}")
            return "Error in analysis", "Error in experience"

        if self.debug_mode:
            print("\n >>> [REFLECTION] LLM Response: \n", response)
            print(">>> Press 'Enter' to continue")
            input()



        return response

    def get_prompts_find_gems(self, population):
        """
        Analyze heuristics to uncover hidden valuable components.
        For each heuristic, extract at most one high-value component.
        Returns full natural language response (no parsing).
        """
        # Sort from best to worst (assuming lower objective = better)
        sorted_pop = sorted(population, key=lambda x: x.get('objective', float('inf')), reverse=False)

        lst_method = ""
        for i, ind in enumerate(sorted_pop):
            obj = ind.get('objective', 'unknown')
            code_snippet = ind['code'].strip()
            lst_method += f"**[{i + 1}]** Objective: {obj:.3f}\n"
            lst_method += f"```python\n{code_snippet}\n```\n\n"

        prompt = f'''
You are an expert in the domain of optimization heuristics. Your task is to provide insightful advice by analyzing both successful and failed designs.

### List of Heuristics
Below is a list of algorithms ranked from best (top) to worst (bottom):
{lst_method}

### Instructions
Analyze the following heuristics — including poorly performing ones — to uncover hidden gems:
- What smart mechanism is buried inside?
- When does it work well?
- Could it be reused in a different context?

For each heuristic, extract at most one high-value component (e.g., adaptive noise, early stopping, restart logic, gradient monitoring).

Each individual heuristic analysis must be **under 50 words** — be precise and insightful. 

Be insightful, meticulous, and reward-focused.

The response must be in Markdown style and contain only:
"**Analysis:**  
 

In there:
+ Focus on **one high-value component per heuristic** — prioritize novelty, adaptiveness, robustness.
+ Identify how weak heuristics might contain reusable mechanisms despite poor overall performance.

I'm going to tip $999K for a better heuristic! Let's think step by step.
'''
        return prompt


    def find_gems(self, pop):
        prompt = self.get_prompts_find_gems(pop)

        if self.debug_mode:
            print("\n >>> [REFLECTION] Prompt for self-reflection: \n", prompt)
            print(">>> Press 'Enter' to continue")
            input()

        try:
            response = self.interface_llm.get_response(prompt)
        except Exception as e:
            print(f"Error in reflection: {e}")
            return "Error in analysis", "Error in experience"

        if self.debug_mode:
            print("\n >>> [REFLECTION] LLM Response: \n", response)
            print(">>> Press 'Enter' to continue")
            input()

        return response

    def get_prompt_synthesizer_combine_opposites(self, analysis):
        """
        Takes full analysis (from analyst) and synthesizes a new adaptive principle.
        Follows same expert format: structured Markdown + $999K motivation.
        """
        prompt = f'''
You are an expert in the domain of optimization heuristics. Your task is to create a novel, high-performance heuristic by creatively combining insights from existing ones.

### Analysis Input
Below is a detailed comparison of current heuristics, including valuable mechanisms hidden in underperforming ones:
{analysis}

### Guide
- Focus on conflict and complementarity: identify opposing strategies (e.g., pure exploitation vs heavy exploration).
- Design a new adaptive principle that resolves the tension intelligently.
- Use runtime signals (e.g., stagnation, gradient variance, iteration count) to switch or blend behaviors.
- The response must be in Markdown style and nothing else, with this structure:
"**Insight:**  
**Adaptive Principle:**"  

In there:
+ Explain how you combine strengths and neutralize weaknesses.
+ Example: "The best uses gradient descent but gets stuck; the worst explores randomly but wastes time. A hybrid could use gradient normally, but trigger random jumps when improvement stalls."
+ Propose one clear, implementable adaptive rule.

I'm going to tip $999K for a better heuristic! Let's think step by step.
'''
        return prompt

    def synthesizer_combine_opposites(self, pop):
        prompt_find_gems = self.get_prompts_find_gems(pop)
        analysis = self.interface_llm.get_response(prompt_find_gems)
        prompt = self.get_prompt_synthesizer_combine_opposites(analysis)

        if self.debug_mode:
            print("\n >>> [REFLECTION] Prompt for self-reflection: \n", prompt)
            print(">>> Press 'Enter' to continue")
            input()

        try:
            response = self.interface_llm.get_response(prompt)
        except Exception as e:
            print(f"Error in reflection: {e}")
            return "Error in analysis", "Error in experience"

        if self.debug_mode:
            print("\n >>> [REFLECTION] LLM Response: \n", response)
            print(">>> Press 'Enter' to continue")
            input()

        return response

    def get_prompt_engineer_implement(self, pop):

        synthesis = self.synthesizer_combine_opposites(pop)

        prompt = f'''
You are an expert in combinatorial optimization.

{self.prompt_task}

The new function must adopt the adaptive principles:
{synthesis}

First, describe your implementation and main adaptation logic in one sentence. The description must be inside braces {{}}.
Next, implement it in Python as a function named {self.prompt_func_name}. This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. {self.prompt_inout_inf} {self.prompt_other_inf}.

Do not give additional explanations. Do not use special character.
'''
        return prompt

    def recombination(self, pop):

        prompt_content = self.get_prompt_engineer_implement(pop)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ recombination ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)
        return self.response_to_individual(code_all, algorithm, 0)


