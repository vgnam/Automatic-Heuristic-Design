import hydra
import logging
import os
from pathlib import Path
import subprocess

import sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)

import sys
sys.stdout.reconfigure(encoding='utf-8')

os.environ["HYDRA_FULL_ERROR"] = "1"

os.environ["MISTRAL_API_KEY"] = "kG8lsMi5PW0Y7alZkY0TBVJlvnjm2jM9"
os.environ["GEMINI_API_KEY"] = "AIzaSyD6BKmewlxhOqdBIerixihw0DJ3iYDIj9g"

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    print(workspace_dir)
    # # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.model}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    if cfg.algorithm == "hsevo":
        from hsevo import HSEvo as LHH

    elif cfg.algorithm == "reevo":
        from baselines.reevo import ReEvo as LHH
    elif cfg.algorithm == "reevo-island":
        from baselines.reevo import ReEvo_Island as LHH
    elif cfg.algorithm == "reevo-hs":
        from variants.reevo import ReEvoHS as LHH
    elif cfg.algorithm == "reevo-rf":
        from variants.reevo import ReEvoRF as LHH
    elif cfg.algorithm == "eoh-island":
        from baselines.eoh import EoH_Island as LHH
    elif cfg.algorithm == "eoh":
        from baselines.eoh import EoH as LHH
    elif cfg.algorithm == "eoh-novelty":
        from baselines.eoh import EoH_Novelty as LHH
    else:
        raise NotImplementedError


    # Main algorithm
    lhh = LHH(cfg, ROOT_DIR)
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")

    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w') as file:
        file.writelines(best_code_overall + '\n')
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {test_script}")
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
    logging.info(f"Validation script finished. Results are saved in {test_script_stdout}.")

    # Print the results
    with open(test_script_stdout, 'r') as file:
        for line in file.readlines():
            logging.info(line.strip())

if __name__ == "__main__":
    main()
