import os
import pandas as pd


CHECKPOINTS_DIR = "./checkpoints/"
ALGORITHMS = ["ddpg", "td3", "sac"]


def collect_runs_over_seeds(algorithm: str, checkpoints_dir: str = "./checkpoints/") -> pd.DataFrame:
    """
    Expected folder structure of checkpoints: 
   
    |-- checkpoints
    |   |-- ddpg
    |   |   |-- ant-v5-ddpg-seed0
    |   |   |   |-- actor.pt
    |   |   |   |-- critic.pt
    |   |   |   |-- evaluation.csv
    |   |   |-- ant-v5-ddpg-seed1
    |   |   |   |-- actor.pt
    |   |   |   |-- critic.pt
    |   |   |   |-- evaluation.csv
    |   |   |-- halfcheetah-v5-ddpg-seed0
    |   |   |   |-- actor.pt
    |   |   |   |-- critic.pt
    |   |   |   |-- evaluation.csv
    .   .   .   . 
    .   .   .   .
    .   .   .   .
    |   |--td3
    |   |   |-- ant-v5-ddpg-seed0
    |   |   |   |-- actor.pt
    |   |   |   |-- critic.pt
    |   |   |   |-- evaluation.csv
    .   .   .   .
    .   .   .   .
    .   .   .   . 
    """ 
    path = os.path.join(checkpoints_dir, algorithm)
    alg_dir = os.listdir(path)

    dfs_lst = []
    for run_name in alg_dir:
        run_dir_path = os.path.join(path, run_name)

        if not os.path.isdir(run_dir_path):
            continue 
        
        run_dir = os.listdir(run_dir_path)

        for file in run_dir:
            if file.endswith(".csv"):
                csv_path = os.path.join(run_dir_path, file)

                df = pd.read_csv(csv_path)
                df["algorithm"] = str(algorithm)
                df["env"] = str(run_name.split("-")[0])
                df["step"] = range(len(df))
                dfs_lst.append(df)
    
    df_combined = pd.concat(dfs_lst, ignore_index=True)

    result = df_combined.groupby(["algorithm", "env", "step"]).agg(
        mean_return=("average_rewards", "mean"),
        std_return=("std_rewards", "std")
    ).reset_index()

    return result


def all_runs_to_one_csv(algorithms: list[str], checkpoints_dir: str) -> None:
    dfs_lst = []
    for alg in algorithms:
        df = collect_runs_over_seeds(alg, checkpoints_dir)
        dfs_lst.append(df)
    result = pd.concat(dfs_lst, ignore_index=True)
    return result


def main() -> None:
    df_all_runs = all_runs_to_one_csv(ALGORITHMS, CHECKPOINTS_DIR)
    df_all_runs.to_csv("mujoco_runs.csv", index=False)


if __name__ == "__main__":
    main()
