import argparse
import random
import itertools

from collections import defaultdict
import gymnasium as gym
import gym_PBN
import torch
from gym_PBN.envs.bittner.base import findAttractors

import numpy as np
from sympy import symbols
from sympy.logic import SOPform

import math

from gattaca_model import GATTACA

import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl

from gym_PBN.utils.get_attractors_from_cabean import get_attractors

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, required=True)
parser.add_argument('--model-path', required=True)
parser.add_argument('--attractors', type=int, default=3)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--assa-file', type=str, default=None)
parser.add_argument('--matlab-file', type=str, default=None)


args = parser.parse_args()

# model_cls = GraphClassifier
# model_name = "GraphClassifier"


N = args.n
model_path = args.model_path
# min_attractors = args.attractors

# load ispl model
if args.assa_file is not None and args.matlab_file is None:
    from gattaca_model.utils import AgentConfig
    model_cls = GATTACA
    model_name = "GATTACA"
    with open(args.assa_file, "r") as env_file:
        genes = []
        logic_funcs = defaultdict(list)

        for line in env_file:
            line = line.split()

            if len(line) == 0:
                continue

            # get all vars
            if line[0] == "Vars:":
                while True:
                    line = next(env_file)
                    line = line.split()

                    if line[0] == "end":
                        break

                    if line[0][-1] == ":":
                        genes.append(line[0][:-1])
                    else:
                        genes.append(line[0])

            if line[0] == "Evolution:":
                while True:
                    line = next(env_file)
                    line = line.split()

                    if len(line) == 0:
                        continue

                    if line[0] == "end":
                        break

                    target_gene = line[0].split("=")[0]
                    if line[0].split("=")[1] == "false":
                        continue

                    for i in range(len(line)):
                        sline = line[i].split("=")
                        if sline[-1] == "false":
                            line[i] = f"( not {sline[0]} )"
                        else:
                            line[i] = sline[0]

                    target_fun = " ".join(line[2:])
                    target_fun = target_fun.replace("(", " ( ")
                    target_fun = target_fun.replace(")", " ) ")
                    target_fun = target_fun.replace("|", " or ")
                    target_fun = target_fun.replace("&", " and ")
                    target_fun = target_fun.replace("~", " not ")
                    logic_funcs[target_gene].append((target_fun, 1.0))

    print(list(logic_funcs.keys()))
    print(list(logic_funcs.values()))

    for i in range(len(genes)):
        print(list(logic_funcs.keys())[i], list(logic_funcs.values())[i])

    # Load env
    env = gym.make(f"gym-PBN/PBNEnv",
                   N=args.n,
                   genes=list(logic_funcs.keys()),
                   logic_functions=list(logic_funcs.values()),
                   min_attractors=args.attractors)


print(type(env.env.env))
env.reset()

config = AgentConfig()
model = model_cls(N, N + 1, config, env)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.EPSILON = 0

state, _ = env.reset()

action = model.predict(state, state)
state, *_ = env.step(action)

all_attractors = env.all_attractors

lens = []
failed = 0
total = 0

failed_pairs = []

all_attractors = env.divided_attractors
print("genereted attractors:")
for a in all_attractors:
    print(a)

# list: (state, target) -> list of lens
lens = [[] for _ in range(args.attractors * args.attractors)]
failed = 0
total = 0

failed_pairs = []

result_matrix = np.zeros((args.attractors))
data = defaultdict(int)

runs = args.runs
gene_stats = []
genes_used = defaultdict(int)
total = 0
count = 0
print("testing on ", len(all_attractors), " attractros")
for i in range(runs):
    print("testing round ", i)
    id = -1

    for attractor_id in range(args.attractors):
        gens_used_tmp = defaultdict(int)
        gu = []

        # print(f"processing initial_state, target_state = {attractor_id}, {target_id}")
        model.EPSILON = 0.
        id += 1
        attractor = all_attractors[attractor_id]
        # target = all_attractors[target_id]
        # target_state = target[0]
        initial_state = attractor
        total += 1
        actions = []
        state = initial_state
        state = [0 if i == '*' else i for i in list(state)]
        _ = env.reset()
        env.graph.setState(state)
        total += count
        count = 0

        # env.setTarget(target)

        while not env.in_target(state):
            gu_tmp = []
            count += 1

            # policy, value = model.predict(state, target_state)
            # policy = policy.numpy()
            # action = [np.random.choice(range(N+1), p=policy)]
            action = model.predict(state, state)
            al = action.tolist()

            for gen in al:
                gens_used_tmp[gen-1] += 1
                gu_tmp.append(gen-1)

            gu.append(gu_tmp)

            _ = env.step(action)
            state = env.render()
            # action_named = [gen_ids[a-1] for a in action]

            if count > 100:
                print(f"failed to converge for {attractor_id}")
                # print(f"final state was 		     {tuple(state)}")
                print(id)
                failed += 1
                # raise ValueError
                break
        else:
            print(f"for initial state {attractor_id} got (total of {count} steps), on: \n"
                  f"{[[env.graph.nodes[a].name for a in action] for action in gu]}")


print('got total steps ', total, 'in ', args.attractors * args.runs, 'runs')
print('average ', total / (args.attractors * args.runs))
gene_stats = gene_stats[1:]
print("gene used: ", {env.graph.nodes[i].name: genes_used[i] for i in genes_used if i not in env.forbidden_actions})
