# Rigid-origami
A **gym environment and commandline tool** for automating rigid origami crease pattern design. 

> **Note** We provide an introduction to the underlying principles of rigid origami and some practical use cases in our paper [Automating Rigid Origami Design](https://arxiv.org/abs/2211.13219).

We reformulate the rigid-origami problem as a board game, where an agent interacts with the rigid-origami **gym environment** according to a set of rules which define an origami-specific problem. 

<figure>
    <img src="/assets/TeaserFigure.png"
         alt="The rigid origami game"
         height="200"
    />
</figure>

*Figure: The rigid origami game for shape-approximation. The target shape (a) approximated by an agent playing the game (b) to find a pattern (c) which in its folded state (d) approximates the target.*

Our commandline tool comprises a set of agents (or classical search methods and algorithms) for 3D shape approximation, packaging, foldable funiture and more. 

> **Note** The rigid-origami environment is not limited to these particular origami design challenges and agents. You can also deploy it on your own custom use case.


## Installing
> **Note** Before installing the dependencies, it is recommended that you setup a [conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Setup and activate the conda environment using the [conda_rigid_origami_env.yml](conda_rigid_origami_env.yml) package list.

```
$ conda create --name=rigid-origami --file=conda_rigid_ori_env.yml
$ conda activate rigid-origami
```

Next install the gym-environment.

```
(rigid-origami) $ cd gym-rori
(rigid-origami) $ pip install -e .
```

The environment is now all set up.

## Example
We *play* the rigid-origami game for shape-approximation. 

```
(rigid-origami) $ python main.py --objective=shape-approx --search-algorithm=RDM --num-steps=100000 --board-length=25 --num-symmetries=2 --optimize-psi  
```

Adjust the game objective, agent, or any other conditions by setting specific options.

> **Note** You can utilize the environment for different design tasks or objectives. You can also add a custom reward function in the [rewarder](gym-rori/rewarders.py).

A non-exhaustive list of the basic game settings is given below.

|  Option                       | Flag                  | Value                                                 |   Default value   |
| -------------                 |-------------:         | :-----                                                |   :---:           |
| **Name**                      | --name                | *string*                                              |   "0"             |
| **Number of env interactions**| --num-steps           | *int*                                                 |   500             |
| **Game objective**            | --objective           | {shape-approx, packaging, chair, table, shelf, bucket}|   shape-approx    |
| **Agent**                     | --search-algorithm    | {RDM, MCTS, evolution, PPO, DFTS, BFTS, human}        |   RDM             |
| **Seed pattern**              | --base                | {plain, simple, simple-vert, single, quad}            |   plain           |
| **Seed sequence**             | --start-sequence      | *list*                                                |   []              |
| **Seed pattern size**         | --seed-pattern-size   | *int*                                                 |   2               |
| **Board edge length**         | --board-length        | *int*                                                 |   13              |
| **Number of symmetry axes**   | --num-symmetries      | {0,1,2,3}                                             |   2               |
| **Maximum number of vertices**| --max-vertices        | *int*                                                 |   100             |
| **(Max.) fold angle**         | --psi                 | *float*                                               |   3.14            |
| **Auto optimize fold angle**  | --optimize-psi        |                                                       |   True            |
| **Maximum crease length**     | --cl-max              | *float*                                               |   infinity        |
| **Random seed**               | --seed                | *int*                                                 |   16711           |
| **Allow source action**       | --allow-source-action | *string*                                              |   False           |
| **Target**                    | --target              | *string*                                              |   "target.obj"    | 
| **Target transform**          | --target-transform    | *transform*                                           |   [0,0,0]         | 
| **Auto target mesh transform**| --auto-mesh-transform |                                                       |   False           |
| **Count interior vertices**   | --count-interior      |                                                       |   False           |
| **Mode**                      | --mode                | {TRAIN, DEBUG}                                        |   TRAIN           |
| **Resume run**                | --resume              |                                                       |   False           |
| **Local directory**           | --local-dir           | *string*                                              |   "cwd"           |
| **Branching factor**          | --bf                  | *int*                                                 |   10              |
| **Number of workers (RL)**    | --num-workers         | *int*                                                 |   0               |
| **Training iterations RL)**   | --training-iteration  | *int*                                                 |   100             |
| **Number of CPUs (RL)**       | --ray-num-cpus        | *int*                                                 |   1               |
| **Number of GPUs (RL)**       | --ray-num-gpus        | *int*                                                 |   0               |
| **Animation view**            | --anim-view           | *list*                                                |   [90, -90, 23]   |


> **Note** The action- and configuration space complexity grows exponentially with the board size. On the contrary additional symmetries help reduce the complexity.

## Components

### Environment
The gym environment class [RoriEnv](gym-rori/gym_rori/envs/rori_env.py) contains the methods for the agents to interact with.

In essence agents construct graphs of connected [single vertices](gym-rori/single_vertex.py) in the environment, from which they receive sparse rewards in return.

Rewards depend on the set objective and [rewarder](gym-rori/rewarders.py). 

> **Note** You can add deploy your custom [rewarder](gym-rori/rewarders.py). To make it run you also need to add and call your custom objective from the [main](main.py). 

A game terminates if a terminal state is reached, either by choice of the terminal action of the agent or by violation of a posteriori foldability [conditions](#rules).

### Agents
Agents interact with the environment. They can be human or artificial. We provide a list of standard search algorithms as artificial agents.

|       Agent               |   Search Algorithm                                    |
|   :-----------            |   :---------------                                    |
| [RDM](main.py)            | Uniform Random Search                                 |
| [BFTS](gym-rori/bfts.py)  | Breadth First Tree Search                             |
| [DFTS](gym-rori/dfts.py)  | Depth First Tree Search                               |
| [MCTS](gym-rori/mcts.py)  | Monte Carlo Tree Search                               |
| [evolution](main.py)      | Evolutionary Search                                   |
| [PPO](main.py)            | Proximal Policy Optimization (Reinforcement Learning) |

> **Note** You can add you own custom agent or search-algorithm in the [main](main.py).

### Rules
<a href="#rules"></a>
[Rules](gym-rori/rules.py) and [symmetry-rules](gym-rori/symmetry_rules.py) are enforced a priori through action masking, constraining the action space of agents.

A game can however reach a non-foldable state. A particular state is foldable if and only if it complies with the following conditions:

1. Faces do not intersect during folding as proven by a [triangle-triangle intersection test](gym-rori/tritri_intsec_check.py).
2. The corresponding Origami Pattern is rigidly foldable as validated by the [kinematic model](gym-rori/kinematic_model_num.py).

Any violation of the two conditions results in a terminal game state.

### Results
The best episodes (origami patterns) are documented in the results directory per experiment. For each best episode three files, a *PNG* of the corresponding origami graph, folded shape *OBJ* and animation *GIF* are stored.

|       Origami Graph (Pattern)                     |   Folded Shape                                    |   Folding Animation
|   :-----------:                                   |   :---------------:                               |   :---------------: 
| <img src="assets/chair_pattern.png" width="500"/> | <img src="assets/chair_folded.png" width="500"/>  | <img src="assets/animations/chair.gif" width="500"/>




