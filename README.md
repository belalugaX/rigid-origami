# Rigid-origami
A **gym environment** and commandline tool for automating rigid origami crease pattern design generation and imagination. 

![The rigid origami game](/assets/images/method.png)

> **Note** We provide an introduction to the underlying principles of rigid origami and some practical use cases in our paper [Automating Rigid Origami Design](https://arxiv.com).

We reformulate the rigid-origami problem as a board game. 

Agents (players) interact with the rigid-origami **gym environment** (board) according to a set of rules which define an origami-specific problem. 

Our commandline tool comprises a set of agents (or classical search methods and algorithms) for 3D shape approximation, packaging, foldable funiture and more. Whereas the environment is not limited to these particular origami design challenges and agents. 


## Installing
> **Note:** Before installing the dependencies, it is recommended that you setup a [conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Setup the conda environment using the rigid-origami/conda_rigid_ori_env.yml package list:

```
$ conda create --name=rigid-origami --file=conda_rigid_ori_env.yml
```

Next install the gym-environment:

```
(rigid-origami) $ cd gym-rori
(rigid-origami) $ pip install -e .
```

The environment is now all set up.

## Example
To play the rigid-origami via commandline simply execute the following:

```
(rigid-origami) $ python main.py --objective=packaging  
```

Adjust the game objective, agent, or any other conditions by setting specific options.

A non-exhaustive list of the basic game settings is given below:

|  Option                       | Flag                | Value                                       |
| -------------                 |-------------:       | :-----                                      |
| **Name**                      | --name              | "Experiment 0"                              |
| **Game objective**            | --objective         | shape-approx, packaging, chair, table, shelf|
| **Agent**                     | --search-algorithm  | RDM, MCTS, evolution, PPO, DFTS, BFTS       |
| **Number of env interactions**| --num-steps         | 1000                                        |
| **Number of symmetry axes**   | --num-symmetries    | 1                                           |
| **Board edge length**         | --board-length      | 25                                          |
| **Seed pattern**              | --base              | simple                                      |
| **Random seed**               | --seed              | 123                                         |
| **Auto optimize fold angle**  | --optimize-psi      |                                             |

