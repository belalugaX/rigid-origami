import numpy as np
from copy import deepcopy
from env_config import place_source, create_base

def build_pattern_from_decision_chain(env,action_list_o,board_size):
    for action in action_list_o:
        env.step(action)

    # if env.symmetry_axis == "point":
    #     board_size = env.half_board_length+1
    #     board_size2 = board_size
    # elif env.symmetry_axis == "y-axis":
    #     board_size2 = env.half_board_length+1
    # else:
    #     board_size2 = board_size
    # action_list = deepcopy(action_list_o)
    # action_list.reverse()
    # while action_list and not env.done:
    #     action = action_list.pop()
    #     if isinstance(action,tuple):
    #         act = action[0]
    #         row,col = env.vertex_objs[act].board_pos
    #         act = row+(col*board_size)
    #         rbm = action[1]
    #         act = int(act+max(-rbm,0)*board_size*board_size2)
    #         env.step(act)
    #     elif isinstance(action,list):
    #         pseudo_vtx, c, rbm = action
    #     #        rbm = 1 if rbm>=0 else -1
    #         if pseudo_vtx == None:
    #             create_base(env,c,rbm)
    #         else:
    #             # place_source(self, a, b, c, dir=rbm)
    #             place_source(env, pseudo_vtx, rbm)
    #         # a, b, coords, rbm = action
    #         # if action[0] is None:
    #         #     create_base(env,coords,rbm)
    #         # else:
    #         #     place_source(env,a,b,coords,dir=rbm)
    #     elif action < board_size*board_size2:
    #         env.step(action)
