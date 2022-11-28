def build_pattern_from_decision_chain(env,action_list_o,board_size):
    for action in action_list_o:
        env.step(action)