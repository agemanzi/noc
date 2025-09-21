from ..engine.state import GameState, Action

def greedy_comfort_policy(s: GameState) -> Action:
    hvac = 0.0
    if s.T_inside > 24: hvac = -0.6
    if s.T_inside < 20: hvac =  0.6
    bat = 0  # placeholder
    return Action(hvac=hvac, battery=bat)
