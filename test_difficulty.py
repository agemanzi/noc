#!/usr/bin/env python3
"""
Test script to demonstrate the new challenging thermal game features.
Shows how different difficulty presets affect building thermal behavior.
"""

from thermal_game.engine.simulation import SimulationEngine
from thermal_game.engine.state import GameState, Action
from thermal_game.engine.settings import GameSettings
from thermal_game.engine.reward import RewardConfig
import numpy as np

def test_difficulty_preset(mode: str = "normal", test_steps: int = 96):
    """Test a difficulty preset for thermal drift behavior"""
    print(f"\n{'='*50}")
    print(f"Testing {mode.upper()} difficulty preset")
    print(f"{'='*50}")
    
    # Create simulation with chosen difficulty
    sim = SimulationEngine()
    sim.apply_difficulty_preset(mode)
    
    # Print key parameters for comparison
    print(f"Building parameters:")
    print(f"  Ci (air thermal mass): {sim.Ci_kwh_per_K:.1f} kWh/K")
    print(f"  Ce (envelope mass): {sim.Ce_kwh_per_K:.1f} kWh/K")
    print(f"  Ria (infiltration resistance): {sim.Ria_degC_per_kW:.1f} °C/kW")
    print(f"  Rea (envelope resistance): {sim.Rea_degC_per_kW:.1f} °C/kW")
    print(f"  Bridge factor: {sim.bridge_factor:.2f}")
    print(f"  HVAC ramp limit: {sim.hvac_ramp_kw_per_step:.1f} kW/step")
    print(f"  Defrost COP penalty: {sim.hvac_defrost_cop_factor:.1f}x")
    
    # Calculate time constant (rough estimate)
    G_total = 1.0/sim.Ria_degC_per_kW + 1.0/(sim.Rea_degC_per_kW/sim.bridge_factor)
    tau_hours = sim.Ci_kwh_per_K / G_total
    print(f"  Estimated time constant: {tau_hours:.1f} hours")
    
    # Test idle drift scenario
    print(f"\nTesting idle drift (no HVAC, 10°C temperature difference):")
    
    # Initial state
    state = GameState(
        t=0.0,
        T_inside=22.0,     # comfortable indoor temp
        T_outside=12.0,    # cold outdoor temp (10°C difference)
        soc=0.5,
        kwh_used=0.0,
        cumulative_reward=0.0,
        cumulative_financial=0.0,
        cumulative_comfort=0.0,
        occupied=1,
        ts=None,
    )
    
    # Add envelope temperature if not present
    if not hasattr(state, 'T_envelope'):
        state.T_envelope = 22.0
    
    settings = GameSettings(hvac_size_kw=2.6)  # challenging sizing
    reward_cfg = RewardConfig()  # use new tighter comfort bands
    
    action = Action(hvac=0.0, battery=0)  # no HVAC action (idle)
    
    temps = []
    penalties = []
    
    for step in range(test_steps):
        inputs = {
            "settings": settings,
            "reward_cfg": reward_cfg,
            "T_outside": 12.0,  # constant cold outdoor temp
            "occupied_home": 1 if step < 48 else 0,  # occupied first 12 hours
            "prev_hvac_kw": 0.0,
            "step_count": step,
        }
        
        result = sim.step(state, action, inputs)
        state = result.state
        
        temps.append(state.T_inside)
        penalties.append(result.metrics["comfort_penalty"])
        
        # Print key steps
        if step % 24 == 0 or step in [4, 8, 12, 16]:
            occupied_str = "occupied" if inputs["occupied_home"] else "unoccupied"
            print(f"  Step {step:2d} ({step*15:3d}min): T_in={state.T_inside:.2f}°C, "
                  f"penalty={result.metrics['comfort_penalty']:.3f}, {occupied_str}")
    
    # Summary statistics
    final_temp = temps[-1]
    temp_drop = 22.0 - final_temp
    max_penalty = max(penalties)
    total_penalty = sum(penalties)
    
    print(f"\nSummary after {test_steps} steps ({test_steps*15} minutes):")
    print(f"  Final temperature: {final_temp:.2f}°C")
    print(f"  Total temperature drop: {temp_drop:.2f}°C")
    print(f"  Max comfort penalty: {max_penalty:.3f}")
    print(f"  Total comfort penalty: {total_penalty:.1f}")
    
    return {
        "mode": mode,
        "tau_hours": tau_hours,
        "final_temp": final_temp,
        "temp_drop": temp_drop,
        "max_penalty": max_penalty,
        "total_penalty": total_penalty,
    }

def test_disturbances():
    """Test the new disturbance features"""
    print(f"\n{'='*50}")
    print(f"Testing DISTURBANCE FEATURES")
    print(f"{'='*50}")
    
    sim = SimulationEngine()
    sim.apply_difficulty_preset("normal")
    
    state = GameState(
        t=0.0, T_inside=22.0, T_outside=15.0, soc=0.5, kwh_used=0.0,
        cumulative_reward=0.0, cumulative_financial=0.0, cumulative_comfort=0.0,
        occupied=1, ts=None, T_envelope=22.0
    )
    
    settings = GameSettings()
    action = Action(hvac=0.0, battery=0)
    
    print("Testing disturbances over 20 steps...")
    
    for step in range(20):
        inputs = {
            "settings": settings,
            "T_outside": 15.0,
            "occupied_home": 1,
            "wind_mps": np.random.uniform(0, 5),  # random wind
            "prev_hvac_kw": 0.0,
            "step_count": step,
        }
        
        result = sim.step(state, action, inputs)
        state = result.state
        
        # Show interesting disturbances
        wind = inputs["wind_mps"]
        solar_pop = result.metrics.get("solar_pop_kw", 0)
        window_open = result.metrics.get("window_open_frac", 0)
        internal_noise = result.metrics.get("internal_noise_kw", 0)
        
        if solar_pop > 0 or window_open > 0 or abs(internal_noise) > 0.1 or wind > 3:
            print(f"  Step {step:2d}: T_in={state.T_inside:.2f}°C, "
                  f"wind={wind:.1f}m/s, solar_pop={solar_pop:.2f}kW, "
                  f"window={window_open:.2f}, noise={internal_noise:.2f}kW")
    
    print("Disturbance testing complete!")

def main():
    """Run all tests and show comparison"""
    print("THERMAL GAME DIFFICULTY TESTING")
    print("Testing how 'doing nothing' performs under different difficulty settings")
    
    # Test all three difficulty presets
    results = []
    for mode in ["easy", "normal", "hard"]:
        result = test_difficulty_preset(mode, test_steps=96)  # 24 hours
        results.append(result)
    
    # Test disturbances
    test_disturbances()
    
    # Comparison table
    print(f"\n{'='*70}")
    print(f"DIFFICULTY COMPARISON (24 hours of idle operation)")
    print(f"{'='*70}")
    print(f"{'Mode':<8} {'τ (hours)':<10} {'Temp Drop':<12} {'Final °C':<10} {'Total Penalty':<15}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['mode']:<8} {r['tau_hours']:<10.1f} {r['temp_drop']:<12.1f} "
              f"{r['final_temp']:<10.1f} {r['total_penalty']:<15.1f}")
    
    print(f"\nInterpretation:")
    print(f"• Time constant (τ): Lower = faster drift, harder to control")
    print(f"• Temperature drop: How much indoor temp falls in 24h without HVAC")  
    print(f"• Total penalty: Cumulative comfort penalty (higher = more uncomfortable)")
    print(f"• Normal/Hard modes make 'doing nothing' impossible!")

if __name__ == "__main__":
    main()