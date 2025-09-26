#!/usr/bin/env python3
"""
Example usage of the new challenging thermal game features.
Demonstrates practical usage patterns for the improved simulation engine.
"""

from thermal_game.engine.simulation import SimulationEngine
from thermal_game.engine.state import GameState, Action
from thermal_game.engine.settings import GameSettings
from thermal_game.engine.reward import RewardConfig

def create_challenging_game_setup():
    """Create a thermal game setup that requires active management"""
    
    # Create simulation engine with normal difficulty
    sim = SimulationEngine()
    sim.apply_difficulty_preset("normal")  # or "easy", "hard"
    
    # Configure tighter comfort requirements
    reward_config = RewardConfig(
        comfort_target_C=22.0,
        comfort_tolerance_occupied_C=0.5,    # ±0.5°C when occupied
        comfort_tolerance_unoccupied_C=1.0,  # ±1.0°C when unoccupied
        comfort_weight=1.5,                   # higher penalty weight
        complaint_threshold_C=1.0,            # complaint if >1°C off target
        complaint_penalty_per_step=1.0,       # extra penalty for complaints
        complaint_duration_threshold=2,       # complaints after 2 consecutive bad steps
    )
    
    # Challenging HVAC sizing (for ~90-120 m² house)
    settings = GameSettings(
        hvac_size_kw=2.6,    # reduced capacity
        batt_size_kwh=6.0,   # limited battery
        pv_size_kw=6.0,
        comfort_tolerance_occupied_C=0.5,
        comfort_tolerance_unoccupied_C=1.0,
        comfort_weight=1.5,
    )
    
    return sim, settings, reward_config

def demonstrate_gameplay_loop():
    """Show how to run a realistic game loop with the new features"""
    
    sim, settings, reward_config = create_challenging_game_setup()
    
    # Initial state - comfortable but will drift quickly if no action
    state = GameState(
        t=0.0,
        T_inside=22.0,      # comfortable start
        T_outside=10.0,     # cold outdoor conditions
        soc=0.5,            # half battery
        kwh_used=0.0,
        cumulative_reward=0.0,
        cumulative_financial=0.0,
        cumulative_comfort=0.0,
        occupied=1,
        ts=None,
        T_envelope=21.8,    # envelope slightly cooler
    )
    
    print("Challenging Thermal Game - Gameplay Loop Example")
    print("="*60)
    print(f"Difficulty: Normal")
    print(f"HVAC size: {settings.hvac_size_kw} kW")
    print(f"Comfort band (occupied): ±{reward_config.comfort_tolerance_occupied_C}°C")
    print(f"Comfort band (unoccupied): ±{reward_config.comfort_tolerance_unoccupied_C}°C")
    print(f"Starting: T_in={state.T_inside}°C, T_out={state.T_outside}°C")
    print()
    
    # Persistent state tracking
    prev_hvac_kw = 0.0
    complaint_count = 0
    solar_pop_remaining = 0
    window_event_remaining = 0
    prev_tin_measured = state.T_inside
    
    # Simulate 12 hours (48 steps of 15 minutes each)
    for step in range(48):
        # Determine occupancy (typical pattern)
        hour_of_day = (step * 0.25) % 24
        occupied = 1 if 6 <= hour_of_day <= 23 else 0  # home 6am-11pm
        
        # Example weather patterns
        T_outside = 10.0 + 3.0 * (step % 24) / 24  # daily temperature cycle
        wind_mps = 2.0 + 3.0 * (step % 8) / 8      # wind variation
        
        # Example control strategy - simple heating when too cold
        current_temp = state.T_inside
        target_temp = 22.0
        temp_error = target_temp - current_temp
        
        if occupied and temp_error > 0.3:      # need heating when occupied
            hvac_action = min(1.0, temp_error * 2.0)  # proportional heating
        elif not occupied and temp_error > 1.5:  # emergency heating when unoccupied  
            hvac_action = 0.5
        elif temp_error < -0.3:               # need cooling
            hvac_action = max(-1.0, temp_error * 2.0)
        else:
            hvac_action = 0.0                 # no action needed
        
        action = Action(hvac=hvac_action, battery=0)  # simple HVAC control
        
        # Prepare inputs with persistent state
        inputs = {
            "settings": settings,
            "reward_cfg": reward_config,
            "T_outside": T_outside,
            "wind_mps": wind_mps,
            "occupied_home": occupied,
            "q_solar_kw": 0.3 if 10 <= hour_of_day <= 16 else 0.0,  # daytime solar gain
            "q_internal_kw": 1.0 if occupied else 0.2,              # internal gains
            "price": 0.25 + 0.15 * (step % 24) / 24,                # price variation
            # Persistent state for features
            "prev_hvac_kw": prev_hvac_kw,
            "complaint_count": complaint_count,
            "solar_pop_remaining": solar_pop_remaining,
            "window_event_remaining": window_event_remaining,
            "prev_Tin_measured": prev_tin_measured,
            "step_count": step,
        }
        
        # Step the simulation
        result = sim.step(state, action, inputs)
        state = result.state
        
        # Update persistent state for next step
        prev_hvac_kw = result.metrics["prev_hvac_kw"]
        complaint_count = result.metrics["complaint_count"]
        solar_pop_remaining = result.metrics["solar_pop_remaining"]
        window_event_remaining = result.metrics["window_event_remaining"]
        prev_tin_measured = result.metrics["Tin_measured"]
        
        # Show interesting events
        comfort_penalty = result.metrics["comfort_penalty"]
        complaint_penalty = result.metrics["complaint_penalty"]
        hvac_kw = result.metrics["hvac_kw"]
        
        show_step = (step % 12 == 0) or comfort_penalty > 0.1 or complaint_penalty > 0
        
        if show_step:
            occ_str = "occupied  " if occupied else "unoccupied"
            disturbances = ""
            if result.metrics["solar_pop_kw"] > 0:
                disturbances += f" solar_pop={result.metrics['solar_pop_kw']:.1f}kW"
            if result.metrics["window_open_frac"] > 0:
                disturbances += f" window={result.metrics['window_open_frac']:.2f}"
            
            print(f"Step {step:2d} ({hour_of_day:4.1f}h): T_in={state.T_inside:5.1f}°C "
                  f"HVAC={hvac_kw:4.1f}kW {occ_str} "
                  f"comfort_pen={comfort_penalty:5.3f} complaint={complaint_penalty:4.1f}"
                  f"{disturbances}")
    
    # Final summary
    print()
    print("Final Results:")
    print(f"  Final temperature: {state.T_inside:.1f}°C")
    print(f"  Total energy used: {state.kwh_used:.1f} kWh")
    print(f"  Cumulative reward: {state.cumulative_reward:.1f}")
    print(f"  Comfort score: {state.cumulative_comfort:.1f}")
    print(f"  Financial score: {state.cumulative_financial:.1f}")
    
def compare_control_strategies():
    """Show how different control strategies perform"""
    print("\nControl Strategy Comparison")
    print("="*60)
    
    strategies = [
        ("Do Nothing", lambda temp_err, occ: 0.0),
        ("Simple Bang-Bang", lambda temp_err, occ: 0.8 if temp_err > 0.5 else 0.0),
        ("Proportional", lambda temp_err, occ: min(1.0, max(-1.0, temp_err * 3.0))),
    ]
    
    for strategy_name, control_fn in strategies:
        sim, settings, reward_config = create_challenging_game_setup()
        
        state = GameState(
            t=0.0, T_inside=22.0, T_outside=8.0, soc=0.5, kwh_used=0.0,
            cumulative_reward=0.0, cumulative_financial=0.0, cumulative_comfort=0.0,
            occupied=1, ts=None, T_envelope=21.5
        )
        
        prev_hvac_kw = 0.0
        total_comfort_penalty = 0.0
        
        # Test for 24 steps (6 hours)
        for step in range(24):
            occupied = 1 if step < 16 else 0  # occupied for 4 hours
            temp_error = 22.0 - state.T_inside
            
            hvac_action = control_fn(temp_error, occupied)
            action = Action(hvac=hvac_action, battery=0)
            
            inputs = {
                "settings": settings,
                "reward_cfg": reward_config,
                "T_outside": 8.0,
                "occupied_home": occupied,
                "prev_hvac_kw": prev_hvac_kw,
                "complaint_count": 0,
                "step_count": step,
            }
            
            result = sim.step(state, action, inputs)
            state = result.state
            prev_hvac_kw = result.metrics["prev_hvac_kw"]
            total_comfort_penalty += result.metrics["comfort_penalty"]
        
        print(f"{strategy_name:15}: Final T_in={state.T_inside:5.1f}°C, "
              f"Energy={state.kwh_used:4.1f}kWh, "
              f"Comfort penalty={total_comfort_penalty:6.1f}")

def main():
    """Run demonstration examples"""
    demonstrate_gameplay_loop()
    compare_control_strategies()
    
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("• Normal/Hard difficulty requires active control to maintain comfort")
    print("• Tighter comfort bands (±0.5°C occupied) demand precise control")
    print("• HVAC ramp limiting prevents instant response")
    print("• Disturbances (wind, solar pops, window events) add unpredictability")
    print("• Complaint penalties punish sustained discomfort")
    print("• Game rewards both comfort and energy efficiency")

if __name__ == "__main__":
    main()