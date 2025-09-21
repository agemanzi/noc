#therman_game/engine/physics.py
def hvac_power_kw(hvac: float) -> float:
    # map -1..1 to cooling/heating power; sign -> direction, magnitude -> kW
    return 3.0 * max(-1.0, min(1.0, hvac))

def thermal_step(Ti, To, hvac_kw, dt):
    # simple RC model
    R, C = 2.0, 3.0       # house params
    q_hvac = 0.8 * hvac_kw  # effective thermal power (kW_th)
    dT = ( (To - Ti)/R + q_hvac ) * dt / C
    return Ti + dT

def battery_step(soc, act, dt):
    rate = 0.25  # C-rate per hour
    ds = act * rate * (dt/3600.0)
    return min(1.0, max(0.0, soc + ds))
