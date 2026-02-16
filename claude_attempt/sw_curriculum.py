curriculum_stages = [
    # Stage 0: Stationary target, no gravity, big target, no enemy ammo
    {"grav_multiplier": 0.0, "size_multiplier": 3.0, "target_speed": 0.0, "target_ammo": 0.0},
    # Stage 1: Introduce weak gravity
    {"grav_multiplier": 0.3, "size_multiplier": 3.0, "target_speed": 0.0, "target_ammo": 0.0},
    # Stage 2: Full gravity, still big stationary target
    {"grav_multiplier": 1.0, "size_multiplier": 2.0, "target_speed": 0.0, "target_ammo": 0.0},
    # Stage 3: Normal-sized stationary target
    {"grav_multiplier": 1.0, "size_multiplier": 1.0, "target_speed": 0.0, "target_ammo": 0.0},
    # Stage 4: Slow moving target
    {"grav_multiplier": 1.0, "size_multiplier": 1.0, "target_speed": 0.3, "target_ammo": 0.0},
    # Stage 5: Full speed target
    {"grav_multiplier": 1.0, "size_multiplier": 1.0, "target_speed": 1.0, "target_ammo": 0.0},
    # Stage 6: Target shoots back (partial)
    {"grav_multiplier": 1.0, "size_multiplier": 1.0, "target_speed": 1.0, "target_ammo": 0.5},
    # Stage 7: Full difficulty
    {"grav_multiplier": 1.0, "size_multiplier": 1.0, "target_speed": 1.0, "target_ammo": 1.0},
]