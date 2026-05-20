# Reproduce

The C++ controller runs inside the lab's SCL/OpenSai sim. When
that binary is not available, the real Kuka iiwa peg-in-hole runs
on Modal CPU under PyBullet:

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install modal numpy
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

# Closed-loop OSC torque control (lambda inertia + null-space
# posture + gravity compensation). Real τ commanded into the joints.
modal run python/real_pybullet_osc.py::main --seeds 0,1,2 --n-steps 1500

# Position-control variant (per-joint POSITION_CONTROL with IK
# targets per phase). Reaches deep insertion with sustained contact.
modal run python/real_pybullet.py::main --seeds 0,1,2 --n-steps 1500
```

Both use the standard `pybullet_data` Kuka iiwa model, a peg
(1.5 cm radius, 10 cm length) fixed-constrained to the EE link
with a verified world-aligned offset, and a 4-wall receiver with
3 mm radial clearance vs the peg. A per-seed lateral target
offset in [12, 18] mm forces the peg to scrape a wall during
descent.

Headline checks after a run:

* `results/real_pybullet/peg_in_hole_v2.json` (position control)
  has 3 rows; insert-phase `force_share` ≈ 1.00 and
  `mean_r_force` 299–316 N.
* `results/real_pybullet/osc_v3.json` (OSC torque control) has
  3 rows; approach-phase `force_share` ≈ 0.4–0.5 with
  `mean_r_force` 471–902 N during the OSC settle transient.
* `final_ee_z` per seed reports how deep the peg went; v2 lands
  near 0.18 m (inside the receiver), v3 settles slightly above.

The full per-tick traces are not bundled in the repo (the JSON
balloons to >1 MB per seed); rerun via `modal run` to regenerate
them.
