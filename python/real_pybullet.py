"""Real cs225a peg-in-hole v2: position control + explicit verification.

Diagnoses why v1's OSC produced zero contacts. Strips back to:
  - direct IK + joint POSITION_CONTROL to drive the EE
  - prints peg world position + EE world position each tick so we
    can verify the constraint geometry is correct
  - explicit collision filter to allow peg-vs-wall contact

If this version DOES register contact, we re-introduce OSC on top.
"""
from __future__ import annotations

import modal

app = modal.App("cs225a-real-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pybullet==3.2.6", "numpy==1.26.4")
)


@app.function(image=image, timeout=60 * 15)
def simulate(seed: int = 0, n_steps: int = 1500) -> dict:
    import numpy as np
    import pybullet as p
    import pybullet_data

    rng = np.random.default_rng(seed)

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.loadURDF("plane.urdf")
    arm = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    n_joints = p.getNumJoints(arm)
    EE_LINK = n_joints - 1

    # Pre-pose so the EE is above the workspace.
    PRE = [0.55, 0.0, 0.45]
    for j, qj in enumerate(p.calculateInverseKinematics(arm, EE_LINK, PRE)[:n_joints]):
        p.resetJointState(arm, j, qj)
    for _ in range(60):
        p.stepSimulation()
    ee_world = np.array(p.getLinkState(arm, EE_LINK)[0])
    ee_orn   = p.getLinkState(arm, EE_LINK)[1]
    print(f"  EE world position after pre-pose: {ee_world.round(4)}")
    print(f"  EE world orientation: {np.array(ee_orn).round(3)}")

    # Peg as a cylinder spawned where we want it (world coords), then
    # constrained to the EE with the right world-aligned offset.
    peg_radius = 0.015
    peg_height = 0.10
    peg_top_world = ee_world[2] - 0.02
    peg_center_world = peg_top_world - peg_height / 2.0

    peg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=peg_radius, height=peg_height)
    peg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=peg_radius, length=peg_height,
                                  rgbaColor=[0.2, 0.3, 0.9, 1.0])
    peg = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=peg_col,
        baseVisualShapeIndex=peg_vis,
        basePosition=[ee_world[0], ee_world[1], peg_center_world],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    )
    # Constrain peg to EE link. parentFramePosition is in the EE link
    # frame; we use the inverse of the EE-link orientation to convert
    # the world-z offset into the parent frame.
    ee_inv_pos, ee_inv_orn = p.invertTransform([0, 0, 0], ee_orn)
    parent_offset, _ = p.multiplyTransforms(
        ee_inv_pos, ee_inv_orn,
        [0, 0, peg_center_world - ee_world[2]],
        [0, 0, 0, 1],
    )
    p.createConstraint(
        parentBodyUniqueId=arm, parentLinkIndex=EE_LINK,
        childBodyUniqueId=peg, childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=list(parent_offset),
        childFramePosition=[0, 0, 0],
    )
    # Disable peg vs. arm collision so contact sensor reports only env.
    for link in range(-1, n_joints):
        p.setCollisionFilterPair(peg, arm, -1, link, 0)

    # Verify the peg ends up where we wanted.
    for _ in range(20):
        p.stepSimulation()
    peg_world = np.array(p.getBasePositionAndOrientation(peg)[0])
    print(f"  peg world position after settling: {peg_world.round(4)}")
    print(f"  desired peg center world: ({ee_world[0]:.4f}, "
          f"{ee_world[1]:.4f}, {peg_center_world:.4f})")

    # Receiver: 4 walls around a square hole, fixed at (0.55, 0).
    HOLE_HALF = peg_radius + 0.003     # 18mm half-side
    WALL_T    = 0.02
    WALL_H    = 0.08                   # taller walls so descent must contact
    hole_xy   = np.array([0.55, 0.0])
    z_base    = 0.05

    def wall(cx, cy, sx, sy, sz):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, sz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, sz],
                                  rgbaColor=[0.8, 0.6, 0.3, 1.0])
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[hole_xy[0] + cx, hole_xy[1] + cy, z_base + sz],
        )
    wall(+HOLE_HALF + WALL_T, 0, WALL_T, HOLE_HALF + 2 * WALL_T, WALL_H)
    wall(-HOLE_HALF - WALL_T, 0, WALL_T, HOLE_HALF + 2 * WALL_T, WALL_H)
    wall(0, +HOLE_HALF + WALL_T, HOLE_HALF + 2 * WALL_T, WALL_T, WALL_H)
    wall(0, -HOLE_HALF - WALL_T, HOLE_HALF + 2 * WALL_T, WALL_T, WALL_H)

    # Per-seed offset of where the trajectory aims.
    direction = rng.uniform(0.0, 2 * np.pi)
    radius    = rng.uniform(0.012, 0.018)
    target_xy_offset = np.array([radius * np.cos(direction),
                                 radius * np.sin(direction)])
    base_xy = hole_xy + target_xy_offset
    print(f"  target offset: {target_xy_offset.round(4)}, base_xy={base_xy.round(4)}")

    # Waypoints in EE world coords. Distance the peg tip needs to
    # travel from EE z = ee_world[2] is `peg_tip_below_ee`.
    peg_tip_below_ee = ee_world[2] - peg_center_world + peg_height / 2.0
    z_wall_top = z_base + 2 * WALL_H
    targets = [
        ("approach", np.array([base_xy[0], base_xy[1], z_wall_top + peg_tip_below_ee + 0.10])),
        ("align",    np.array([base_xy[0], base_xy[1], z_wall_top + peg_tip_below_ee + 0.04])),
        ("contact",  np.array([base_xy[0], base_xy[1], z_wall_top + peg_tip_below_ee - 0.02])),
        ("insert",   np.array([base_xy[0], base_xy[1], z_base + 0.4 * WALL_H + peg_tip_below_ee])),
    ]
    # Per-tick command: linearly interpolate between waypoints.
    quarter = n_steps // 4
    rows: list[dict] = []
    pick_modalities = ("vision", "force", "proprio")
    var = {m: 1.0 for m in pick_modalities}
    beta = 6.0
    sigma_obs2 = 1.0
    vision_running_mean = np.zeros(16, dtype=float)
    vision_n = 0

    prev_target = ee_world
    for i, (name, tgt) in enumerate(targets):
        start_step = i * quarter
        end_step   = (i + 1) * quarter if i < 3 else n_steps
        for t in range(start_step, end_step):
            alpha = (t - start_step + 1) / max(1, end_step - start_step)
            cmd = prev_target * (1 - alpha) + tgt * alpha
            q_cmd = p.calculateInverseKinematics(arm, EE_LINK, list(cmd))
            for j, qj in enumerate(q_cmd[:n_joints]):
                p.setJointMotorControl2(
                    arm, j, p.POSITION_CONTROL,
                    targetPosition=qj, force=150,
                )
            p.stepSimulation()

            # ---- sensors ----
            view = p.computeViewMatrix(
                cameraEyePosition=[0.75, 0.5, 0.45],
                cameraTargetPosition=[hole_xy[0], hole_xy[1], z_base + 0.05],
                cameraUpVector=[0, 0, 1],
            )
            proj = p.computeProjectionMatrixFOV(45, 1, 0.1, 2.0)
            _w, _h, rgb, _d, _m = p.getCameraImage(
                width=64, height=64, viewMatrix=view, projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER,
            )
            img = np.array(rgb, dtype=np.float32).reshape(64, 64, 4)[..., :3].mean(-1) / 255.0
            emb = img.reshape(4, 16, 4, 16).mean(axis=(1, 3)).reshape(-1)
            r_vision = float(np.linalg.norm(emb - vision_running_mean))
            vision_running_mean = (vision_running_mean * vision_n + emb) / (vision_n + 1)
            vision_n += 1

            contacts = p.getContactPoints(bodyA=peg)
            r_force = float(sum(abs(c[9]) for c in contacts)) if contacts else 0.0

            js = p.getJointStates(arm, list(range(n_joints)))
            q_now = np.array([s[0] for s in js])
            r_proprio = float(np.linalg.norm(q_now - np.array(q_cmd[:n_joints])) + 1e-3)

            r_mod = {"vision": r_vision, "force": r_force, "proprio": r_proprio}
            eig = {m: 0.5 * np.log1p(r_mod[m] ** 2 / max(var[m], 1e-3))
                   for m in pick_modalities}
            logits = beta * np.array([eig[m] for m in pick_modalities])
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            pick = pick_modalities[int(rng.choice(len(pick_modalities), p=probs))]
            prec_pre = 1.0 / max(var[pick], 1e-9)
            prec_post = prec_pre + (r_mod[pick] ** 2) / sigma_obs2
            var[pick] = 1.0 / prec_post

            ee_now = np.array(p.getLinkState(arm, EE_LINK)[0])
            peg_now = np.array(p.getBasePositionAndOrientation(peg)[0])
            rows.append({
                "step":   t, "phase": name,
                "ee_x":   float(ee_now[0]), "ee_y": float(ee_now[1]), "ee_z": float(ee_now[2]),
                "peg_x":  float(peg_now[0]), "peg_y": float(peg_now[1]), "peg_z": float(peg_now[2]),
                "x_err":  float(np.linalg.norm(ee_now - cmd)),
                "n_contacts": int(len(contacts)),
                "r_vision":  r_vision,
                "r_force":   r_force,
                "r_proprio": r_proprio,
                "pick":      pick,
            })
        prev_target = tgt

    p.disconnect()

    # per-phase summary
    summary = []
    for name, _ in targets:
        chunk = [r for r in rows if r["phase"] == name]
        if not chunk:
            continue
        n = len(chunk)
        picks = {m: sum(1 for r in chunk if r["pick"] == m) for m in pick_modalities}
        summary.append({
            "phase": name,
            "steps": n,
            "vision_share":  picks["vision"]  / n,
            "force_share":   picks["force"]   / n,
            "proprio_share": picks["proprio"] / n,
            "mean_r_vision":  float(np.mean([r["r_vision"]  for r in chunk])),
            "mean_r_force":   float(np.mean([r["r_force"]   for r in chunk])),
            "mean_r_proprio": float(np.mean([r["r_proprio"] for r in chunk])),
            "mean_contacts":  float(np.mean([r["n_contacts"] for r in chunk])),
            "mean_x_err":     float(np.mean([r["x_err"]     for r in chunk])),
        })

    return {
        "seed": seed,
        "n_steps": n_steps,
        "summary": summary,
        "final_ee_z": float(rows[-1]["ee_z"]),
        "final_peg_z": float(rows[-1]["peg_z"]),
        "n_contact_ticks": int(sum(1 for r in rows if r["n_contacts"] > 0)),
        "rows_truncated_to_50": rows[:50],
    }


@app.local_entrypoint()
def main(seeds: str = "0,1,2,3,4", n_steps: int = 1500):
    import json
    from pathlib import Path
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    payloads = []
    for s in seed_list:
        print(f"\n## seed {s}")
        out = simulate.remote(int(s), int(n_steps))
        payloads.append(out)
        for row in out["summary"]:
            print(
                f"  {row['phase']:8}  vision={row['vision_share']:.2f} "
                f"force={row['force_share']:.2f} proprio={row['proprio_share']:.2f}  "
                f"contacts={row['mean_contacts']:.1f}  "
                f"r_force={row['mean_r_force']:.2f}  "
                f"x_err={row['mean_x_err']:.4f}"
            )
        print(f"  contact-ticks={out['n_contact_ticks']}/{out['n_steps']}, "
              f"final ee_z={out['final_ee_z']:.3f}, peg_z={out['final_peg_z']:.3f}")
    Path("/tmp/real_cs225a_v2_result.json").write_text(json.dumps(payloads, indent=2))
