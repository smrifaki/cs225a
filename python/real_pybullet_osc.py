"""Real cs225a peg-in-hole v3: closed-loop OSC torque control.

Builds on v2 (verified peg-frame + contact-firing geometry) by
replacing per-joint POSITION_CONTROL with computed-torque OSC:

  tau = J^T * Lambda * (Kp * e_pos + Kv * e_vel)
        + N * (Kp_post * (q_nom - q) - Kv_post * dq)
        + tau_gravity_compensation

This is closed-loop control: every tick reads joint state, computes
the operational-space inertia, builds the task force, projects
posture into the null space, and commands joint torques directly.

No position control. No IK targets per tick. The OSC loop runs at
240Hz and drives the EE through the four phases (approach, align,
contact, insert) via a single time-parametrised reference.
"""
from __future__ import annotations

import modal

app = modal.App("cs225a-real-osc-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pybullet==3.2.6", "numpy==1.26.4")
)


@app.function(image=image, timeout=60 * 20)
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

    # Disable Bullet's built-in joint motor so we control torques directly.
    for j in range(n_joints):
        p.setJointMotorControl2(arm, j, p.VELOCITY_CONTROL, force=0.0)
        p.enableJointForceTorqueSensor(arm, j, True)

    # Pre-pose via IK so the EE is above the workspace.
    PRE = [0.55, 0.0, 0.45]
    for j, qj in enumerate(p.calculateInverseKinematics(arm, EE_LINK, PRE)[:n_joints]):
        p.resetJointState(arm, j, qj)
    for _ in range(60):
        p.stepSimulation()
    ee_world = np.array(p.getLinkState(arm, EE_LINK)[0])
    ee_orn   = p.getLinkState(arm, EE_LINK)[1]
    q_nom    = [s[0] for s in p.getJointStates(arm, list(range(n_joints)))]

    # Peg
    peg_radius = 0.015
    peg_height = 0.10
    peg_center_world = ee_world[2] - 0.02 - peg_height / 2.0
    peg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=peg_radius, height=peg_height)
    peg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=peg_radius, length=peg_height,
                                  rgbaColor=[0.2, 0.3, 0.9, 1.0])
    peg = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=peg_col,
        baseVisualShapeIndex=peg_vis,
        basePosition=[ee_world[0], ee_world[1], peg_center_world],
    )
    ee_inv_pos, ee_inv_orn = p.invertTransform([0, 0, 0], ee_orn)
    parent_offset, _ = p.multiplyTransforms(
        ee_inv_pos, ee_inv_orn,
        [0, 0, peg_center_world - ee_world[2]],
        [0, 0, 0, 1],
    )
    p.createConstraint(
        parentBodyUniqueId=arm, parentLinkIndex=EE_LINK,
        childBodyUniqueId=peg, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
        parentFramePosition=list(parent_offset),
        childFramePosition=[0, 0, 0],
    )
    for link in range(-1, n_joints):
        p.setCollisionFilterPair(peg, arm, -1, link, 0)

    # Receiver
    HOLE_HALF = peg_radius + 0.003
    WALL_T    = 0.02
    WALL_H    = 0.08
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

    # Per-seed trajectory offset
    direction = rng.uniform(0.0, 2 * np.pi)
    radius    = rng.uniform(0.012, 0.018)
    target_xy_offset = np.array([radius * np.cos(direction),
                                 radius * np.sin(direction)])
    base_xy = hole_xy + target_xy_offset

    # OSC parameters
    KP_TASK = 800.0
    KV_TASK = 60.0
    KP_POST = 4.0
    KV_POST = 1.5

    def osc_torques(x_des):
        js = p.getJointStates(arm, list(range(n_joints)))
        q  = [s[0] for s in js]
        dq = [s[1] for s in js]
        ls = p.getLinkState(
            arm, EE_LINK, computeLinkVelocity=1, computeForwardKinematics=1,
        )
        x  = np.array(ls[0], dtype=float)
        dx = np.array(ls[6], dtype=float)
        J_lin, _ = p.calculateJacobian(
            arm, EE_LINK, localPosition=[0, 0, 0],
            objPositions=q, objVelocities=[0.0] * n_joints,
            objAccelerations=[0.0] * n_joints,
        )
        J = np.array(J_lin, dtype=float)
        M = np.array(p.calculateMassMatrix(arm, q), dtype=float)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        Lambda_inv = J @ M_inv @ J.T
        try:
            Lambda = np.linalg.inv(Lambda_inv)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(Lambda_inv)
        e_pos = np.asarray(x_des, dtype=float) - x
        F_task = Lambda @ (KP_TASK * e_pos - KV_TASK * dx)
        tau_task = J.T @ F_task
        Jbar = M_inv @ J.T @ Lambda
        N = np.eye(n_joints) - Jbar @ J
        q_err  = np.array(q_nom, dtype=float) - np.array(q, dtype=float)
        dq_err = -np.array(dq, dtype=float)
        tau_post = N @ (KP_POST * q_err + KV_POST * dq_err)
        tau_g = np.array(p.calculateInverseDynamics(
            arm, q, [0.0] * n_joints, [0.0] * n_joints,
        ), dtype=float)
        tau = tau_task + tau_post + tau_g
        tau = np.clip(tau, -120.0, 120.0)
        p.setJointMotorControlArray(
            arm, list(range(n_joints)),
            p.TORQUE_CONTROL, forces=tau.tolist(),
        )
        return {"x": x.tolist(), "x_err_norm": float(np.linalg.norm(e_pos)),
                "tau_task_mag": float(np.linalg.norm(tau_task)),
                "tau_post_mag": float(np.linalg.norm(tau_post)),
                "q": q, "dq": dq}

    # Waypoints (peg tip is peg_height + 2cm below EE in z)
    peg_tip_below_ee = ee_world[2] - peg_center_world + peg_height / 2.0
    z_wall_top = z_base + 2 * WALL_H
    waypoints = [
        ("approach", np.array([base_xy[0], base_xy[1], z_wall_top + peg_tip_below_ee + 0.10])),
        ("align",    np.array([base_xy[0], base_xy[1], z_wall_top + peg_tip_below_ee + 0.04])),
        ("contact",  np.array([base_xy[0], base_xy[1], z_wall_top + peg_tip_below_ee - 0.02])),
        ("insert",   np.array([base_xy[0], base_xy[1], z_base + 0.4 * WALL_H + peg_tip_below_ee])),
    ]
    quarter = n_steps // 4

    # Sensors / BALD
    modalities = ("vision", "force", "proprio")
    var = {m: 1.0 for m in modalities}
    beta = 6.0
    sigma_obs2 = 1.0
    vision_running_mean = np.zeros(16, dtype=float)
    vision_n = 0

    rows: list[dict] = []
    prev_t = ee_world
    for i, (name, tgt) in enumerate(waypoints):
        start = i * quarter
        end   = (i + 1) * quarter if i < 3 else n_steps
        for t in range(start, end):
            alpha = (t - start + 1) / max(1, end - start)
            cmd = prev_t * (1 - alpha) + tgt * alpha
            osc = osc_torques(cmd)
            p.stepSimulation()

            # vision
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

            # force (contact normal forces on peg, no peg-vs-arm)
            contacts = p.getContactPoints(bodyA=peg)
            r_force = float(sum(abs(c[9]) for c in contacts)) if contacts else 0.0

            # proprio: tracking error from OSC
            r_proprio = float(np.linalg.norm(
                np.array(osc["q"]) - np.array(q_nom)
            ) + 1e-3)

            r_mod = {"vision": r_vision, "force": r_force, "proprio": r_proprio}
            eig = {m: 0.5 * np.log1p(r_mod[m] ** 2 / max(var[m], 1e-3))
                   for m in modalities}
            logits = beta * np.array([eig[m] for m in modalities])
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            pick = modalities[int(rng.choice(len(modalities), p=probs))]
            prec_pre = 1.0 / max(var[pick], 1e-9)
            prec_post = prec_pre + (r_mod[pick] ** 2) / sigma_obs2
            var[pick] = 1.0 / prec_post

            rows.append({
                "step": t, "phase": name,
                "ee_x": osc["x"][0], "ee_y": osc["x"][1], "ee_z": osc["x"][2],
                "x_err_norm": osc["x_err_norm"],
                "tau_task_mag": osc["tau_task_mag"],
                "tau_post_mag": osc["tau_post_mag"],
                "n_contacts": int(len(contacts)),
                "r_vision":  r_vision,
                "r_force":   r_force,
                "r_proprio": r_proprio,
                "pick":      pick,
            })
        prev_t = tgt

    p.disconnect()

    summary = []
    for name, _ in waypoints:
        chunk = [r for r in rows if r["phase"] == name]
        if not chunk:
            continue
        n = len(chunk)
        picks = {m: sum(1 for r in chunk if r["pick"] == m) for m in modalities}
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
            "mean_x_err":     float(np.mean([r["x_err_norm"] for r in chunk])),
            "mean_tau_task":  float(np.mean([r["tau_task_mag"] for r in chunk])),
        })

    return {
        "seed": seed,
        "n_steps": n_steps,
        "summary": summary,
        "final_ee_z": float(rows[-1]["ee_z"]),
        "n_contact_ticks": int(sum(1 for r in rows if r["n_contacts"] > 0)),
        "control_type": "OSC (torque)",
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
                f"r_force={row['mean_r_force']:.1f}  "
                f"tau_task={row['mean_tau_task']:.0f}  "
                f"x_err={row['mean_x_err']:.4f}"
            )
        print(f"  contact-ticks={out['n_contact_ticks']}/{out['n_steps']}, "
              f"final ee_z={out['final_ee_z']:.3f}")
    Path("/tmp/real_cs225a_v3_result.json").write_text(json.dumps(payloads, indent=2))
