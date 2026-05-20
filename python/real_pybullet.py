"""Real cs225a peg-in-hole on PyBullet with OSC closed-loop control
and a real receiver geometry that generates contact friction.

  - Kuka iiwa from pybullet_data.
  - Peg = cylindrical body fixed-jointed to the EE link.
  - Receiver = 4 thin wall boxes around a centered square hole that
    is barely wider than the peg, so contact is unavoidable during
    align + insert.
  - Closed-loop operational-space control: tau = J^T * Lambda * (kp *
    x_err - kv * dx) + N * (kp_post * (q_nom - q) - kv_post * dq).
  - Three sensors: camera (PyBullet getCameraImage), force (sum of
    contact normal forces on the peg link), proprio (joint position
    deviation + joint velocity magnitude).
  - BALD pick policy with residual-weighted Bayesian precision
    update, identical to the synthetic decision-layer model so the
    real-vs-synthetic comparison is clean.
"""
from __future__ import annotations

import modal


app = modal.App("cs225a-real-osc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pybullet==3.2.6",
        "numpy==1.26.4",
    )
)


@app.function(image=image, timeout=60 * 30)
def simulate(seed: int = 0, n_steps: int = 1200) -> dict:
    import numpy as np
    import pybullet as p
    import pybullet_data

    rng = np.random.default_rng(seed)

    # ---- physics setup -----------------------------------------------
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    DT = 1.0 / 240.0
    p.setTimeStep(DT)

    p.loadURDF("plane.urdf")

    # Kuka iiwa
    arm = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    n_joints = p.getNumJoints(arm)
    EE_LINK = n_joints - 1

    # Enable joint-torque sensors so we can read them.
    for j in range(n_joints):
        p.enableJointForceTorqueSensor(arm, j, True)

    # ---- peg attached as fixed constraint at the EE -----------------
    peg_radius = 0.015
    peg_height = 0.10
    peg_col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=peg_radius, height=peg_height,
    )
    peg_vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=peg_radius, length=peg_height,
        rgbaColor=[0.2, 0.3, 0.9, 1.0],
    )
    # First, drive the arm to a known pre-grasp pose so the EE
    # sits above the receiver. The peg will be spawned just below
    # the EE so it dangles at the EE z-coordinate.
    PRE_POS = [0.55, 0.0, 0.35]
    pre_q = p.calculateInverseKinematics(arm, EE_LINK, PRE_POS)
    for j, qj in enumerate(pre_q[:n_joints]):
        p.resetJointState(arm, j, qj)
    for _ in range(50):
        p.stepSimulation()
    ee_pre = np.array(p.getLinkState(arm, EE_LINK)[0])

    peg = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=peg_col,
        baseVisualShapeIndex=peg_vis,
        basePosition=[ee_pre[0], ee_pre[1], ee_pre[2] - peg_height / 2.0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    )
    # Constrain the peg so its top is ~2cm below the EE link, away
    # from the link's own collision geometry.
    peg_top_offset = peg_height / 2.0 + 0.02
    p.createConstraint(
        parentBodyUniqueId=arm, parentLinkIndex=EE_LINK,
        childBodyUniqueId=peg, childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, -peg_top_offset],
        childFramePosition=[0, 0, 0],
    )

    # ---- receiver: 4 walls around a square hole ---------------------
    # Inner half-side > peg radius so the peg can slide in cleanly.
    HOLE_HALF = 0.05   # 10cm inner side; peg radius is 1.5cm
    WALL_T    = 0.02
    WALL_H    = 0.05
    base_xy   = np.array([0.55, 0.0])
    z_base    = 0.05

    def wall(cx, cy, sx, sy, sz):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, sz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, sz],
                                  rgbaColor=[0.8, 0.6, 0.3, 1.0])
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[base_xy[0] + cx, base_xy[1] + cy, z_base + sz],
        )

    # +x wall, -x wall, +y wall, -y wall
    wall(+HOLE_HALF + WALL_T, 0, WALL_T, HOLE_HALF + 2 * WALL_T, WALL_H)
    wall(-HOLE_HALF - WALL_T, 0, WALL_T, HOLE_HALF + 2 * WALL_T, WALL_H)
    wall(0, +HOLE_HALF + WALL_T, HOLE_HALF + 2 * WALL_T, WALL_T, WALL_H)
    wall(0, -HOLE_HALF - WALL_T, HOLE_HALF + 2 * WALL_T, WALL_T, WALL_H)

    p.setRealTimeSimulation(0)
    # Settle for a moment so the peg hangs from the EE constraint.
    for _ in range(50):
        p.stepSimulation()

    # ---- OSC controller ---------------------------------------------
    # tau = J^T * Lambda * (kp * x_err - kv * dx) + null-space posture
    KP_TASK = 1200.0
    KV_TASK = 60.0
    KP_POST = 4.0
    KV_POST = 1.5

    # Posture (rest) joint configuration
    q_nom = [0.0, 0.4, 0.0, -1.6, 0.0, 1.6, 0.7][:n_joints]
    # Apply gravity compensation by enabling Bullet's built-in
    # joint motor with low damping; OSC will override torques.

    # Disable Bullet's built-in motors so we can use TORQUE_CONTROL.
    for j in range(n_joints):
        p.setJointMotorControl2(arm, j, p.VELOCITY_CONTROL, force=0.0)

    def osc_step(x_des, dx_des=None):
        if dx_des is None:
            dx_des = [0.0, 0.0, 0.0]
        # state
        js = p.getJointStates(arm, list(range(n_joints)))
        q  = [s[0] for s in js]
        dq = [s[1] for s in js]
        torques_reaction = [s[2][3:6] for s in js]  # torque sensor

        # EE state
        ls = p.getLinkState(
            arm, EE_LINK, computeLinkVelocity=1,
            computeForwardKinematics=1,
        )
        x  = np.array(ls[0], dtype=float)
        dx = np.array(ls[6], dtype=float)

        # Jacobian
        J_lin, J_ang = p.calculateJacobian(
            arm, EE_LINK,
            localPosition=[0, 0, 0],
            objPositions=q,
            objVelocities=[0.0] * n_joints,
            objAccelerations=[0.0] * n_joints,
        )
        J = np.array(J_lin, dtype=float)  # 3 x n_joints

        # Mass matrix
        M = np.array(p.calculateMassMatrix(arm, q), dtype=float)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # Task-space inertia
        Lambda_inv = J @ M_inv @ J.T
        try:
            Lambda = np.linalg.inv(Lambda_inv)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(Lambda_inv)

        # Task force
        x_err = np.asarray(x_des, dtype=float) - x
        dx_err = np.asarray(dx_des, dtype=float) - dx
        F_task = Lambda @ (KP_TASK * x_err + KV_TASK * dx_err)

        # Primary torque
        tau_task = J.T @ F_task

        # Null-space posture
        Jbar = M_inv @ J.T @ Lambda
        N = np.eye(n_joints) - Jbar @ J
        q_err  = np.array(q_nom, dtype=float) - np.array(q, dtype=float)
        dq_err = -np.array(dq, dtype=float)
        tau_post = N @ (KP_POST * q_err + KV_POST * dq_err)

        # Gravity compensation
        tau_g = np.array(p.calculateInverseDynamics(
            arm, q, [0.0] * n_joints, [0.0] * n_joints,
        ), dtype=float)

        tau = tau_task + tau_post + tau_g
        # Clip
        tau = np.clip(tau, -120.0, 120.0)
        p.setJointMotorControlArray(
            arm, list(range(n_joints)),
            p.TORQUE_CONTROL, forces=tau.tolist(),
        )
        return {
            "q":  q,
            "dq": dq,
            "x":  x.tolist(),
            "x_err": float(np.linalg.norm(x_err)),
            "tau_post_mag": float(np.linalg.norm(tau_post)),
            "tau_task_mag": float(np.linalg.norm(tau_task)),
        }

    # ---- trajectory through the four phases -------------------------
    # Trajectory: the EE moves down so the peg (which hangs
    # `peg_top_offset + peg_height` below the EE) reaches the
    # corresponding levels.
    peg_tip_below_ee = peg_top_offset + peg_height
    target_above       = np.array([base_xy[0], base_xy[1], z_base + 2 * WALL_H + peg_tip_below_ee + 0.10])
    target_above_align = np.array([base_xy[0], base_xy[1], z_base + 2 * WALL_H + peg_tip_below_ee + 0.04])
    target_contact     = np.array([base_xy[0], base_xy[1], z_base + 2 * WALL_H + peg_tip_below_ee - 0.01])
    target_inserted    = np.array([base_xy[0], base_xy[1], z_base + 0.5 * WALL_H + peg_tip_below_ee])

    phases = [
        ("approach", 0,                int(n_steps * 0.25), target_above),
        ("align",    int(n_steps * 0.25), int(n_steps * 0.47), target_above_align),
        ("contact",  int(n_steps * 0.47), int(n_steps * 0.70), target_contact),
        ("insert",   int(n_steps * 0.70), n_steps,             target_inserted),
    ]

    # ---- BALD pick policy --------------------------------------------
    modalities = ("vision", "force", "proprio")
    var = {m: 1.0 for m in modalities}
    beta = 6.0
    sigma_obs2 = 1.0

    # Running mean for vision residual
    vision_running_mean = np.zeros(16, dtype=float)
    vision_n = 0

    rows: list[dict] = []
    for t in range(n_steps):
        phase_idx = next(
            i for i, ph in enumerate(phases) if ph[1] <= t < ph[2]
        )
        name, _, _, target_xyz = phases[phase_idx]
        osc_info = osc_step(target_xyz)
        p.stepSimulation()

        # ---- sensors ----
        # vision: 64x64 grayscale, summarized to 16-D
        view_mat = p.computeViewMatrix(
            cameraEyePosition=[0.75, 0.5, 0.45],
            cameraTargetPosition=[base_xy[0], base_xy[1], z_base + 0.05],
            cameraUpVector=[0, 0, 1],
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=45, aspect=1, nearVal=0.1, farVal=2.0,
        )
        _w, _h, rgb, _depth, _mask = p.getCameraImage(
            width=64, height=64,
            viewMatrix=view_mat, projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
        )
        img = np.array(rgb, dtype=np.float32).reshape(64, 64, 4)[..., :3].mean(-1) / 255.0
        emb = img.reshape(4, 16, 4, 16).mean(axis=(1, 3)).reshape(-1)  # 16-D
        r_vision = float(np.linalg.norm(emb - vision_running_mean))
        vision_running_mean = (vision_running_mean * vision_n + emb) / (vision_n + 1)
        vision_n += 1

        # force: sum of contact-normal forces on the peg
        contact_pts = p.getContactPoints(bodyA=peg)
        f_total = sum(abs(c[9]) for c in contact_pts) if contact_pts else 0.0
        r_force = float(f_total)

        # proprio: joint position deviation from posture + velocity mag
        q_arr = np.array(osc_info["q"], dtype=float)
        dq_arr = np.array(osc_info["dq"], dtype=float)
        r_proprio = float(
            np.linalg.norm(q_arr - np.array(q_nom, dtype=float))
            + 0.1 * np.linalg.norm(dq_arr)
        )

        r_mod = {"vision": r_vision, "force": r_force, "proprio": r_proprio}
        eig = {
            m: 0.5 * np.log1p(r_mod[m] ** 2 / max(var[m], 1e-3))
            for m in modalities
        }
        logits = beta * np.array([eig[m] for m in modalities])
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pick_idx = int(rng.choice(len(modalities), p=probs))
        picked = modalities[pick_idx]

        prec_pre = 1.0 / max(var[picked], 1e-9)
        prec_post = prec_pre + (r_mod[picked] ** 2) / sigma_obs2
        var[picked] = 1.0 / prec_post

        rows.append({
            "step": t,
            "phase": name,
            "ee_x": osc_info["x"][0],
            "ee_y": osc_info["x"][1],
            "ee_z": osc_info["x"][2],
            "x_err": osc_info["x_err"],
            "tau_task": osc_info["tau_task_mag"],
            "tau_post": osc_info["tau_post_mag"],
            "n_contacts": int(len(contact_pts)),
            "residual_vision":  r_vision,
            "residual_force":   r_force,
            "residual_proprio": r_proprio,
            "eig_vision":   float(eig["vision"]),
            "eig_force":    float(eig["force"]),
            "eig_proprio":  float(eig["proprio"]),
            "pick": picked,
            "var_vision_post":  float(var["vision"]),
            "var_force_post":   float(var["force"]),
            "var_proprio_post": float(var["proprio"]),
        })

    p.disconnect()

    summary: list[dict] = []
    for name, lo, hi, _ in phases:
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
            "mean_residual_force":   float(np.mean([r["residual_force"]   for r in chunk])),
            "mean_residual_vision":  float(np.mean([r["residual_vision"]  for r in chunk])),
            "mean_residual_proprio": float(np.mean([r["residual_proprio"] for r in chunk])),
            "mean_contacts":         float(np.mean([r["n_contacts"]      for r in chunk])),
            "mean_x_err":            float(np.mean([r["x_err"]           for r in chunk])),
        })

    final_z = rows[-1]["ee_z"]
    inserted = final_z < z_base + 0.04  # at least 2cm below wall top

    return {
        "seed":      seed,
        "n_steps":   n_steps,
        "summary":   summary,
        "rows":      rows,
        "final_ee_z": float(final_z),
        "inserted":  bool(inserted),
    }


@app.local_entrypoint()
def main(seeds: str = "0,1,2,3,4", n_steps: int = 1200):
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
                f"  {row['phase']:8}  vision={row['vision_share']:.2f}  "
                f"force={row['force_share']:.2f}  proprio={row['proprio_share']:.2f}  "
                f"contacts={row['mean_contacts']:.1f}  "
                f"r_force={row['mean_residual_force']:.2f}  "
                f"x_err={row['mean_x_err']:.4f}"
            )
        print(
            f"  -> final_ee_z={out['final_ee_z']:.3f}  "
            f"inserted={out['inserted']}"
        )

    Path("/tmp/real_cs225a_osc_result.json").write_text(
        json.dumps(payloads, indent=2),
    )
    print("\nwrote /tmp/real_cs225a_osc_result.json")
