"""
Scripted-expert data collection for the Franka gear assembly task.

Runs a Markovian finite-state machine in which every FSM state is uniquely
determined by the current environment observation (no latched bits, no memory
of the previous step). The FSM has three high-level buckets — ``grasping``,
``transporting``, ``released`` — each with a few leaf sub-states. Each leaf
emits a target EEF pose plus a binary gripper command; the IK-rel controller
turns the (target − current) delta into a smooth motion over multiple steps.

The HDF5 output schema matches ``record_demos.py``.

Usage:
    # Headless, batched (recommended for production).
    python scripts/tools/collect_demos_scripted.py \
        --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
        --dataset_file ./datasets/gear_assembly_scripted.hdf5 \
        --num_envs 32 --num_demos 200 \
        --enable_cameras --headless

    # GUI mode for visual debugging (num_envs forced to 1).
    python scripts/tools/collect_demos_scripted.py \
        --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
        --dataset_file ./datasets/gear_assembly_scripted.hdf5 \
        --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI parsing must happen before the AppLauncher boots Isaac Sim.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Scripted-expert data collection for gear assembly.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0",
    help="Isaac Lab task id.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/gear_assembly_scripted.hdf5",
    help="Output HDF5 path. Appends if it already exists (matches record_demos.py behaviour).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=32,
    help="Parallel envs. Forced to 1 when running with the GUI (no --headless).",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Target *total* number of successful demos in the dataset file. 0 = run indefinitely.",
)
parser.add_argument(
    "--max_steps_per_demo",
    type=int,
    default=None,
    help="Episode timeout in env steps (default: 300 Markovian, 450 non-Markovian).",
)
parser.add_argument(
    "--step_hz",
    type=int,
    default=None,
    help="Rate limit. Default: 30 Hz in GUI mode, unlimited when --headless.",
)
parser.add_argument(
    "--n-video-trials",
    type=int,
    default=20,
    help="Number of initial scripted policy attempts to record as video.",
)
# --- Non-Markovian expert ----------------------------------------------------
parser.add_argument(
    "--non_markovian",
    action="store_true",
    help="Use the non-Markovian expert (sequential FSM + latent plan + sticky transitions + alignment cycles).",
)
parser.add_argument(
    "--nm_seed",
    type=int,
    default=None,
    help="Seed for the NM expert's hidden variables (default: derived from --seed, else random).",
)
# --- Analysis / debugging flags (all default to the original behaviour) -----
parser.add_argument("--no_noise", action="store_true", help="Zero all target, rotation and action noise.")
parser.add_argument("--no_record", action="store_true", help="Do not export an HDF5 dataset (for probe runs).")
parser.add_argument("--seed", type=int, default=None, help="Env seed (initial-pose randomisation).")
parser.add_argument(
    "--fsm_log",
    type=str,
    default=None,
    help="Directory for per-step FSM log (steps.csv) and per-episode summary (episodes.csv).",
)
parser.add_argument("--verbose_fsm", action="store_true", help="Per-step debug prints (default: GUI mode only).")
parser.add_argument(
    "--episodes_per_env",
    type=int,
    default=0,
    help="Stop once every env has completed this many episodes (any outcome). 0 = disabled.",
)
# --- Probe hooks for tolerance tests (Phase 1). Unset = Markovian default. ---
parser.add_argument("--probe_grasp_xy", type=str, default=None, help="'dx,dy' (m) world offset of grasp point from bore.")
parser.add_argument("--probe_grasp_yaw_deg", type=float, default=None, help="World yaw (deg) of top-down grasp.")
parser.add_argument("--probe_insert_xy", type=str, default=None, help="'dx,dy' (m) bore-vs-shaft XY error held in INSERT.")
parser.add_argument("--probe_insert_yaw_deg", type=float, default=None, help="Yaw error (deg) added to mesh target.")
parser.add_argument(
    "--probe_insert_latch",
    action="store_true",
    help="Once INSERT is entered, stay in INSERT (while grasped) until insert depth. Auto-on with --probe_insert_xy.",
)
parser.add_argument(
    "--probe_hold",
    type=str,
    default=None,
    help="'POINT:N[,POINT:N]' freeze-hold N steps once per episode. POINT in "
    "pre_grasp, grasp, hover, mid_insert, bottom, release.",
)
parser.add_argument("--probe_z_pregrasp", type=float, default=None, help="Override Z_PREGRASP (m).")
parser.add_argument("--probe_z_transport", type=float, default=None, help="Override Z_TRANSPORT (m).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.max_steps_per_demo is None:
    args_cli.max_steps_per_demo = 450 if args_cli.non_markovian else 300

# Force num_envs=1 in GUI mode so the human can actually watch one rollout.
if not args_cli.headless and args_cli.num_envs != 1:
    print(f"[scripted_expert] GUI mode detected — overriding --num_envs {args_cli.num_envs} → 1.")
    args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Remaining imports.
# ---------------------------------------------------------------------------
import json
import math
import sys
import time

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg

import isaaclab_mimic.envs  # noqa: F401 — registers Isaac-*-Mimic-v0

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ---------------------------------------------------------------------------
# Geometry pulled from gear_assembly_env_cfg.py.
# ---------------------------------------------------------------------------
# Gear's bore (the hole that drops onto the shaft) sits at this offset in the
# gear's local frame relative to its root prim. Used to align both the grasp
# (gripper centred on the bore) and the insertion (bore over the shaft).
BORE_LOCAL_OFFSET = (-0.045375, 0.0, 0.0)


# =========================================================================== #
# Tunables — edit these to retune the expert. All distances in meters.        #
# =========================================================================== #

# Z heights (env frame). The table surface is roughly z = 0.005; the gear's
# half-height is 0.02 so a resting gear sits with gear_pos.z ≈ 0.02.
Z_PREGRASP = 0.12  # hover over the gear before descending
Z_GRASP = 0.023  # Z position of gripper when grasping the gear
Z_TRANSPORT = 0.15  # lift height after grasp and during retreat
Z_INSERT_DEPTH = 0.06  # fingertip Z at end of insertion
Z_TABLE = 0.02
Z_TABLE_TOL = 0.020  # "gear on table" if gear_pos.z < Z_TABLE + Z_TABLE_TOL

# Position tolerances for advancing to the next FSM sub-state.
XY_TOL = 0.008
XY_TOL_INSERT = 0.002
Z_TOL = 0.006

# Gripper detection + binary commands.
GRIPPER_OPEN_VAL = 0.04  # joint pos at "fully open"
GRIPPER_CLOSED_THRESHOLD = 0.052  # min(fingers) < this  →  closed
GRIPPER_OPEN_CMD = +1.0  # BinaryJointPositionActionCfg convention: >0 → open
GRIPPER_CLOSE_CMD = -1.0  # ≤0 → close

# Per-step delta bounds (action units). The clamp is applied to the *magnitude*
# (L2 norm) of the 3-vector, not per-component, so commanded motion is purely
# along the line from current → target. ``MIN`` boosts very small deltas up to
# at least this magnitude so the robot doesn't crawl when close to the target;
# ``MAX`` scales large deltas down so single-step commands stay feasible for
# the IK-rel controller. Exactly-zero deltas pass through unchanged (so "hold
# pose" states do not jitter).
# The IK-rel controller scales by 0.5 internally — i.e. a commanded magnitude
# of 0.05 corresponds to ≈2.5 cm of actual motion per env step.
MIN_DELTA_POS = 0.02
MAX_DELTA_POS = 0.18
MIN_DELTA_ROT = 0.02
MAX_DELTA_ROT = 0.10

# Top-down ("vertical") grasp orientation in the world frame, as a (w, x, y, z)
# quaternion. (0, 1, 0, 0) is a 180° rotation about world X, which points the
# panda hand's approach axis (body +Z) straight down. Used by the MOVE_TO_GEAR_Z
# branch so the gripper is vertical before descending onto the gear.
GRASP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)

# Number of teeth on the large (held) gear. Sets the gear's rotational
# symmetry: the gear is indistinguishable under rotations of
# (360° / LARGE_GEAR_NUM_TEETH), so the expert wraps the commanded yaw into
# the nearest equivalent within ±(half tooth period) of the current gear yaw,
# producing the minimum rotation that achieves the desired mesh.
LARGE_GEAR_NUM_TEETH = 60

# Desired large-gear yaw *relative to the medium gear's yaw* during the
# transport→insert phase, in degrees. The expert reads the medium gear's actual
# orientation from the scene and rotates the gripper so that
# ``yaw(large_gear) ≡ yaw(medium_gear) + RELATIVE_YAW_OFFSET_DEG``
# (modulo the gear's tooth-symmetry period, see LARGE_GEAR_NUM_TEETH). Mesh is
# invariant under that symmetry, so the useful tuning range is just
# ±(360° / N_teeth / 2); the rest is symmetry.
RELATIVE_YAW_OFFSET_DEG = 0.7

# Target-pose noise — added to (target_x, target_y, target_z) each step
# before delta computation. Provides trajectory diversity for downstream IL.
NOISE_STD_XY = 0.006
NOISE_STD_Z = 0.005

# Target-rotation noise — a small random rotation applied to ``target_quat``
# each step before delta computation, for states that actively command an
# orientation. Expressed as a per-axis std (radians) on the rotation vector
# (axis-angle), so the perturbation is isotropic in SO(3) for small angles.
NOISE_STD_ROT = 0.025

# Action noise — added directly to the (delta_pos, delta_rot) action
# components after magnitude clamping, just before the action is sent to
# the controller. Unlike the target-pose noise above (which perturbs the
# *goal* and produces correlated multi-step drift toward a shifted target),
# action noise perturbs the *commanded motion* independently each step,
# providing additional state-distribution coverage for downstream IL.
# Expressed as per-axis std on the 3D delta vectors (m for pos, rad for rot
# axis-angle). Keep these small relative to MIN_DELTA_*/MAX_DELTA_* so the
# noisy action stays well within the controller's operating range without
# needing to re-clamp.
ACTION_NOISE_STD_POS = 0.0025
ACTION_NOISE_STD_ROT = 0.02

# =========================================================================== #

# Line-buffer stdout so progress prints survive an abrupt exit.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

if args_cli.no_noise:
    NOISE_STD_XY = NOISE_STD_Z = NOISE_STD_ROT = 0.0
    ACTION_NOISE_STD_POS = ACTION_NOISE_STD_ROT = 0.0
if args_cli.probe_z_pregrasp is not None:
    Z_PREGRASP = args_cli.probe_z_pregrasp
if args_cli.probe_z_transport is not None:
    Z_TRANSPORT = args_cli.probe_z_transport

VERBOSE_FSM = args_cli.verbose_fsm or not args_cli.headless


def _parse_xy(s):
    if s is None:
        return None
    dx, dy = (float(v) for v in s.split(","))
    return (dx, dy)


PROBE_GRASP_XY = _parse_xy(args_cli.probe_grasp_xy)
PROBE_INSERT_XY = _parse_xy(args_cli.probe_insert_xy)
PROBE_INSERT_LATCH = args_cli.probe_insert_latch or PROBE_INSERT_XY is not None
HOLD_POINTS = ("pre_grasp", "grasp", "hover", "mid_insert", "bottom", "release")
PROBE_HOLDS: dict = {}
if args_cli.probe_hold:
    for tok in args_cli.probe_hold.split(","):
        name, n = tok.split(":")
        assert name in HOLD_POINTS, f"unknown hold point {name!r}"
        PROBE_HOLDS[name] = int(n)
# Transition (prev_state, new_state) that triggers each hold, and the gripper held during it.
HOLD_TRIGGERS = {
    "pre_grasp": (("MOVE_TO_GEAR_Z", "GRASP_GEAR"), +1.0),
    "grasp": (("GRASP_GEAR", "RAISE_GEAR_Z"), -1.0),
    "hover": (("MOVE_TO_PEG_XY", "INSERT"), -1.0),
    "bottom": (("INSERT", "RELEASE"), -1.0),
    "release": (("RELEASE", "RETREAT_UP"), +1.0),
}


INITIAL_FSM_STATE = "MOVE_TO_GEAR_XY"

ALL_FSM_STATES = (
    # "grasping" bucket
    "MOVE_TO_GEAR_XY",
    "MOVE_TO_GEAR_Z",
    "GRASP_GEAR",
    # "transporting" bucket
    "RAISE_GEAR_Z",
    "MOVE_TO_PEG_XY",
    "INSERT",
    "RELEASE",
    # "released" bucket
    "RETREAT_UP",
)


def _bore_world(gear_pos: torch.Tensor, gear_quat: torch.Tensor) -> torch.Tensor:
    """``gear_bore_world = gear_pos + R(gear_quat) @ BORE_LOCAL_OFFSET``. Single env: (3,)."""
    offset = torch.tensor(BORE_LOCAL_OFFSET, dtype=gear_pos.dtype, device=gear_pos.device)
    return gear_pos + math_utils.quat_apply(gear_quat.unsqueeze(0), offset.unsqueeze(0))[0]


def _topdown_quat_with_yaw(yaw: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Top-down panda orientation with an extra rotation of ``yaw`` (rad) about world Z.

    Composition is ``R_z(yaw) · R_x(π)``, which in (w, x, y, z) is
    ``(0, cos(yaw/2), sin(yaw/2), 0)``.
    """
    half = yaw / 2.0
    return torch.tensor([0.0, math.cos(half), math.sin(half), 0.0], dtype=dtype, device=device)


def _yaw_from_quat(quat: torch.Tensor) -> float:
    """Extract the yaw component (rotation about world +Z) from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _clamp_magnitude(vec: torch.Tensor, min_mag: float, max_mag: float) -> torch.Tensor:
    """Clamp the L2 norm of ``vec`` into ``[min_mag, max_mag]``, preserving direction.

    - If the input is essentially zero (norm < 1e-9), it is returned unchanged so
      that "hold pose" states (target == current) don't produce spurious motion.
    - Otherwise the vector is rescaled to ``norm ∈ [min_mag, max_mag]``.
    """
    norm = torch.linalg.vector_norm(vec)
    if float(norm) < 1e-9:
        return vec
    target_norm = torch.clamp(norm, min=min_mag, max=max_mag)
    return vec * (target_norm / norm)


class GearAssemblyScriptedExpert:
    """Markovian FSM scripted expert for the gear-assembly task.

    Maintains one FSM state string per env in ``self.fsm_state``. Both
    ``compute_FSM_state`` and ``compute_action`` operate on a single env at a
    time; the driver loop fans them out across envs.
    """

    def __init__(self, num_envs: int, env=None):
        # The "global" FSM state — one string per env.
        self.fsm_state: list[str] = [INITIAL_FSM_STATE] * num_envs
        # Env reference is used to read the medium-gear quat from the scene
        # (it isn't exposed via the policy obs). Stays optional so the class
        # is still constructible without an env for unit testing.
        self.env = env
        # Probe-only bookkeeping (inactive unless --probe_hold is given).
        self.hold_remaining: list[int] = [0] * num_envs
        self.hold_gripper: list[float] = [GRIPPER_OPEN_CMD] * num_envs
        self.holds_fired: list[set] = [set() for _ in range(num_envs)]

    def reset_env(self, env_idx: int) -> None:
        """Reset the FSM for a single env (called when that env terminates)."""
        self.fsm_state[env_idx] = INITIAL_FSM_STATE
        self.hold_remaining[env_idx] = 0
        self.holds_fired[env_idx] = set()

    def update_holds(self, prev_state: str, new_state: str, obs: dict, env_idx: int) -> None:
        """Probe: arm a freeze-hold the first time a configured hold point is reached."""
        if not PROBE_HOLDS or self.hold_remaining[env_idx] > 0:
            return
        fired = self.holds_fired[env_idx]
        for name, n in PROBE_HOLDS.items():
            if name in fired:
                continue
            if name == "mid_insert":
                z = float(obs["policy"]["eef_pos"][env_idx][2])
                hit = new_state == "INSERT" and z < 0.5 * (Z_TRANSPORT + Z_INSERT_DEPTH)
                grip = GRIPPER_CLOSE_CMD
            else:
                (a, b), grip = HOLD_TRIGGERS[name]
                hit = prev_state == a and new_state == b
            if hit:
                fired.add(name)
                self.hold_remaining[env_idx] = n
                self.hold_gripper[env_idx] = grip
                return

    def _grasp_ref_xy(self, bore: torch.Tensor) -> torch.Tensor:
        """XY the gripper aims at when grasping (bore + optional probe offset)."""
        if PROBE_GRASP_XY is None:
            return bore[:2]
        return bore[:2] + torch.tensor(PROBE_GRASP_XY, dtype=bore.dtype, device=bore.device)

    def _insert_ref_xy(self, shaft_pos: torch.Tensor) -> torch.Tensor:
        """XY the bore is driven to during INSERT (shaft + optional probe error)."""
        if PROBE_INSERT_XY is None:
            return shaft_pos[:2]
        return shaft_pos[:2] + torch.tensor(PROBE_INSERT_XY, dtype=shaft_pos.dtype, device=shaft_pos.device)

    def mesh_yaw_error_deg(self, env_idx: int, gear_quat: torch.Tensor) -> float:
        """Signed yaw error (deg) of the held gear vs. the mesh target, wrapped to ±half tooth period."""
        if self.env is None:
            return float("nan")
        medium_yaw = _yaw_from_quat(self.env.scene["gear_medium"].data.root_quat_w[env_idx])
        period = 2.0 * math.pi / LARGE_GEAR_NUM_TEETH
        err = _yaw_from_quat(gear_quat) - medium_yaw - math.radians(RELATIVE_YAW_OFFSET_DEG)
        return math.degrees(((err + period / 2.0) % period) - period / 2.0)

    def _mesh_yaw_target_quat(
        self,
        env_idx: int,
        gear_quat: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Top-down quat that meshes the large gear's teeth with the medium gear's.

        Goal::

            yaw(large_gear) ≡ yaw(medium_gear) + RELATIVE_YAW_OFFSET_DEG
                (mod 2π / LARGE_GEAR_NUM_TEETH)

        Because the large gear is rotationally symmetric every tooth period,
        any rotation equivalent to the canonical target produces the same
        mesh. We pick the equivalent that is *closest to the current gear
        yaw* — i.e. wrap the required rotation into ``[-period/2, +period/2)``
        — so the gripper rotates by the minimum amount, never more than a
        half-tooth-period (3° for a 60-tooth gear).

        The gripper holds the gear rigidly, so once we have the minimum gear
        rotation ``Δ``, the gripper rotates by the same ``Δ``:
        ``target_grip_yaw = grip_yaw + Δ``. The result is composed with the
        top-down base orientation, ``R_z(target_grip_yaw) · R_x(π)``.
        """
        if self.env is None:
            # Fallback: ignore the medium gear and just apply the offset as an
            # absolute world-frame yaw. Useful for offline unit tests.
            return _topdown_quat_with_yaw(math.radians(RELATIVE_YAW_OFFSET_DEG), dtype, device)

        medium_gear = self.env.scene["gear_medium"]
        medium_quat = medium_gear.data.root_quat_w[env_idx]

        # All three orientations are read in the world frame; we only care
        # about their yaw component (rotation about world +Z). The gripper
        # quat is fetched on demand from the obs buffer.
        current_gripper_quat = self.env.obs_buf["policy"]["eef_quat"][env_idx]

        medium_yaw = _yaw_from_quat(medium_quat)
        gear_yaw = _yaw_from_quat(gear_quat)
        gripper_yaw = _yaw_from_quat(current_gripper_quat)

        # Canonical (one representative of the equivalence class) target yaw
        # for the large gear.
        canonical_target_gear_yaw = medium_yaw + math.radians(RELATIVE_YAW_OFFSET_DEG)
        if args_cli.probe_insert_yaw_deg is not None:
            canonical_target_gear_yaw += math.radians(args_cli.probe_insert_yaw_deg)

        # Wrap the required gear rotation into [-period/2, +period/2). Using
        # Python's floor-mod (always non-negative for positive divisor) means
        # the formula below also absorbs the natural 2π wraparound of atan2's
        # output, so adversarial cases like canonical ≈ +π and gear_yaw ≈ −π
        # produce the small in-period delta, not a 2π-sized swing.
        period = 2.0 * math.pi / LARGE_GEAR_NUM_TEETH
        delta = canonical_target_gear_yaw - gear_yaw
        delta_min = ((delta + period / 2.0) % period) - period / 2.0

        target_gripper_yaw = gripper_yaw + delta_min

        return _topdown_quat_with_yaw(target_gripper_yaw, dtype, device)

    def compute_FSM_state(self, obs: dict, env_idx: int) -> str:
        """Inspect the current obs for ``env_idx`` and return the new FSM-state string."""
        pol = obs["policy"]
        eef_pos = pol["eef_pos"][env_idx]  # (3,)
        gear_pos = pol["gear_pos"][env_idx]  # (3,)
        gear_quat = pol["gear_quat"][env_idx]  # (4,)
        shaft_pos = pol["shaft_pos"][env_idx]  # (3,)
        gripper_pos = pol["gripper_pos"][env_idx]  # (2,)
        if VERBOSE_FSM:
            print(f"    eef_pos: {eef_pos}")
            print(f"    gear_pos: {gear_pos}")
            print(f"    gear_quat: {gear_quat}")
            print(f"    shaft_pos: {shaft_pos}")
            print(f"    gripper_pos: {gripper_pos}")

        bore = _bore_world(gear_pos, gear_quat)

        grasped = math.fabs(gripper_pos[0] - gripper_pos[1]) < GRIPPER_CLOSED_THRESHOLD
        gear_on_table = float(gear_pos[2]) < (Z_TABLE + Z_TABLE_TOL)
        if VERBOSE_FSM:
            print(f"    grasped: {grasped}")
            print(f"    gear_on_table: {gear_on_table}")

        # Probe latch: hold INSERT (ignoring the 2 mm XY gate) until insertion depth.
        if (
            PROBE_INSERT_LATCH
            and self.fsm_state[env_idx] == "INSERT"
            and grasped
            and float(eef_pos[2]) > Z_INSERT_DEPTH + Z_TOL
        ):
            return "INSERT"

        if gear_on_table and not grasped:
            # "grasping" bucket.
            eef_xy_to_bore = float(torch.linalg.vector_norm(eef_pos[:2] - self._grasp_ref_xy(bore)))
            eef_z_to_grasp_z = float(math.fabs(eef_pos[2] - Z_GRASP))
            if eef_xy_to_bore > XY_TOL:
                return "MOVE_TO_GEAR_XY"
            if eef_z_to_grasp_z > Z_TOL:
                return "MOVE_TO_GEAR_Z"
            return "GRASP_GEAR"

        if grasped:
            # "transporting" bucket.
            bore_xy_to_shaft = float(torch.linalg.vector_norm(bore[:2] - shaft_pos[:2]))
            if float(eef_pos[2]) < Z_TRANSPORT - Z_TOL and bore_xy_to_shaft > XY_TOL_INSERT:
                return "RAISE_GEAR_Z"
            if bore_xy_to_shaft > XY_TOL_INSERT:
                return "MOVE_TO_PEG_XY"
            if float(eef_pos[2]) > Z_INSERT_DEPTH + Z_TOL:
                return "INSERT"
            return "RELEASE"

        # "released" bucket — everything else.
        return "RETREAT_UP"

    def compute_action(self, obs: dict, env_idx: int) -> torch.Tensor:
        """Compute the 7D action for ``env_idx`` given ``self.fsm_state[env_idx]``."""
        state = self.fsm_state[env_idx]
        if VERBOSE_FSM:
            print(f"STATE: {state}")
        pol = obs["policy"]
        # Probe freeze-hold: zero motion, fixed gripper.
        if self.hold_remaining[env_idx] > 0:
            self.hold_remaining[env_idx] -= 1
            dev, dt = pol["eef_pos"].device, pol["eef_pos"].dtype
            return torch.tensor([0, 0, 0, 0, 0, 0, self.hold_gripper[env_idx]], dtype=dt, device=dev)
        eef_pos = pol["eef_pos"][env_idx]  # (3,)
        gear_pos = pol["gear_pos"][env_idx]
        gear_quat = pol["gear_quat"][env_idx]
        shaft_pos = pol["shaft_pos"][env_idx]
        device, dtype = eef_pos.device, eef_pos.dtype

        bore = _bore_world(gear_pos, gear_quat)

        # Default: hold current pose with gripper open. ``target_quat`` stays
        # ``None`` for states that don't actively command orientation, which
        # produces ``delta_rot = 0`` (= "no rotation change") below.
        target_pos = eef_pos.clone()
        target_quat: torch.Tensor | None = None
        gripper = GRIPPER_OPEN_CMD

        grasp_xy = self._grasp_ref_xy(bore)
        insert_xy = self._insert_ref_xy(shaft_pos)

        if state == "MOVE_TO_GEAR_XY":
            target_pos[0] = grasp_xy[0]
            target_pos[1] = grasp_xy[1]
            target_pos[2] = Z_PREGRASP
            # Align the gripper to "vertical" (top-down) before descending so
            # the fingers come down straight onto the gear regardless of any
            # orientation drift accumulated by the IK-rel controller.
            if args_cli.probe_grasp_yaw_deg is None:
                target_quat = torch.tensor(GRASP_QUAT_WXYZ, dtype=dtype, device=device)
            else:
                target_quat = _topdown_quat_with_yaw(math.radians(args_cli.probe_grasp_yaw_deg), dtype, device)
            gripper = GRIPPER_OPEN_CMD
        elif state == "MOVE_TO_GEAR_Z":
            target_pos[0] = grasp_xy[0]
            target_pos[1] = grasp_xy[1]
            target_pos[2] = Z_GRASP
            gripper = GRIPPER_OPEN_CMD
        elif state == "GRASP_GEAR":
            # Hold pose; close gripper.
            gripper = GRIPPER_CLOSE_CMD
        elif state == "RAISE_GEAR_Z":
            target_pos[2] = Z_TRANSPORT
            gripper = GRIPPER_CLOSE_CMD
        elif state == "MOVE_TO_PEG_XY":
            target_pos[0] = eef_pos[0] + (shaft_pos[0] - bore[0])
            target_pos[1] = eef_pos[1] + (shaft_pos[1] - bore[1])
            target_pos[2] = Z_TRANSPORT
            # Yaw the gripper (and therefore the held gear) so the large gear's
            # yaw matches the medium gear's yaw + RELATIVE_YAW_OFFSET_DEG. No
            # jam risk up here at hover height.
            target_quat = self._mesh_yaw_target_quat(env_idx, gear_quat, dtype, device)
            gripper = GRIPPER_CLOSE_CMD
        elif state == "INSERT":
            target_pos[0] = eef_pos[0] + (insert_xy[0] - bore[0])
            target_pos[1] = eef_pos[1] + (insert_xy[1] - bore[1])
            target_pos[2] = Z_INSERT_DEPTH
            # Hold the same gear-relative-to-medium yaw while descending.
            target_quat = self._mesh_yaw_target_quat(env_idx, gear_quat, dtype, device)
            gripper = GRIPPER_CLOSE_CMD
        elif state == "RELEASE":
            # Hold pose; open gripper.
            gripper = GRIPPER_OPEN_CMD
        elif state == "RETREAT_UP":
            target_pos[2] = Z_TRANSPORT
            gripper = GRIPPER_OPEN_CMD
        else:
            raise ValueError(f"Unknown FSM state: {state!r}")

        # Small Gaussian noise on the target before computing the delta.
        noise = torch.randn(3, dtype=dtype, device=device) * torch.tensor(
            [NOISE_STD_XY, NOISE_STD_XY, NOISE_STD_Z], dtype=dtype, device=device
        )
        target_pos = target_pos + noise

        delta_pos = _clamp_magnitude(target_pos - eef_pos, MIN_DELTA_POS, MAX_DELTA_POS)

        if target_quat is None:
            # No orientation command this state — keep current EEF rotation.
            delta_rot = torch.zeros(3, dtype=dtype, device=device)
        else:
            # Perturb the target orientation by a small random rotation
            # (axis-angle noise composed in the world frame) for trajectory
            # diversity, mirroring the position noise above.
            rot_noise = torch.randn(3, dtype=dtype, device=device) * NOISE_STD_ROT
            noise_quat = math_utils.quat_from_angle_axis(
                torch.linalg.vector_norm(rot_noise).unsqueeze(0),
                rot_noise.unsqueeze(0),
            )[0]
            target_quat = math_utils.quat_mul(
                noise_quat.unsqueeze(0), target_quat.unsqueeze(0)
            )[0]

            # delta_quat such that ``delta_quat * current_quat = target_quat``
            # (world-frame composition, matches the convention used by the
            # mimic env's target_eef_pose_to_action).
            current_quat = pol["eef_quat"][env_idx]
            delta_quat = math_utils.quat_mul(
                target_quat.unsqueeze(0),
                math_utils.quat_conjugate(current_quat.unsqueeze(0)),
            )[0]
            delta_rot = math_utils.axis_angle_from_quat(delta_quat.unsqueeze(0))[0]
            delta_rot = _clamp_magnitude(delta_rot, MIN_DELTA_ROT, MAX_DELTA_ROT)

        # Per-step action noise on top of the (clamped) deltas. Applied to
        # both components regardless of state — for orientation-holding
        # states this introduces a small random rotation drift, which is
        # self-corrected on entry to the next orientation-commanding state.
        delta_pos = delta_pos + torch.randn(3, dtype=dtype, device=device) * ACTION_NOISE_STD_POS
        delta_rot = delta_rot + torch.randn(3, dtype=dtype, device=device) * ACTION_NOISE_STD_ROT

        gripper_t = torch.tensor([gripper], dtype=dtype, device=device)

        return torch.cat([delta_pos, delta_rot, gripper_t], dim=0)


# =========================================================================== #
# Non-Markovian expert (--non_markovian)                                       #
# =========================================================================== #
# Port of the FurnitureSim non-Markovian scripted expert (furniture-bench
# parts/part.py + parts/leg.py + envs/furniture_sim_env.py) to gear insertion.
# It reuses the Markovian expert's geometry, gains and noise, and adds:
#   1. Sequential FSM: the FSM state is latched; only the current state's exit
#      condition is checked (cf. leg.py's NM branch of compute_state).
#   2. Latent plan: episode-level hidden offsets on waypoints / grasp
#      (cf. Part.latent_offsets + _apply_latent_offset).
#   3. Sticky transitions: a satisfied exit condition fires only after a hidden
#      random countdown (cf. Part._nm_sticky_delay).
#   4. Latent-count alignment maneuvers: fixed-duration (step-counted) target
#      shifts before precision steps, ending with a clean pass (cf. leg.py
#      reach_leg_floor_z / reach_table_top_z alignment cycles).
# Pauses, the virtual-target walk and temporally correlated (OU) action noise
# are ported but disabled by default, as in FurnitureSim.
# Step counts are ~3x FurnitureSim's because this env runs at 30 Hz (~10 Hz there).
# Magnitudes come from the Phase 1 tolerance probes (runs/phase1/PHASE1_REPORT.md).

# ── Feature toggles ─────────────────────────────────────────────────────────
_NM_LATENT_PLAN: bool = True
_NM_STICKY: bool = True
_NM_ALIGN_CYCLES: bool = True
_NM_PAUSES: bool = False
_NM_VIRTUAL_TARGET: bool = False
_NM_CORR_NOISE: bool = False

# ── Latent plan (sampled once per episode) ──────────────────────────────────
# Grasp point offset from the bore, in the *gripper* frame (Phase 1: the grasp
# tolerates <=12 mm along the finger pads but only <=6 mm along the closing axis).
_NM_GRASP_SLIDE_STD = 0.004  # m, along the finger pads
_NM_GRASP_SLIDE_CLIP = 0.008
_NM_GRASP_CLOSE_STD = 0.002  # m, along the finger closing direction
_NM_GRASP_CLOSE_CLIP = 0.004
_NM_GRASP_YAW_MAX_DEG = 15.0  # uniform world yaw of the top-down grasp (45 deg breaks)
_NM_Z_PREGRASP_RANGE = (0.10, 0.14)  # hover height above the gear
_NM_Z_TRANSPORT_RANGE = (0.14, 0.17)  # lift / transport / hover-over-shaft height
# XY offsets of coarse transit waypoints (pre-grasp hover, lift, transit to shaft).
_NM_WAYPOINT_XY_STD = 0.007
_NM_WAYPOINT_XY_CLIP = 0.014

# ── Sticky transitions ──────────────────────────────────────────────────────
_NM_STICKY_MIN_DELAY = 15  # steps (0.5 s); Phase 1: freezes up to 60 steps are harmless
_NM_STICKY_MAX_DELAY = 25
_NM_STICKY_TRANSITIONS = (
    ("MOVE_TO_GEAR_Z", "GRASP_GEAR"),  # linger above the hub before closing
    ("GRASP_GEAR", "RAISE_GEAR_Z"),  # keep squeezing before lifting
    ("MOVE_TO_PEG_XY", "INSERT"),  # linger at the aligned hover before committing
)

# ── Alignment cycles (a): staged descent onto the gear (MOVE_TO_GEAR_Z) ─────
# Each cycle targets the grasp point shifted along the slide axis and raised;
# both shifts ramp linearly to 0 over the cycles, then a clean final pass.
_NM_DESCENT_MIN_CYCLES = 2
_NM_DESCENT_MAX_CYCLES = 3
_NM_DESCENT_STEPS_MIN = 12
_NM_DESCENT_STEPS_MAX = 18
_NM_DESCENT_SLIDE_MAX = 0.010  # slide-axis shift at the first cycle (m)
_NM_DESCENT_Z_MAX = 0.030  # height above Z_GRASP at the first cycle (m)
_NM_DESCENT_SLIDE_JITTER_STD = 0.003  # per-cycle random slide jitter (clipped at 1 std)
_NM_DESCENT_CLOSE_JITTER_STD = 0.0015  # per-cycle random closing-axis jitter (clipped at 1 std)
_NM_GRASP_SLIDE_ABS_MAX = 0.012  # absolute clamp on total slide offset (latent + cycle)
_NM_GRASP_CLOSE_ABS_MAX = 0.005  # absolute clamp on total closing offset (latent + cycle)

# ── Alignment cycles (b): hover over the shaft (MOVE_TO_PEG_XY) ─────────────
_NM_HOVER_MIN_CYCLES = 1
_NM_HOVER_MAX_CYCLES = 3
_NM_HOVER_STEPS_MIN = 18
_NM_HOVER_STEPS_MAX = 24
_NM_HOVER_XY_STD = 0.006  # bore-vs-shaft XY offset per cycle (m)
_NM_HOVER_XY_CLIP = 0.009
_NM_HOVER_YAW_MAX_DEG = 2.0  # yaw offset per cycle (hover only; insert yaw is exact)
# Clean final pass before INSERT (Phase 1: insert yaw tolerance < 0.75 deg).
_NM_CLEAN_XY_TOL = XY_TOL_INSERT  # 2 mm, same gate as the Markovian expert
_NM_CLEAN_YAW_TOL_DEG = 0.3

# ── Exit-condition tolerances ───────────────────────────────────────────────
_NM_PREGRASP_Z_TOL = 0.015  # must actually reach the (latent) hover height
_NM_PREGRASP_ORI_TOL = 0.12  # rad, to the latent grasp orientation
_NM_GRASP_XY_TOL = 0.005  # EEF vs (latent) grasp point before closing
# Robust "holding the gear" test (Phase 1: held opening 0.047-0.052, empty close ~0.0002).
_NM_HOLD_OPENING_RANGE = (0.040, 0.056)
_NM_HOLD_GEAR_XY = 0.015  # m, bore XY within this of the EEF
_NM_HOLD_GEAR_DZ = (0.0, 0.06)  # m, EEF height above the bore (resting bore z ~ -0.0105, EEF ~0.023)
_NM_RELEASED_OPENING = 0.07

# ── Noise tiers ─────────────────────────────────────────────────────────────
# Precision sub-states get no target / rotation / action noise (cf. FurnitureSim's
# no-noise states). Phase 1: jittering XY of ~2-3 mm during insertion jams the gear.
_NM_ZERO_NOISE_SUBSTATES = frozenset({"INSERT", "PEG_CLEAN"})

# ── Pauses (off by default) ─────────────────────────────────────────────────
_NM_PAUSE_MIN = 18  # steps
_NM_PAUSE_MAX = 36
_NM_PAUSE_TRANSITIONS = (
    ("MOVE_TO_GEAR_XY", "MOVE_TO_GEAR_Z"),
    ("RAISE_GEAR_Z", "MOVE_TO_PEG_XY"),
)

# ── Virtual-target walk (off by default; position only) ─────────────────────
_NM_VT_ALPHA_POS = 0.6  # velocity momentum
_NM_VT_K_POS = 0.15  # spring toward the FSM goal
_NM_VT_SIGMA_POS = 0.005  # noise std (m) at full distance
_NM_VT_FALLOFF_FRAC_POS = 0.80  # noise ramps off within this fraction of the initial distance

# ── Correlated OU action noise (off by default; executed and recorded) ──────
_NM_CORR_NOISE_ALPHA = 0.9
_NM_CORR_NOISE_STD_POS = 0.003
_NM_CORR_NOISE_STD_ROT = math.radians(3.0)


class GearAssemblyNMScriptedExpert(GearAssemblyScriptedExpert):
    """Non-Markovian scripted expert for gear assembly (see the block comment above).

    All hidden variables live in ``self.latents[i]`` (sampled at reset, fixed for
    the episode) and ``self.nm[i]`` (runtime counters: cycle index, sticky
    countdowns, latched waypoints). None of them are observable by a policy.
    """

    def __init__(self, num_envs: int, env=None, nm_seed: int | None = None):
        super().__init__(num_envs, env)
        self.rng = np.random.default_rng(nm_seed)
        self.latents: list[dict] = [{} for _ in range(num_envs)]
        self.nm: list[dict] = [{} for _ in range(num_envs)]
        for i in range(num_envs):
            self.reset_env(i)

    # ------------------------------------------------------------------ setup
    def reset_env(self, env_idx: int) -> None:
        super().reset_env(env_idx)
        self.latents[env_idx] = self._sample_latents()
        self.nm[env_idx] = {
            "t": 0,  # FSM steps this episode
            "entry_t": 0,
            "eef_xy": None,
            "raise_xy": None,  # latched lift waypoint
            "desc_idx": 0,
            "desc_end": 0,
            "peg_phase": "transit",  # transit -> cycle -> clean
            "hover_idx": 0,
            "hover_end": 0,
            "sticky": {},  # transition -> remaining countdown
            "sticky_fired": {},  # transition -> delay actually spent
            "pause_remaining": 0,
            "pause_gripper": GRIPPER_OPEN_CMD,
            "pending": None,
            "vt_pos": None,
            "vt_vel": None,
            "vt_falloff": 1e-8,
            "vt_state": None,
            "corr_pos": np.zeros(3),
            "corr_rot": np.zeros(3),
            "sub": "",
            "transitions": [],
        }

    def _sample_latents(self) -> dict:
        r = self.rng

        def clipn(std, clip, size=None):
            return np.clip(r.normal(0.0, std, size), -clip, clip)

        L: dict = {}
        if _NM_LATENT_PLAN:
            L["grasp_yaw_deg"] = float(r.uniform(-_NM_GRASP_YAW_MAX_DEG, _NM_GRASP_YAW_MAX_DEG))
            L["grasp_slide"] = float(clipn(_NM_GRASP_SLIDE_STD, _NM_GRASP_SLIDE_CLIP))
            L["grasp_close"] = float(clipn(_NM_GRASP_CLOSE_STD, _NM_GRASP_CLOSE_CLIP))
            L["z_pregrasp"] = float(r.uniform(*_NM_Z_PREGRASP_RANGE))
            L["z_transport"] = float(r.uniform(*_NM_Z_TRANSPORT_RANGE))
            L["hover_off"] = clipn(_NM_WAYPOINT_XY_STD, _NM_WAYPOINT_XY_CLIP, 2).tolist()
            L["raise_off"] = clipn(_NM_WAYPOINT_XY_STD, _NM_WAYPOINT_XY_CLIP, 2).tolist()
            L["transit_off"] = clipn(_NM_WAYPOINT_XY_STD, _NM_WAYPOINT_XY_CLIP, 2).tolist()
        else:
            L.update(grasp_yaw_deg=0.0, grasp_slide=0.0, grasp_close=0.0, z_pregrasp=Z_PREGRASP,
                     z_transport=Z_TRANSPORT, hover_off=[0.0, 0.0], raise_off=[0.0, 0.0], transit_off=[0.0, 0.0])

        L["descent_cycles"] = []
        L["hover_cycles"] = []
        if _NM_ALIGN_CYCLES:
            n = int(r.integers(_NM_DESCENT_MIN_CYCLES, _NM_DESCENT_MAX_CYCLES + 1))
            for k in range(n, 0, -1):  # offsets ramp down: k/n, ..., 1/n
                frac = k / n
                L["descent_cycles"].append({
                    "slide": float(_NM_DESCENT_SLIDE_MAX * frac
                                   + clipn(_NM_DESCENT_SLIDE_JITTER_STD, _NM_DESCENT_SLIDE_JITTER_STD)),
                    "close": float(clipn(_NM_DESCENT_CLOSE_JITTER_STD, _NM_DESCENT_CLOSE_JITTER_STD)),
                    "dz": float(_NM_DESCENT_Z_MAX * frac),
                    "steps": int(r.integers(_NM_DESCENT_STEPS_MIN, _NM_DESCENT_STEPS_MAX + 1)),
                })
            n = int(r.integers(_NM_HOVER_MIN_CYCLES, _NM_HOVER_MAX_CYCLES + 1))
            for _ in range(n):
                dxy = clipn(_NM_HOVER_XY_STD, _NM_HOVER_XY_CLIP, 2)
                L["hover_cycles"].append({
                    "dx": float(dxy[0]),
                    "dy": float(dxy[1]),
                    "yaw_deg": float(r.uniform(-_NM_HOVER_YAW_MAX_DEG, _NM_HOVER_YAW_MAX_DEG)),
                    "steps": int(r.integers(_NM_HOVER_STEPS_MIN, _NM_HOVER_STEPS_MAX + 1)),
                })

        L["sticky_delays"] = {
            f"{a}->{b}": (int(r.integers(_NM_STICKY_MIN_DELAY, _NM_STICKY_MAX_DELAY + 1)) if _NM_STICKY else 0)
            for a, b in _NM_STICKY_TRANSITIONS
        }
        L["pause_steps"] = {
            f"{a}->{b}": (int(r.integers(_NM_PAUSE_MIN, _NM_PAUSE_MAX + 1)) if _NM_PAUSES else 0)
            for a, b in _NM_PAUSE_TRANSITIONS
        }
        return L

    # -------------------------------------------------------------- geometry
    def _quantities(self, obs: dict, i: int) -> dict:
        pol = obs["policy"]
        gp, gq, sp, fp = pol["gear_pos"][i], pol["gear_quat"][i], pol["shaft_pos"][i], pol["gripper_pos"][i]
        bore = _bore_world(gp, gq)
        eef = pol["eef_pos"][i]
        return {
            "eef": eef,
            "eef_quat": pol["eef_quat"][i],
            "gear_quat": gq,
            "shaft": sp,
            "bore": bore,
            "opening": abs(float(fp[0]) - float(fp[1])),
            "bs_xy": float(torch.linalg.vector_norm(bore[:2] - sp[:2])),
            "yaw_err_deg": self.mesh_yaw_error_deg(i, gq),
        }

    def _grasp_quat(self, i: int, dtype, device) -> torch.Tensor:
        return _topdown_quat_with_yaw(math.radians(self.latents[i]["grasp_yaw_deg"]), dtype, device)

    def _grasp_point_xy(self, i: int, bore: torch.Tensor, extra_slide: float = 0.0, extra_close: float = 0.0):
        """Bore XY + latent (+ cycle) offset expressed in the gripper frame.

        For a top-down grasp R_z(yaw)·R_x(pi), the finger pads slide along the hand
        x-axis = (cos yaw, sin yaw) and the fingers close along (-sin yaw, cos yaw)
        (= world y at yaw 0, matching the Phase 1 probes).
        """
        L = self.latents[i]
        psi = math.radians(L["grasp_yaw_deg"])
        s = float(np.clip(L["grasp_slide"] + extra_slide, -_NM_GRASP_SLIDE_ABS_MAX, _NM_GRASP_SLIDE_ABS_MAX))
        c = float(np.clip(L["grasp_close"] + extra_close, -_NM_GRASP_CLOSE_ABS_MAX, _NM_GRASP_CLOSE_ABS_MAX))
        off = torch.tensor(
            [s * math.cos(psi) - c * math.sin(psi), s * math.sin(psi) + c * math.cos(psi)],
            dtype=bore.dtype, device=bore.device,
        )
        return bore[:2] + off

    def _hover_xy(self, i: int, bore: torch.Tensor) -> torch.Tensor:
        off = torch.tensor(self.latents[i]["hover_off"], dtype=bore.dtype, device=bore.device)
        return self._grasp_point_xy(i, bore) + off

    def _peg_ref_xy(self, i: int, shaft: torch.Tensor):
        """Where the bore is driven in MOVE_TO_PEG_XY, plus the hover yaw offset (rad)."""
        st, L = self.nm[i], self.latents[i]
        if st["peg_phase"] == "transit":
            off, yaw = L["transit_off"], 0.0
        elif st["peg_phase"] == "cycle":
            c = L["hover_cycles"][st["hover_idx"]]
            off, yaw = [c["dx"], c["dy"]], math.radians(c["yaw_deg"])
        else:  # clean
            off, yaw = [0.0, 0.0], 0.0
        return shaft[:2] + torch.tensor(off, dtype=shaft.dtype, device=shaft.device), yaw

    # ------------------------------------------------------- sticky / pauses
    def _sticky_delay(self, i: int, key: str) -> bool:
        """True while the hidden countdown for ``key`` is still running (cf. Part._nm_sticky_delay).

        The countdown only advances on steps where the exit condition holds, so an
        exit condition that flickers off (e.g. an empty grasp) freezes it.
        """
        st = self.nm[i]
        if key not in st["sticky"]:
            st["sticky"][key] = self.latents[i]["sticky_delays"].get(key, 0)
            st["sticky_fired"][key] = st["sticky"][key]
        if st["sticky"][key] > 0:
            st["sticky"][key] -= 1
            return True
        del st["sticky"][key]
        return False

    # ------------------------------------------------------------ sub-phases
    def _update_subphase(self, i: int, state: str, q: dict) -> None:
        st, L = self.nm[i], self.latents[i]
        t = st["t"]
        if state == "MOVE_TO_GEAR_Z":
            cyc = L["descent_cycles"]
            if st["desc_idx"] < len(cyc) and t >= st["desc_end"]:
                st["desc_idx"] += 1
                if st["desc_idx"] < len(cyc):
                    st["desc_end"] = t + cyc[st["desc_idx"]]["steps"]
            st["sub"] = f"desc{st['desc_idx']}" if st["desc_idx"] < len(cyc) else "desc_clean"
        elif state == "MOVE_TO_PEG_XY":
            cyc = L["hover_cycles"]
            if st["peg_phase"] == "transit":
                ref, _ = self._peg_ref_xy(i, q["shaft"])
                if float(torch.linalg.vector_norm(q["bore"][:2] - ref)) < XY_TOL:
                    if cyc:
                        st["peg_phase"], st["hover_idx"], st["hover_end"] = "cycle", 0, t + cyc[0]["steps"]
                    else:
                        st["peg_phase"] = "clean"
            elif st["peg_phase"] == "cycle" and t >= st["hover_end"]:
                st["hover_idx"] += 1
                if st["hover_idx"] < len(cyc):
                    st["hover_end"] = t + cyc[st["hover_idx"]]["steps"]
                else:
                    st["peg_phase"] = "clean"
            st["sub"] = (f"hover{st['hover_idx']}" if st["peg_phase"] == "cycle" else
                         "PEG_CLEAN" if st["peg_phase"] == "clean" else "transit")
        else:
            st["sub"] = state

    def _enter(self, i: int, new_state: str, q: dict) -> str:
        st, L = self.nm[i], self.latents[i]
        st["transitions"].append((st["t"], self.fsm_state[i], new_state))
        st["entry_t"] = st["t"]
        st["vt_pos"] = None
        if new_state == "MOVE_TO_GEAR_Z":
            st["desc_idx"] = 0
            cyc = L["descent_cycles"]
            st["desc_end"] = st["t"] + (cyc[0]["steps"] if cyc else 0)
        elif new_state == "RAISE_GEAR_Z":
            st["raise_xy"] = (q["eef"][:2] + torch.tensor(L["raise_off"], dtype=q["eef"].dtype,
                                                           device=q["eef"].device)).clone()
        elif new_state == "MOVE_TO_PEG_XY":
            st["peg_phase"] = "transit"
        return new_state

    # ------------------------------------------------------------------- FSM
    def _exit_condition(self, i: int, state: str, q: dict) -> str:
        """Next state if the *current* state's exit condition holds, else ``state``."""
        st, L = self.nm[i], self.latents[i]
        eef = q["eef"]
        if state == "MOVE_TO_GEAR_XY":
            xy_ok = float(torch.linalg.vector_norm(eef[:2] - self._hover_xy(i, q["bore"]))) < XY_TOL
            z_ok = abs(float(eef[2]) - L["z_pregrasp"]) < _NM_PREGRASP_Z_TOL
            gq = self._grasp_quat(i, eef.dtype, eef.device)
            ori_err = float(math_utils.quat_error_magnitude(q["eef_quat"].unsqueeze(0), gq.unsqueeze(0))[0])
            if xy_ok and z_ok and ori_err < _NM_PREGRASP_ORI_TOL:
                return "MOVE_TO_GEAR_Z"
        elif state == "MOVE_TO_GEAR_Z":
            cycles_done = st["desc_idx"] >= len(L["descent_cycles"])
            xy_ok = float(torch.linalg.vector_norm(eef[:2] - self._grasp_point_xy(i, q["bore"]))) < _NM_GRASP_XY_TOL
            if cycles_done and xy_ok and abs(float(eef[2]) - Z_GRASP) < Z_TOL:
                return "GRASP_GEAR"
        elif state == "GRASP_GEAR":
            lo, hi = _NM_HOLD_OPENING_RANGE
            dz = float(eef[2] - q["bore"][2])
            near = (float(torch.linalg.vector_norm(eef[:2] - q["bore"][:2])) < _NM_HOLD_GEAR_XY
                    and _NM_HOLD_GEAR_DZ[0] <= dz <= _NM_HOLD_GEAR_DZ[1])
            if lo <= q["opening"] <= hi and near:
                return "RAISE_GEAR_Z"
        elif state == "RAISE_GEAR_Z":
            if float(eef[2]) >= L["z_transport"] - Z_TOL:
                return "MOVE_TO_PEG_XY"
        elif state == "MOVE_TO_PEG_XY":
            if (st["peg_phase"] == "clean" and q["bs_xy"] < _NM_CLEAN_XY_TOL
                    and abs(q["yaw_err_deg"]) < _NM_CLEAN_YAW_TOL_DEG):
                return "INSERT"
        elif state == "INSERT":
            if float(eef[2]) <= Z_INSERT_DEPTH + Z_TOL:
                return "RELEASE"
        elif state == "RELEASE":
            if q["opening"] >= _NM_RELEASED_OPENING:
                return "RETREAT_UP"
        return state

    def compute_FSM_state(self, obs: dict, env_idx: int) -> str:
        i = env_idx
        st = self.nm[i]
        st["t"] += 1
        cur = self.fsm_state[i]
        if st["pause_remaining"] > 0:
            st["sub"] = "pause"
            return cur
        q = self._quantities(obs, i)
        # Deferred transition: a pause just completed — fire the pending state change now.
        if st["pending"] is not None:
            nxt, st["pending"] = st["pending"], None
            nxt = self._enter(i, nxt, q)
            self._update_subphase(i, nxt, q)
            return nxt
        self._update_subphase(i, cur, q)
        nxt = self._exit_condition(i, cur, q)
        if nxt == cur:
            return cur
        key = f"{cur}->{nxt}"
        if _NM_STICKY and (cur, nxt) in _NM_STICKY_TRANSITIONS and self._sticky_delay(i, key):
            st["sub"] = f"sticky:{cur}"
            return cur  # linger in the current state
        if _NM_PAUSES and (cur, nxt) in _NM_PAUSE_TRANSITIONS and self.latents[i]["pause_steps"][key] > 0:
            st["pending"] = nxt
            st["pause_remaining"] = self.latents[i]["pause_steps"][key]
            st["pause_gripper"] = GRIPPER_CLOSE_CMD if cur in ("RAISE_GEAR_Z",) else GRIPPER_OPEN_CMD
            return cur
        nxt = self._enter(i, nxt, q)
        self._update_subphase(i, nxt, q)
        return nxt

    # ---------------------------------------------------------------- action
    def compute_action(self, obs: dict, env_idx: int) -> torch.Tensor:
        i = env_idx
        pol = obs["policy"]
        dev, dt = pol["eef_pos"].device, pol["eef_pos"].dtype
        # Probe freeze-hold (Phase 1 tooling) and NM pause: zero motion, fixed gripper.
        if self.hold_remaining[i] > 0:
            self.hold_remaining[i] -= 1
            return torch.tensor([0, 0, 0, 0, 0, 0, self.hold_gripper[i]], dtype=dt, device=dev)
        st, L = self.nm[i], self.latents[i]
        if st["pause_remaining"] > 0:
            st["pause_remaining"] -= 1
            return torch.tensor([0, 0, 0, 0, 0, 0, st["pause_gripper"]], dtype=dt, device=dev)

        state = self.fsm_state[i]
        q = self._quantities(obs, i)
        eef, bore, shaft = q["eef"], q["bore"], q["shaft"]
        target_pos = eef.clone()
        target_quat: torch.Tensor | None = None
        gripper = GRIPPER_OPEN_CMD
        zero_noise = False

        if state == "MOVE_TO_GEAR_XY":
            target_pos[:2] = self._hover_xy(i, bore)
            target_pos[2] = L["z_pregrasp"]
            target_quat = self._grasp_quat(i, dt, dev)
        elif state == "MOVE_TO_GEAR_Z":
            cyc = L["descent_cycles"]
            if st["desc_idx"] < len(cyc):
                c = cyc[st["desc_idx"]]
                target_pos[:2] = self._grasp_point_xy(i, bore, c["slide"], c["close"])
                target_pos[2] = Z_GRASP + c["dz"]
            else:
                target_pos[:2] = self._grasp_point_xy(i, bore)
                target_pos[2] = Z_GRASP
            target_quat = self._grasp_quat(i, dt, dev)
        elif state == "GRASP_GEAR":
            gripper = GRIPPER_CLOSE_CMD  # hold pose; close
        elif state == "RAISE_GEAR_Z":
            if st["raise_xy"] is not None:
                target_pos[:2] = st["raise_xy"]
            target_pos[2] = L["z_transport"]
            gripper = GRIPPER_CLOSE_CMD
        elif state == "MOVE_TO_PEG_XY":
            ref_xy, yaw_off = self._peg_ref_xy(i, shaft)
            target_pos[:2] = eef[:2] + (ref_xy - bore[:2])
            target_pos[2] = L["z_transport"]
            target_quat = self._mesh_yaw_target_quat(i, q["gear_quat"], dt, dev)
            if yaw_off != 0.0:
                zq = math_utils.quat_from_angle_axis(
                    torch.tensor([yaw_off], dtype=dt, device=dev), torch.tensor([[0.0, 0.0, 1.0]], dtype=dt, device=dev)
                )[0]
                target_quat = math_utils.quat_mul(zq.unsqueeze(0), target_quat.unsqueeze(0))[0]
            gripper = GRIPPER_CLOSE_CMD
            zero_noise = st["peg_phase"] == "clean"
        elif state == "INSERT":
            target_pos[:2] = eef[:2] + (shaft[:2] - bore[:2])
            target_pos[2] = Z_INSERT_DEPTH
            target_quat = self._mesh_yaw_target_quat(i, q["gear_quat"], dt, dev)
            gripper = GRIPPER_CLOSE_CMD
            zero_noise = True
        elif state == "RELEASE":
            gripper = GRIPPER_OPEN_CMD
        elif state == "RETREAT_UP":
            target_pos[2] = L["z_transport"]
        else:
            raise ValueError(f"Unknown FSM state: {state!r}")

        # ── Virtual-target walk (off by default) ─────────────────────────────
        if _NM_VIRTUAL_TARGET and not zero_noise:
            goal = target_pos.detach().cpu().numpy()
            if st["vt_pos"] is None or st["vt_state"] != (state, st["sub"]):
                st["vt_pos"] = eef.detach().cpu().numpy().copy()
                st["vt_vel"] = np.zeros(3)
                st["vt_falloff"] = max(_NM_VT_FALLOFF_FRAC_POS * float(np.linalg.norm(goal - st["vt_pos"])), 1e-8)
                st["vt_state"] = (state, st["sub"])
            to_goal = goal - st["vt_pos"]
            dist = float(np.linalg.norm(to_goal))
            scale = np.clip(dist / st["vt_falloff"], 0.0, 1.0)
            vel = (_NM_VT_ALPHA_POS * st["vt_vel"] + _NM_VT_K_POS * to_goal
                   + _NM_VT_SIGMA_POS * scale * self.rng.normal(size=3) * np.array([1.0, 1.0, 0.35]))
            if dist > 1e-5:  # project out the backward component so the VT always converges
                back = float(np.dot(vel, to_goal))
                if back < 0:
                    vel -= back / dist**2 * to_goal
            st["vt_pos"], st["vt_vel"] = st["vt_pos"] + vel, vel
            target_pos = torch.tensor(st["vt_pos"], dtype=dt, device=dev)

        noise_on = not zero_noise
        if noise_on:
            target_pos = target_pos + torch.randn(3, dtype=dt, device=dev) * torch.tensor(
                [NOISE_STD_XY, NOISE_STD_XY, NOISE_STD_Z], dtype=dt, device=dev
            )
        delta_pos = _clamp_magnitude(target_pos - eef, MIN_DELTA_POS, MAX_DELTA_POS)

        if target_quat is None:
            delta_rot = torch.zeros(3, dtype=dt, device=dev)
        else:
            if noise_on:
                rot_noise = torch.randn(3, dtype=dt, device=dev) * NOISE_STD_ROT
                noise_quat = math_utils.quat_from_angle_axis(
                    torch.linalg.vector_norm(rot_noise).unsqueeze(0), rot_noise.unsqueeze(0)
                )[0]
                target_quat = math_utils.quat_mul(noise_quat.unsqueeze(0), target_quat.unsqueeze(0))[0]
            delta_quat = math_utils.quat_mul(
                target_quat.unsqueeze(0), math_utils.quat_conjugate(q["eef_quat"].unsqueeze(0))
            )[0]
            delta_rot = math_utils.axis_angle_from_quat(delta_quat.unsqueeze(0))[0]
            delta_rot = _clamp_magnitude(delta_rot, MIN_DELTA_ROT, MAX_DELTA_ROT)

        # ── Correlated OU action noise (off by default) ──────────────────────
        if _NM_CORR_NOISE and noise_on:
            a = _NM_CORR_NOISE_ALPHA
            s = math.sqrt(1.0 - a * a)
            st["corr_pos"] = a * st["corr_pos"] + s * self.rng.normal(0.0, _NM_CORR_NOISE_STD_POS, 3)
            st["corr_rot"] = a * st["corr_rot"] + s * self.rng.normal(0.0, _NM_CORR_NOISE_STD_ROT, 3)
            delta_pos = delta_pos + torch.tensor(st["corr_pos"], dtype=dt, device=dev)
            delta_rot = delta_rot + torch.tensor(st["corr_rot"], dtype=dt, device=dev)

        if noise_on:
            delta_pos = delta_pos + torch.randn(3, dtype=dt, device=dev) * ACTION_NOISE_STD_POS
            delta_rot = delta_rot + torch.randn(3, dtype=dt, device=dev) * ACTION_NOISE_STD_ROT

        return torch.cat([delta_pos, delta_rot, torch.tensor([gripper], dtype=dt, device=dev)], dim=0)

    # -------------------------------------------------------------- logging
    def nm_debug(self, env_idx: int) -> tuple[str, int]:
        st = self.nm[env_idx]
        sticky_left = max(st["sticky"].values()) if st["sticky"] else -1
        return st["sub"], sticky_left

    def episode_record(self, env_idx: int) -> dict:
        st = self.nm[env_idx]
        return {
            "latents": self.latents[env_idx],
            "sticky_fired": st["sticky_fired"],
            "transitions": st["transitions"],
            "final_state": self.fsm_state[env_idx],
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _make_video_frame(env_obs, env_idx: int, video_cam_keys: list) -> np.ndarray:
    """Build a side-by-side video frame for env_idx by concatenating cameras
    at source resolution. All cameras must share the same H.
    """
    pol = env_obs["policy"] if isinstance(env_obs, dict) and "policy" in env_obs else env_obs
    panes = []
    for k in video_cam_keys:
        if k not in pol:
            continue
        img = pol[k][env_idx].detach().cpu().numpy().astype(np.uint8)
        if img.shape[-1] == 4:
            img = img[..., :3]
        panes.append(img)
    if not panes:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    h = panes[0].shape[0]
    panes = [p if p.shape[0] == h else cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panes]
    return np.concatenate(panes, axis=1)


def _write_mp4(frames, path, fps=10):
    """Write a list of (H, W, 3) uint8 frames to MP4."""
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)


def _count_existing_demos(path: str) -> int:
    """Number of demos already in ``path`` (or 0 if file absent / unreadable)."""
    if not os.path.isfile(path):
        return 0
    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                return 0
            return sum(1 for k in f["data"].keys() if k.startswith("demo_"))
    except Exception as e:
        print(f"[scripted_expert] Could not read existing demo count from '{path}': {e}")
        return 0


def _next_demo_index(path: str) -> int:
    """Index the recorder will give the next exported demo (max existing index + 1)."""
    if not os.path.isfile(path):
        return 0
    try:
        with h5py.File(path, "r") as f:
            idx = [int(k.split("_")[1]) for k in f.get("data", {}).keys() if k.startswith("demo_")]
            return max(idx) + 1 if idx else 0
    except Exception:
        return 0


def _attach_nm_latents(path: str, records: list) -> None:
    """Store each successful NM episode's hidden variables as a JSON attr on its demo group.

    Guarded by a length check (demo num_samples vs. episode length) so a mapping
    mismatch never mislabels a demo; mismatches are reported and skipped (the
    sidecar JSONL always has everything).
    """
    if not os.path.isfile(path):
        return
    ok = bad = 0
    with h5py.File(path, "a") as f:
        for rec in records:
            if rec.get("demo") is None:
                continue
            g = f.get(f"data/demo_{rec['demo']}")
            n = int(g.attrs.get("num_samples", -1)) if g is not None else -1
            if g is None or abs(n - rec["length"]) > 1:
                bad += 1
                continue
            g.attrs["nm_latents"] = json.dumps(rec)
            ok += 1
    print(f"[scripted_expert] NM latents attached to {ok} demo(s); {bad} skipped (length mismatch).")


def main():
    # --- Output dir ----------------------------------------------------------
    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # --- Env config ----------------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # The FSM accesses obs by name, so we need per-key obs (not concatenated).
    env_cfg.observations.policy.concatenate_terms = False
    if hasattr(env_cfg.observations, "subtask_terms"):
        env_cfg.observations.subtask_terms.concatenate_terms = False

    # Map --max_steps_per_demo onto the env's time-out (env steps = sim steps /
    # decimation). This forces stuck rollouts to recycle and not block the
    # batch.
    sim_dt = env_cfg.sim.dt
    decimation = env_cfg.decimation
    env_cfg.episode_length_s = float(args_cli.max_steps_per_demo * decimation * sim_dt)

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # Recorder: write *successful* demos only (env auto-flushes on terminate).
    if args_cli.no_record:
        # No recorder terms at all: otherwise every step's obs (incl. camera images)
        # is buffered on the GPU even when nothing is exported.
        env_cfg.recorders = RecorderManagerBaseCfg()
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_filename
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # --- Env + expert --------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = torch.device(args_cli.device)
    if args_cli.non_markovian:
        nm_seed = args_cli.nm_seed
        if nm_seed is None and args_cli.seed is not None:
            nm_seed = args_cli.seed + 7919  # reproducible hidden variables for a seeded run
        expert = GearAssemblyNMScriptedExpert(num_envs=env.num_envs, env=env, nm_seed=nm_seed)
        print(
            f"[scripted_expert] NON-MARKOVIAN expert (nm_seed={nm_seed}): latent_plan={_NM_LATENT_PLAN} "
            f"sticky={_NM_STICKY} align_cycles={_NM_ALIGN_CYCLES} pauses={_NM_PAUSES} "
            f"virtual_target={_NM_VIRTUAL_TARGET} corr_noise={_NM_CORR_NOISE}"
        )
    else:
        expert = GearAssemblyScriptedExpert(num_envs=env.num_envs, env=env)

    existing_demos = _count_existing_demos(args_cli.dataset_file)
    print(f"[scripted_expert] {existing_demos} existing demos in {args_cli.dataset_file}.")
    if args_cli.num_demos > 0:
        print(f"[scripted_expert] target = {args_cli.num_demos} total demos.")
    print(
        f"[scripted_expert] num_envs={env.num_envs}, "
        f"max_steps_per_demo={args_cli.max_steps_per_demo}, "
        f"headless={args_cli.headless}"
    )

    # --- Rate limit (GUI ~30 Hz so the human can follow; headless: full speed)
    rate_limit = (not args_cli.headless) or (args_cli.step_hz is not None)
    target_hz = args_cli.step_hz if args_cli.step_hz is not None else 30
    last_step_time = time.time()

    # --- Driver loop ---------------------------------------------------------
    obs, _ = env.reset()
    env.recorder_manager.reset()
    last_exported = env.recorder_manager.exported_successful_episode_count
    last_logged_state: list[str | None] = [None] * env.num_envs
    episode_steps = [0] * env.num_envs
    # Running counters for the per-episode success-rate print. ``total_episodes``
    # increments every time an env terminates *or* truncates (i.e. every demo
    # attempt, regardless of outcome). ``total_successes`` counts only the
    # ones that fired the env's success termination.
    total_episodes = 0
    total_successes = 0

    # Per-env completed-episode counters (for --episodes_per_env) and optional FSM logs.
    completed_per_env = [0] * env.num_envs
    ep_start_info: list[dict | None] = [None] * env.num_envs
    steps_f = episodes_f = None
    if args_cli.fsm_log:
        os.makedirs(args_cli.fsm_log, exist_ok=True)
        steps_f = open(os.path.join(args_cli.fsm_log, "steps.csv"), "w")
        steps_f.write(
            "env,ep,step,state,hold,eef_x,eef_y,eef_z,eef_grasp_xy,bore_shaft_xy,bore_z_minus_shaft,"
            "gear_z,f1,f2,grasped,yaw_err_deg,nm_sub,nm_sticky\n"
        )
        episodes_f = open(os.path.join(args_cli.fsm_log, "episodes.csv"), "w")
        episodes_f.write("env,ep,counted,gear_x,gear_y,gear_yaw_deg,outcome,length\n")

    # NM hidden variables: one JSON line per finished episode (sidecar to the HDF5).
    latents_f = None
    nm_records: list[dict] = []
    next_demo_idx = _next_demo_index(args_cli.dataset_file)
    if args_cli.non_markovian:
        lat_dir = args_cli.fsm_log if (args_cli.no_record and args_cli.fsm_log) else output_dir
        latents_path = os.path.join(lat_dir, f"{output_filename}_nm_latents.jsonl")
        latents_f = open(latents_path, "a")
        print(f"[scripted_expert] NM hidden variables -> {latents_path}")

    def _log_step(i: int, ep: int, step: int) -> None:
        pol = obs["policy"]
        eef = pol["eef_pos"][i]
        gp, gq, sp, fp = pol["gear_pos"][i], pol["gear_quat"][i], pol["shaft_pos"][i], pol["gripper_pos"][i]
        bore = _bore_world(gp, gq)
        g_xy = float(torch.linalg.vector_norm(eef[:2] - expert._grasp_ref_xy(bore)))
        bs_xy = float(torch.linalg.vector_norm(bore[:2] - sp[:2]))
        grasped = abs(float(fp[0]) - float(fp[1])) < GRIPPER_CLOSED_THRESHOLD
        nm_sub, nm_sticky = expert.nm_debug(i) if hasattr(expert, "nm_debug") else ("", "")
        steps_f.write(
            f"{i},{ep},{step},{expert.fsm_state[i]},{int(expert.hold_remaining[i] > 0)},"
            f"{float(eef[0]):.5f},{float(eef[1]):.5f},{float(eef[2]):.5f},{g_xy:.5f},{bs_xy:.5f},"
            f"{float(bore[2] - sp[2]):.5f},{float(gp[2]):.5f},{float(fp[0]):.5f},{float(fp[1]):.5f},"
            f"{int(grasped)},{expert.mesh_yaw_error_deg(i, gq):.3f},{nm_sub},{nm_sticky}\n"
        )

    # Video recording setup
    videos_dir = os.path.join(output_dir, "scripted_videos")
    videos_started = 0
    record_env = [False] * env.num_envs
    frame_buffers = [[] for _ in range(env.num_envs)]
    video_cam_keys = []
    if args_cli.n_video_trials > 0:
        os.makedirs(videos_dir, exist_ok=True)
        # Find camera keys in the policy obs dict
        pol = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
        video_cam_keys = sorted([k for k in pol.keys() if "cam" in k])
        if not video_cam_keys:
            print("[scripted_expert] Warning: --n-video-trials > 0 but no camera obs keys detected.")
        
        for i in range(env.num_envs):
            if videos_started < args_cli.n_video_trials:
                record_env[i] = True
                videos_started += 1
                frame_buffers[i].append(_make_video_frame(obs, i, video_cam_keys))

    total_episodes_started = 0
    stop_requested = False
    while simulation_app.is_running() and not stop_requested:
        with torch.inference_mode():
            # 1) Update FSM state per env from the current obs.
            for i in range(env.num_envs):
                if episode_steps[i] == 0:
                    gp, gq = obs["policy"]["gear_pos"][i], obs["policy"]["gear_quat"][i]
                    ep_start_info[i] = {
                        "ep": total_episodes_started,
                        "x": float(gp[0]),
                        "y": float(gp[1]),
                        "yaw": math.degrees(_yaw_from_quat(gq)),
                    }
                    total_episodes_started += 1
                prev_state = expert.fsm_state[i]
                expert.fsm_state[i] = expert.compute_FSM_state(obs, i)
                expert.update_holds(prev_state, expert.fsm_state[i], obs, i)
                if steps_f is not None:
                    _log_step(i, ep_start_info[i]["ep"], episode_steps[i])

            # 2) Compute per-env action, then stack into a (num_envs, 7) tensor.
            # The IK-rel arm action is 6D + 1D gripper = 7D total.
            actions = torch.stack([expert.compute_action(obs, i) for i in range(env.num_envs)], dim=0)

        # GUI-mode FSM trace: print only on state transitions to keep it readable.
        if not args_cli.headless and env.num_envs == 1:
            if expert.fsm_state[0] != last_logged_state[0]:
                print(f"[fsm] -> {expert.fsm_state[0]}")
                last_logged_state[0] = expert.fsm_state[0]

        obs, _, terminated, truncated, _ = env.step(actions)

        if args_cli.n_video_trials > 0:
            for i in range(env.num_envs):
                if record_env[i]:
                    frame_buffers[i].append(_make_video_frame(obs, i, video_cam_keys))

        # Count the step that just occurred for every env.
        for i in range(env.num_envs):
            episode_steps[i] += 1

        # Per-env reset bookkeeping: the env auto-resets envs whose terminated /
        # truncated fired, so we just reset the FSM string for those envs.
        # (Done before the --num_demos exit check so the final episode's video
        # and summary line are not lost.)
        done = (terminated | truncated).cpu()
        for i in range(env.num_envs):
            if bool(done[i]):
                is_success = bool(terminated[i])
                outcome = "SUCCESS" if is_success else "TIMEOUT"
                counted = args_cli.episodes_per_env <= 0 or completed_per_env[i] < args_cli.episodes_per_env
                completed_per_env[i] += 1
                if counted:
                    total_episodes += 1
                    if is_success:
                        total_successes += 1
                rate = total_successes / max(total_episodes, 1)
                print(
                    f"[scripted_expert] env {i} episode ended ({outcome}) after"
                    f" {episode_steps[i]} steps. Success rate so far:"
                    f" {total_successes}/{total_episodes} ({rate:.1%})."
                )
                if episodes_f is not None and ep_start_info[i] is not None:
                    s = ep_start_info[i]
                    episodes_f.write(
                        f"{i},{s['ep']},{int(counted)},{s['x']:.5f},{s['y']:.5f},{s['yaw']:.2f},"
                        f"{outcome},{episode_steps[i]}\n"
                    )
                    episodes_f.flush()
                    steps_f.flush()
                if latents_f is not None:
                    rec = expert.episode_record(i)
                    rec.update(
                        env=i,
                        ep=ep_start_info[i]["ep"] if ep_start_info[i] is not None else None,
                        outcome=outcome,
                        length=episode_steps[i],
                        demo=None,
                    )
                    if is_success and not args_cli.no_record:
                        # The recorder exports successes of one step in ascending env order,
                        # naming them demo_{n} sequentially.
                        rec["demo"] = next_demo_idx
                        next_demo_idx += 1
                    latents_f.write(json.dumps(rec) + "\n")
                    latents_f.flush()
                    nm_records.append(rec)

                if args_cli.n_video_trials > 0 and record_env[i]:
                    ep_id = ep_start_info[i]["ep"] if ep_start_info[i] is not None else total_episodes
                    video_path = os.path.join(videos_dir, f"trial_{ep_id:04d}_env{i}_{outcome}.mp4")
                    _write_mp4(frame_buffers[i], video_path, fps=10)
                    print(f"[scripted_expert] Saved video: {video_path}")
                    frame_buffers[i] = []
                    if videos_started < args_cli.n_video_trials:
                        record_env[i] = True
                        videos_started += 1
                        frame_buffers[i].append(_make_video_frame(obs, i, video_cam_keys))
                    else:
                        record_env[i] = False

                episode_steps[i] = 0
                expert.reset_env(i)
                last_logged_state[i] = None

        # Track recorder progress.
        current_exported = env.recorder_manager.exported_successful_episode_count
        if current_exported > last_exported:
            new = current_exported - last_exported
            last_exported = current_exported
            total = existing_demos + current_exported
            tgt_str = f"/{args_cli.num_demos}" if args_cli.num_demos > 0 else ""
            print(f"[scripted_expert] +{new} successful demo(s)  →  total: {total}{tgt_str}")
            if args_cli.num_demos > 0 and total >= args_cli.num_demos:
                print("[scripted_expert] target reached, exiting.")
                stop_requested = True

        if args_cli.episodes_per_env > 0 and min(completed_per_env) >= args_cli.episodes_per_env:
            print(f"[scripted_expert] every env completed {args_cli.episodes_per_env} episode(s), exiting.")
            stop_requested = True

        if rate_limit:
            elapsed = time.time() - last_step_time
            sleep = max(0.0, 1.0 / target_hz - elapsed)
            if sleep > 0.0:
                time.sleep(sleep)
            last_step_time = time.time()

    final_total = existing_demos + env.recorder_manager.exported_successful_episode_count
    print(f"[scripted_expert] done. Final demo count in {args_cli.dataset_file}: {final_total}.")
    print(
        f"[scripted_expert] SUMMARY counted_episodes={total_episodes} successes={total_successes} "
        f"rate={total_successes / max(total_episodes, 1):.3f}"
    )
    if steps_f is not None:
        steps_f.close()
        episodes_f.close()
    if latents_f is not None:
        latents_f.close()

    env.close()
    if args_cli.non_markovian and not args_cli.no_record:
        _attach_nm_latents(args_cli.dataset_file, nm_records)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # An uncaught exception otherwise leaves the Kit app running (hung) forever.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
