"""Shared domain-randomization base for the custom Unitree Go2 environments.

The Go2 counterpart of ``unitree_g1.custom_dr``: one tunable set of terms that a
custom Go2 env layers on top of its base config by passing a ``CustomGo2DRCfg``.

- Startup terms (kept in play mode, as in mjlab): base COM offset, encoder bias,
  foot friction, joint armature, base mass/inertia and, if set, PD gains.
- Interval pushes, scaled down from the G1 tracking pushes for the Go2's size.
- Command delay (training only), via :func:`add_custom_go2_actuator_delay`.

Gains and COM follow the task's reference implementation. The defaults are
SuperMimic's Go2 tracking values (+-1 cm COM, no PD-gain randomization); a Go2
velocity env should pass unitree_rl_mjlab's +-5 cm COM, which also leaves the
gains alone.
"""

import math
from dataclasses import dataclass, field, replace

from mjlab.asset_zoo.robots.unitree_g1.custom_dr import randomize_actuator_delay
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

GO2_BASE_BODY = "base_link"


@dataclass
class CustomGo2DRCfg:
  """Tunable ranges for the shared custom Go2 domain randomization."""

  # Base-COM offset (m), per axis {0: x, 1: y, 2: z}. Added to the nominal COM.
  base_com: dict[int, tuple[float, float]] = field(
    default_factory=lambda: {
      0: (-0.01, 0.01),
      1: (-0.01, 0.01),
      2: (-0.01, 0.01),
    }
  )
  # Per-joint encoder offset (rad).
  encoder_bias: tuple[float, float] = (-0.01, 0.01)
  # Foot friction coefficient (absolute), shared across foot geoms.
  foot_friction: tuple[float, float] = (0.3, 1.2)
  # PD gain scale factors (multiplicative). None leaves the gains nominal, as in
  # SuperMimic's Go2 tracking and unitree_rl_mjlab's Go2 velocity task.
  pd_kp: tuple[float, float] | None = None
  pd_kd: tuple[float, float] | None = None
  # Joint armature scale factor (multiplicative).
  joint_armature: tuple[float, float] = (0.95, 1.05)
  # Base (base_link) mass scale factor (multiplicative). Applied via
  # pseudo_inertia so mass and inertia scale together (a density change).
  base_mass: tuple[float, float] = (0.95, 1.05)
  # Push interval (s) and the velocity each push sets on the base. These are the
  # G1 tracking pushes (+-0.5 m/s xy, +-0.2 m/s z, +-0.52/0.52/0.78 rad/s) scaled
  # by 0.6, the Froude ratio sqrt(0.3 m / 0.8 m) of Go2 to G1 base height, so a
  # push is a similar fraction of each robot's natural speed. The angular part is
  # scaled the same way to stay conservative.
  push_interval_s: tuple[float, float] = (1.0, 3.0)
  push_velocity: dict[str, tuple[float, float]] = field(
    default_factory=lambda: {
      "x": (-0.3, 0.3),
      "y": (-0.3, 0.3),
      "z": (-0.12, 0.12),
      "roll": (-0.3, 0.3),
      "pitch": (-0.3, 0.3),
      "yaw": (-0.45, 0.45),
    }
  )


def add_custom_go2_dr(
  cfg: ManagerBasedRlEnvCfg,
  dr_cfg: CustomGo2DRCfg | None = None,
  push: bool = True,
) -> None:
  """Apply the shared custom Go2 DR base in place.

  Retunes the inherited ``base_com``, ``encoder_bias`` and ``foot_friction``
  startup terms, adds ``joint_armature``, ``base_mass`` and (when ``pd_kp`` or
  ``pd_kd`` is set) ``pd_gains``, and sets ``push_robot`` (or removes it when ``push`` is False, e.g. in play mode).

  Args:
    cfg: An already-built env config whose ``events`` already contain
      ``base_com``, ``encoder_bias`` and ``foot_friction`` targeting the Go2.
    dr_cfg: Tunable ranges. Defaults to :class:`CustomGo2DRCfg`.
    push: Whether to add the interval push term.
  """
  if dr_cfg is None:
    dr_cfg = CustomGo2DRCfg()

  # Retune inherited startup terms (keeps the base's per-robot targeting).
  cfg.events["base_com"].params["asset_cfg"].body_names = (GO2_BASE_BODY,)
  cfg.events["base_com"].params["ranges"] = dict(dr_cfg.base_com)
  cfg.events["encoder_bias"].params["bias_range"] = dr_cfg.encoder_bias
  cfg.events["foot_friction"].params["ranges"] = dr_cfg.foot_friction

  # The DC motor PD runs in torch, so pd_gains scales the actuator's own gains.
  if dr_cfg.pd_kp is not None or dr_cfg.pd_kd is not None:
    cfg.events["pd_gains"] = EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
        "kp_range": dr_cfg.pd_kp or (1.0, 1.0),
        "kd_range": dr_cfg.pd_kd or (1.0, 1.0),
        "operation": "scale",
      },
    )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": dr_cfg.joint_armature,
      "operation": "scale",
    },
  )
  # pseudo_inertia scales mass by e^(2*alpha), so a multiplicative mass range
  # (lo, hi) maps to alpha_range = (0.5*ln(lo), 0.5*ln(hi)).
  mass_lo, mass_hi = dr_cfg.base_mass
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(GO2_BASE_BODY,)),
      "alpha_range": (0.5 * math.log(mass_lo), 0.5 * math.log(mass_hi)),
    },
  )

  if push:
    cfg.events["push_robot"] = EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=dr_cfg.push_interval_s,
      params={"velocity_range": dict(dr_cfg.push_velocity)},
    )
  else:
    cfg.events.pop("push_robot", None)


def add_custom_go2_actuator_delay(
  cfg: ManagerBasedRlEnvCfg,
  min_delay_sec: float = 0.0,
  max_delay_sec: float = 0.04,
) -> None:
  """Add a constant per-env command delay to every Go2 actuator.

  Same scheme as ``unitree_g1.custom_dr.add_custom_g1_actuator_delay``: each env
  draws one lag in ``[min, max]`` physics steps at reset and holds it for the
  episode (``delay_hold_prob=1.0`` plus the :func:`randomize_actuator_delay`
  reset event). It applies to any actuator type, including the DC motors.

  Args:
    cfg: An already-built env config with a Go2 ``robot`` entity.
    min_delay_sec: Minimum command delay in seconds.
    max_delay_sec: Maximum command delay in seconds. ``0`` disables delay.
  """
  robot_cfg = cfg.scene.entities["robot"]
  assert isinstance(robot_cfg, EntityCfg)
  assert robot_cfg.articulation is not None

  timestep = cfg.sim.mujoco.timestep
  min_lag = round(min_delay_sec / timestep)
  max_lag = round(max_delay_sec / timestep)

  actuators = tuple(
    replace(
      act,
      delay_min_lag=min_lag,
      delay_max_lag=max_lag,
      delay_hold_prob=1.0,  # never re-sample; the reset event sets the lag
      delay_update_period=cfg.decimation,
      delay_per_env_phase=True,
    )
    for act in robot_cfg.articulation.actuators
  )
  cfg.scene.entities["robot"] = replace(
    robot_cfg,
    articulation=replace(robot_cfg.articulation, actuators=actuators),
  )

  if max_lag > 0:
    cfg.events["actuator_delay"] = EventTermCfg(
      mode="reset",
      func=randomize_actuator_delay,
      params={"min_lag": min_lag, "max_lag": max_lag},
    )
