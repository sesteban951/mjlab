"""Each gait-library task's mdp namespace resolves its own twist reward."""

from mjlab.tasks.crawling_common import mdp as crawl_mdp
from mjlab.tasks.crawling_common.mdp import rewards as crawl_rewards
from mjlab.tasks.crawling_diffdrive import mdp as diffdrive_mdp
from mjlab.tasks.gait_library import mdp as library_mdp
from mjlab.tasks.jog_unicycle import mdp as jog_mdp
from mjlab.tasks.jog_unicycle.mdp import rewards as jog_rewards
from mjlab.tasks.walk_unicycle import mdp as walk_mdp


def test_twist_reward_is_posture_specific() -> None:
  # The shared namespace has none, so no import order can pick the wrong one.
  assert not hasattr(library_mdp, "twist_tracking")
  assert crawl_mdp.twist_tracking is crawl_rewards.twist_tracking
  assert diffdrive_mdp.twist_tracking is crawl_rewards.twist_tracking
  assert jog_mdp.twist_tracking is jog_rewards.twist_tracking
  assert walk_mdp.twist_tracking is jog_rewards.twist_tracking


def test_shared_terms_are_the_library_ones() -> None:
  for mdp in (crawl_mdp, diffdrive_mdp, jog_mdp, walk_mdp):
    for name in (
      "LibraryMotionCommandCfg",
      "BlendingMotionCommandCfg",
      "commanded_twist",
      "motion_phase",
      "egocentric_anchor_position_error_exp",
      "egocentric_anchor_orientation_error_exp",
    ):
      assert getattr(mdp, name) is getattr(library_mdp, name), name
