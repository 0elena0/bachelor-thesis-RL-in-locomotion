"""Boston Dynamics Spot joystick task: Playground's own task, plus a rough-terrain scene.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import mujoco
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants as go1_consts
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick
from mujoco_playground._src.locomotion.spot import spot_constants as consts

XML_DIR = epath.Path(__file__).parent / "xmls"
ROUGH_XML = XML_DIR / "spot_scene_rough_terrain.xml"
TASKS = ("flat_terrain", "rough_terrain")

JOINT_NAMES = ("fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
               "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn")


def get_assets() -> Dict[str, bytes]:
    assets = spot_base.get_assets()
    mjx_env.update_assets(assets, XML_DIR, "*.xml")
    mjx_env.update_assets(assets, go1_consts.ROOT_PATH / "xmls" / "assets")
    return assets


class SpotJoystick(spot_joystick.Joystick):
    """Playground's Spot joystick task with a rough_terrain variant."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = spot_joystick.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
        if task == "flat_terrain":
            super().__init__(task=task, config=config, config_overrides=config_overrides)
            return

        config.naconmax = max(config.naconmax, 8 * 8192)
        config.njmax = max(config.njmax, 12 + 48)

        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(ROUGH_XML.read_text(), assets=self._model_assets)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.dof_damping[6:] = self._config.Kd
        self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = ROUGH_XML.as_posix()
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id for geom in consts.FEET_GEOMS
        ]
        self._imu_site_id = self._mj_model.site("imu").id

        self._post_init()
        self._pert_func = (
            self._maybe_apply_perturbation if config.pert_config.enable else lambda state, _: state
        )
