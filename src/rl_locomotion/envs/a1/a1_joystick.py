""" Menagerie's A1 and Go1 share the kinematic tree and every name Playground's Go1 code refers to. Only the
numbers differ, so the reward terms, observation code, termination and command sampling of `go1.joystick.Joystick`
apply verbatim, and only the model construction is overridden.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from etils import epath
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants as go1_consts
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

XML_DIR = epath.Path(__file__).parent / "xmls"
TASK_TO_XML = {
    "flat_terrain": XML_DIR / "a1_scene_flat_terrain.xml",
    "rough_terrain": XML_DIR / "a1_scene_rough_terrain.xml",
}


def get_assets() -> Dict[str, bytes]:
    """Everything the A1 scenes reference.
    It builds a dictionary mapping each filename to its bytes
    (your scene XMLs, the A1 meshes from Menagerie and Go1's 
    rough-terrain heightfield), so MuJoCo can load the A1 scene 
    from an XML string without reading those files from disk.

    """
    assets: Dict[str, bytes] = {}
    mjx_env.update_assets(assets, XML_DIR, "*.xml")
    mjx_env.update_assets(assets, mjx_env.MENAGERIE_PATH / "unitree_a1" / "assets")
    mjx_env.update_assets(assets, go1_consts.ROOT_PATH / "xmls" / "assets")
    return assets


class A1Joystick(go1_joystick.Joystick):
    """Go1 joystick task on the Unitree A1 model."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = go1_joystick.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if task not in TASK_TO_XML:
            raise ValueError(f"unknown task {task!r}; choose from {list(TASK_TO_XML)}")
        if task.startswith("rough"):
            # Same contact budget Playground gives Go1 on the heightfield.
            config.naconmax = max(config.naconmax, 8 * 8192)
            config.njmax = max(config.njmax, 12 + 48)


        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        xml_path = TASK_TO_XML[task]
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=self._model_assets)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 20

        self._mj_model.dof_damping[6:] = self._config.Kd
        self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path.as_posix()
        self._imu_site_id = self._mj_model.site("imu").id
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id for geom in go1_consts.FEET_GEOMS
        ]

        self._post_init()
