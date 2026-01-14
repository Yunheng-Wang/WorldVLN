import habitat_sim
import os 
from magnum import Vector3

def env_config(config, scene_id):
    env_cfg = habitat_sim.SimulatorConfiguration()
    env_cfg.scene_id = os.path.join(config["data_root"], "scene/mp3d", scene_id, scene_id + ".glb")
    env_cfg.enable_physics = False
    env_cfg.allow_sliding = True
    return env_cfg


def agent_config(config):
    # 1. 配置 agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = config["agent"]["height"] 
    agent_cfg.radius = config["agent"]["radius"]    
    agent_cfg.body_type = config["agent"]["body_type"]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=0)
        ),
    }
    # 2. 配置agent 上的传感器
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.position = config["agent"]["rgb_sensor"]["position"]
    rgb_sensor_spec.orientation = Vector3(config["agent"]["rgb_sensor"]["orientation"][0], config["agent"]["rgb_sensor"]["orientation"][1], config["agent"]["rgb_sensor"]["orientation"][2])
    rgb_sensor_spec.resolution = [config["agent"]["rgb_sensor"]["height"], config["agent"]["rgb_sensor"]["width"]]
    rgb_sensor_spec.hfov = config["agent"]["rgb_sensor"]["hfov"]
    # 3. 将传感器加入到agent身上
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
    return agent_cfg


def environment(config, scene_id):
    env_cfg = env_config(config, scene_id)
    agent_cfg = agent_config(config)
    sim_cfg = habitat_sim.Configuration(env_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    return sim, env_cfg, agent_cfg


def env_config_hm3d(config, scene_id):
    env_cfg = habitat_sim.SimulatorConfiguration()
    env_cfg.scene_id = os.path.join(config["data_root"], "scene/hm3d", scene_id, scene_id.split('-')[1] + ".basis.glb")
    env_cfg.enable_physics = False
    env_cfg.allow_sliding = True
    return env_cfg


def environment_hm3d(config, scene_id):
    env_cfg = env_config_hm3d(config, scene_id)
    agent_cfg = agent_config(config)
    sim_cfg = habitat_sim.Configuration(env_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    return sim, env_cfg, agent_cfg