"""MJCF description of the crib: a planar 2-DoF arm on a table with four toys."""

CRIB_XML = r"""
<mujoco model="crib">
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast" iterations="20" ls_iterations="10"/>
  <visual>
    <global offwidth="320" offheight="240"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/>
  </visual>
  <default>
    <geom condim="3" friction="0.5 0.005 0.0001"/>
    <default class="arm">
      <geom contype="2" conaffinity="2" rgba="0.95 0.75 0.6 1" type="capsule" size="0.012"/>
    </default>
    <default class="obj">
      <geom contype="3" conaffinity="3"/>
    </default>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="0.6 0.5 0.01" pos="0 0.25 0" rgba="0.82 0.82 0.78 1" contype="1" conaffinity="1"/>
    <geom name="wall_l" type="box" size="0.01 0.38 0.03" pos="-0.50 0.23 0.03" rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>
    <geom name="wall_r" type="box" size="0.01 0.38 0.03" pos="0.50 0.23 0.03" rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>
    <geom name="wall_f" type="box" size="0.50 0.01 0.03" pos="0 0.60 0.03" rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>
    <geom name="wall_b" type="box" size="0.50 0.01 0.03" pos="0 -0.14 0.03" rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>
    <camera name="eye" pos="0 -0.12 0.62" xyaxes="1 0 0 0 0.8 0.6" fovy="80"/>
    <camera name="overview" pos="0 -0.35 0.75" xyaxes="1 0 0 0 0.85 0.53" fovy="55"/>

    <body name="upper" pos="0 0 0.03">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-0.3 3.44" damping="0.08" armature="0.002"/>
      <geom class="arm" name="upper_g" fromto="0 0 0 0.22 0 0" mass="0.15"/>
      <body name="fore" pos="0.22 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-2.6 2.6" damping="0.05" armature="0.002"/>
        <geom class="arm" name="fore_g" fromto="0 0 0 0.20 0 0" mass="0.1"/>
        <geom class="arm" name="hand_g" type="sphere" size="0.018" pos="0.20 0 0" mass="0.02" rgba="0.95 0.6 0.5 1"/>
        <site name="hand" pos="0.20 0 0" size="0.022" type="sphere" rgba="0 0 0 0"/>
      </body>
    </body>

    <body name="light_box" pos="-0.16 0.17 0.025">
      <freejoint name="light_box"/>
      <geom class="obj" name="light_box_g" type="box" size="0.025 0.025 0.025" mass="0.05" friction="0.35 0.005 0.0001" rgba="0.2 0.5 0.95 1"/>
    </body>
    <body name="ball" pos="0.05 0.27 0.03">
      <freejoint name="ball"/>
      <geom class="obj" name="ball_g" type="sphere" size="0.03" mass="0.06" condim="6" friction="0.6 0.01 0.0015" rgba="0.95 0.3 0.25 1"/>
    </body>
    <body name="heavy_box" pos="0.22 0.20 0.03">
      <freejoint name="heavy_box"/>
      <geom class="obj" name="heavy_box_g" type="box" size="0.03 0.03 0.03" mass="0.7" friction="0.8 0.005 0.0001" rgba="0.35 0.3 0.25 1"/>
    </body>
    <body name="toy_base" pos="-0.10 0.45 0.0">
      <geom class="obj" name="toy_base_g" type="cylinder" size="0.03 0.012" pos="0 0 0.012" rgba="0.3 0.7 0.3 1"/>
      <geom name="toy_light" type="sphere" size="0.028" pos="0.0 0.035 0.03" contype="0" conaffinity="0" rgba="0.25 0.25 0.1 1"/>
      <body name="lever" pos="0 0 0.03">
        <joint name="lever" type="hinge" axis="0 0 1" range="-1.4 1.4" stiffness="0.04" damping="0.004" springref="0"/>
        <geom class="obj" name="lever_g" type="box" size="0.008 0.05 0.02" pos="0 -0.05 0" mass="0.02" rgba="0.3 0.8 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="3.0" kv="0.25" forcerange="-1.2 1.2" ctrlrange="-0.3 3.44"/>
    <position name="a_elbow" joint="elbow" kp="2.0" kv="0.15" forcerange="-0.8 0.8" ctrlrange="-2.6 2.6"/>
  </actuator>
  <sensor>
    <touch name="touch_hand" site="hand"/>
    <jointpos name="lever_angle" joint="lever"/>
    <contact name="arm_light_box" subtree1="upper" body2="light_box" data="found" num="1"/>
    <contact name="arm_ball" subtree1="upper" body2="ball" data="found" num="1"/>
    <contact name="arm_heavy_box" subtree1="upper" body2="heavy_box" data="found" num="1"/>
    <contact name="arm_toy" subtree1="upper" subtree2="toy_base" data="found" num="1"/>
    <contact name="lb_ball" body1="light_box" body2="ball" data="found" num="1"/>
    <contact name="lb_heavy" body1="light_box" body2="heavy_box" data="found" num="1"/>
    <contact name="ball_heavy" body1="ball" body2="heavy_box" data="found" num="1"/>
    <contact name="light_box_wall_l" body1="light_box" geom2="wall_l" data="found" num="1"/>
    <contact name="light_box_wall_r" body1="light_box" geom2="wall_r" data="found" num="1"/>
    <contact name="light_box_wall_f" body1="light_box" geom2="wall_f" data="found" num="1"/>
    <contact name="light_box_wall_b" body1="light_box" geom2="wall_b" data="found" num="1"/>
    <contact name="ball_wall_l" body1="ball" geom2="wall_l" data="found" num="1"/>
    <contact name="ball_wall_r" body1="ball" geom2="wall_r" data="found" num="1"/>
    <contact name="ball_wall_f" body1="ball" geom2="wall_f" data="found" num="1"/>
    <contact name="ball_wall_b" body1="ball" geom2="wall_b" data="found" num="1"/>
    <contact name="heavy_box_wall_l" body1="heavy_box" geom2="wall_l" data="found" num="1"/>
    <contact name="heavy_box_wall_r" body1="heavy_box" geom2="wall_r" data="found" num="1"/>
    <contact name="heavy_box_wall_f" body1="heavy_box" geom2="wall_f" data="found" num="1"/>
    <contact name="heavy_box_wall_b" body1="heavy_box" geom2="wall_b" data="found" num="1"/>
  </sensor>
</mujoco>
"""
