#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced: source env.sh" >&2
  exit 1
fi

_env_sh_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_ros_setup="/opt/ros/humble/setup.bash"
_cyclonedds_xml="${_env_sh_dir}/cyclonedds/cyclonedds.xml"

if [[ ! -f "${_ros_setup}" ]]; then
  echo "ROS 2 Humble setup not found at ${_ros_setup}" >&2
  return 1
fi

if [[ ! -f "${_cyclonedds_xml}" ]]; then
  echo "CycloneDDS config not found at ${_cyclonedds_xml}" >&2
  return 1
fi

source "${_ros_setup}"
# Source local ROS2 workspace (g1_onboard_msgs, etc.)
_ros2_ws_setup="${_env_sh_dir}/ros2_ws/install/setup.bash"
if [[ -f "${_ros2_ws_setup}" ]]; then
  source "${_ros2_ws_setup}"
fi
unset _ros2_ws_setup
export CYCLONEDDS_URI="file://${_cyclonedds_xml}"
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}
# NX IP for unicast peer discovery — override via env var if the NX address changes.
export DDS_PEER_IP=${DDS_PEER_IP:-192.168.123.164}
echo "Activated ROS ${ROS_DISTRO:-unknown} with ${RMW_IMPLEMENTATION} (${CYCLONEDDS_URI})"

unset _env_sh_dir
unset _ros_setup
unset _cyclonedds_xml
