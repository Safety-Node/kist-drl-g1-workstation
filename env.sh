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
export CYCLONEDDS_URI="file://${_cyclonedds_xml}"
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
echo "Activated ROS ${ROS_DISTRO:-unknown} with ${RMW_IMPLEMENTATION} (${CYCLONEDDS_URI})"

unset _env_sh_dir
unset _ros_setup
unset _cyclonedds_xml
