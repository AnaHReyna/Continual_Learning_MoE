#!/usr/bin/env bash

# This file is meant to be sourced:
# source scripts/00_setup/env.sh

ARCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "$ARCH_ROOT/../.." && pwd)"

cd "$ARCH_ROOT" || exit 1

TEACHER_ROOT="$PROJECT_ROOT/Teacher"
SCENARIO_RUNNER_ROOT="/home/ana/scenario_runner"
CARLA_PYTHONAPI_ROOT="/home/ana/CARLA_0.9.13/PythonAPI/carla"

export PYTHONPATH="$ARCH_ROOT:$ARCH_ROOT/common:$ARCH_ROOT/stage1:$ARCH_ROOT/stage2:$PROJECT_ROOT:$TEACHER_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_PYTHONAPI_ROOT:${PYTHONPATH:-}"

echo "[OK] ARCH_ROOT=$ARCH_ROOT"
echo "[OK] PROJECT_ROOT=$PROJECT_ROOT"
echo "[OK] TEACHER_ROOT=$TEACHER_ROOT"
echo "[OK] SCENARIO_RUNNER_ROOT=$SCENARIO_RUNNER_ROOT"
echo "[OK] CARLA_PYTHONAPI_ROOT=$CARLA_PYTHONAPI_ROOT"
echo "[OK] PYTHONPATH=$PYTHONPATH"
