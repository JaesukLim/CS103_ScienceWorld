#!/usr/bin/env bash

set -euo pipefail

pushd "$(dirname "$(readlink -f "$0")")"

sbt assembly
version=$(grep version build.sbt | cut -d '"' -f2)
jar_path="target/scala-2.12/scienceworld-scala-assembly-${version}.jar"

# Keep the Python package default jar path in sync, while also preserving the
# project-specific jar name for anyone referencing it explicitly.
cp -f "$jar_path" ../cs103_scienceworld/scienceworld.jar
cp -f "$jar_path" ../cs103_scienceworld/cs103_scienceworld.jar

popd
