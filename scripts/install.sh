#!/bin/sh

set -e

TARGET="/usr/local/bin"

echo "creating temp directory"
# Change into temporary directory.
tmp_dir="$(mktemp -d)"

# Download release archive.

echo "unzipping databricks cli to tmpo dir"
# Unzip release archive.
unzip -q -o "../scripts/databricks_cli_linux_amd64.zip" -d $tmp_dir

echo "adding databricks to path"
cd $tmp_dir
# Add databricks to path.
sudo chmod +x ./databricks
sudo cp ./databricks "$TARGET"
echo "Installed $($TARGET/databricks -v) at $TARGET/databricks."
