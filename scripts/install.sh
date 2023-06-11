#!/bin/sh

set -e

echo "creating temp directory"
# Change into temporary directory.
cd "$(mktemp -d)"

# Download release archive.

echo "unzipping databricks cli"
# Unzip release archive.
unzip -q -o "scripts/databricks_cli_linux_amd64.zip"

echo "adding databricks to path"
# Add databricks to path.
sudo chmod +x ./databricks
sudo cp ./databricks "$TARGET"
echo "Installed $($TARGET/databricks -v) at $TARGET/databricks."
