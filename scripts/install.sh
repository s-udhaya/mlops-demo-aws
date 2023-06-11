#!/bin/sh

set -e

# Make sure the target directory is writable.
if [ ! -w "$TARGET" ]; then
    echo "Target directory $TARGET is not writable."
    echo "Please run this script through sudo to allow writing to $TARGET."
    exit 1
fi

# Make sure we don't overwrite an existing installation.
if [ -f "$TARGET/databricks" ]; then
    echo "Target path $TARGET/databricks already exists."
    exit 1
fi

# Change into temporary directory.
cd "$(mktemp -d)"

# Download release archive.

# Unzip release archive.
unzip -q -o "scripts/databricks_cli_linux_amd64.zip"

# Add databricks to path.
chmod +x ./databricks
cp ./databricks "$TARGET"
echo "Installed $($TARGET/databricks -v) at $TARGET/databricks."
