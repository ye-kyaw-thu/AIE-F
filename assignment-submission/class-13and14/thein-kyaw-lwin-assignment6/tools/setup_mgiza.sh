#!/bin/bash
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

echo "Creating training tools directory..."
mkdir -p "$PROJECT_ROOT/tools/training-tools"

# Note: tools/mgiza source directory was removed during space optimization. 
# This script is preserved for compile documentation purposes.
if [ -d "$PROJECT_ROOT/tools/mgiza" ]; then
  echo "Copying compiled MGIZA binaries..."
  cp "$PROJECT_ROOT/tools/mgiza/mgizapp/bin/mgiza" "$PROJECT_ROOT/tools/training-tools/"
  cp "$PROJECT_ROOT/tools/mgiza/mgizapp/bin/mkcls" "$PROJECT_ROOT/tools/training-tools/"
  cp "$PROJECT_ROOT/tools/mgiza/mgizapp/bin/snt2cooc" "$PROJECT_ROOT/tools/training-tools/"

  echo "Copying merge alignment script..."
  cp "$PROJECT_ROOT/tools/mgiza/mgizapp/scripts/merge_alignment.py" "$PROJECT_ROOT/tools/training-tools/"
fi

echo "Making all tools executable..."
chmod +x "$PROJECT_ROOT/tools/training-tools/"*

echo "MGIZA setup complete!"
ls -lh "$PROJECT_ROOT/tools/training-tools"
