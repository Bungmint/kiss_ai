#!/bin/bash

# Script to release to public GitHub repository with file filtering and version tagging,
# and publish to PyPI.
# Repository: https://github.com/ksenxx/kiss_ai
# PyPI: https://pypi.org/project/kiss-agent-framework/

set -e  # Exit on error

# =============================================================================
# CONFIGURATION: Files and directories to EXCLUDE from public release
# =============================================================================
# Add paths relative to repo root that should NOT be pushed to public repo
PRIVATE_FILES=(
    # Add private files/directories here, one per line
    # Example:
    # "private_config.yaml"
    # "internal_docs/"
    # "secrets/"
)

# =============================================================================
# Constants
# =============================================================================
PUBLIC_REMOTE="public"
PUBLIC_REPO_URL="https://github.com/ksenxx/kiss_ai.git"
PUBLIC_REPO_SSH="git@github.com:ksenxx/kiss_ai.git"
VERSION_FILE="src/kiss/_version.py"
README_FILE="README.md"
RELEASE_BRANCH="release-staging"
PYPI_PACKAGE_NAME="kiss-agent-framework"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Get version from _version.py
get_version() {
    if [[ ! -f "$VERSION_FILE" ]]; then
        print_error "Version file not found: $VERSION_FILE"
        exit 1
    fi
    # Extract version string from __version__ = "x.y.z"
    VERSION=$(grep -oP '__version__\s*=\s*"\K[^"]+' "$VERSION_FILE" 2>/dev/null || \
              grep '__version__' "$VERSION_FILE" | sed 's/.*"\(.*\)".*/\1/')
    if [[ -z "$VERSION" ]]; then
        print_error "Could not extract version from $VERSION_FILE"
        exit 1
    fi
    echo "$VERSION"
}

# Bump the patch version (x.y.z -> x.y.(z+1))
bump_version() {
    local current_version="$1"
    
    # Split version into parts
    local major minor patch
    IFS='.' read -r major minor patch <<< "$current_version"
    
    # Increment patch version
    patch=$((patch + 1))
    
    local new_version="${major}.${minor}.${patch}"
    echo "$new_version"
}

# Update version in _version.py
update_version_file() {
    local new_version="$1"
    
    if [[ ! -f "$VERSION_FILE" ]]; then
        print_error "Version file not found: $VERSION_FILE"
        return 1
    fi
    
    # Update the version in the file
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"${new_version}\"/" "$VERSION_FILE"
    rm -f "${VERSION_FILE}.bak"
    
    print_info "Updated $VERSION_FILE to version $new_version"
}

# Check if remote exists, add if not
ensure_remote() {
    if ! git remote get-url "$PUBLIC_REMOTE" &>/dev/null; then
        print_info "Adding remote '$PUBLIC_REMOTE'..."
        git remote add "$PUBLIC_REMOTE" "$PUBLIC_REPO_SSH"
    else
        print_info "Remote '$PUBLIC_REMOTE' exists"
    fi
}

# Update version in README.md
update_readme_version() {
    local version="$1"
    if [[ ! -f "$README_FILE" ]]; then
        print_warn "README file not found: $README_FILE - skipping version update"
        return
    fi
    
    # Update the **Version:** line in README.md
    if grep -q '^\*\*Version:\*\*' "$README_FILE"; then
        sed -i.bak "s/^\*\*Version:\*\* .*/\*\*Version:\*\* $version/" "$README_FILE"
        rm -f "${README_FILE}.bak"
        print_info "Updated version in $README_FILE to $version"
    else
        print_warn "Version line not found in $README_FILE - skipping update"
    fi
}

# Build and publish to PyPI
publish_to_pypi() {
    local version="$1"
    
    print_step "Building package for PyPI..."
    
    # Clean previous builds
    rm -rf dist/ build/
    
    # Check if build and twine are available
    if ! uv run python -c "import build" &>/dev/null; then
        print_info "Installing build package..."
        uv pip install build twine
    fi
    
    # Build the package
    uv run python -m build
    
    if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist/)" ]]; then
        print_error "Build failed - no files in dist/"
        return 1
    fi
    
    print_info "Built packages:"
    ls -la dist/
    
    # Check the package
    print_step "Checking package..."
    uv run python -m twine check dist/*
    
    # Upload to PyPI
    print_step "Uploading to PyPI..."
    
    # Check for PyPI token
    if [[ -z "$UV_PUBLISH_TOKEN" ]]; then
        print_error "UV_PUBLISH_TOKEN environment variable is not set"
        print_info "Please set it with: export UV_PUBLISH_TOKEN='pypi-your-token-here'"
        return 1
    fi
    
    uv run python -m twine upload dist/* -u __token__ -p "$UV_PUBLISH_TOKEN"
    
    print_info "Successfully published version $version to PyPI"
    print_info "View at: https://pypi.org/project/${PYPI_PACKAGE_NAME}/${version}/"
}

# =============================================================================
# Main Release Process
# =============================================================================
main() {
    print_step "Starting release to public repository"
    echo "Repository: $PUBLIC_REPO_URL"
    echo

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Get current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_info "Current branch: $CURRENT_BRANCH"

    # Check for uncommitted changes (allow version file and README to be modified by this script)
    if ! git diff-index --quiet HEAD -- ':!'"$VERSION_FILE" ':!'"$README_FILE"; then
        print_error "You have uncommitted changes. Please commit or stash them first."
        exit 1
    fi

    # Get current version and bump it
    CURRENT_VERSION=$(get_version)
    VERSION=$(bump_version "$CURRENT_VERSION")
    TAG_NAME="v$VERSION"
    
    print_info "Current version: $CURRENT_VERSION"
    print_info "New version: $VERSION (tag: $TAG_NAME)"
    
    # Update version file
    print_step "Bumping version..."
    update_version_file "$VERSION"
    
    # Commit the version bump
    git add "$VERSION_FILE"
    git commit -m "Bump version to $VERSION"
    print_info "Committed version bump"

    # Ensure remote exists
    ensure_remote

    # Check if there are private files to exclude
    if [[ ${#PRIVATE_FILES[@]} -eq 0 ]]; then
        print_info "No private files configured - pushing entire repo"
        
        # Simple push without filtering
        print_step "Pushing to public remote..."
        git push "$PUBLIC_REMOTE" "$CURRENT_BRANCH:main" --force-with-lease
        
    else
        print_info "Private files to exclude:"
        for file in "${PRIVATE_FILES[@]}"; do
            echo "  - $file"
        done
        echo

        # Create a temporary branch for the filtered release
        print_step "Creating filtered release branch..."
        
        # Delete release branch if it exists
        git branch -D "$RELEASE_BRANCH" 2>/dev/null || true
        
        # Create new branch from current HEAD
        git checkout -b "$RELEASE_BRANCH"

        # Remove private files from the release branch
        print_step "Removing private files from release..."
        for file in "${PRIVATE_FILES[@]}"; do
            if [[ -e "$file" ]]; then
                git rm -rf --cached "$file" 2>/dev/null || true
                print_info "Removed: $file"
            else
                print_warn "File not found (skipping): $file"
            fi
        done

        # Check if there are changes to commit
        if ! git diff-index --quiet HEAD --; then
            git commit -m "Release $VERSION - remove private files"
        fi

        # Push the filtered branch to public remote
        print_step "Pushing filtered branch to public remote..."
        git push "$PUBLIC_REMOTE" "$RELEASE_BRANCH:main" --force-with-lease

        # Return to original branch
        print_step "Cleaning up..."
        git checkout "$CURRENT_BRANCH"
        git branch -D "$RELEASE_BRANCH"
    fi

    # Handle version tagging
    print_step "Creating version tag..."
    
    # Update version in README.md
    update_readme_version "$VERSION"
    
    # Commit README version update if there are changes
    if ! git diff --quiet "$README_FILE" 2>/dev/null; then
        git add "$README_FILE"
        git commit -m "Update version to $VERSION in README.md"
        print_info "Committed README version update"
        
        # Re-push the branch with the version update
        print_step "Re-pushing branch with version update..."
        if [[ ${#PRIVATE_FILES[@]} -eq 0 ]]; then
            git push "$PUBLIC_REMOTE" "$CURRENT_BRANCH:main" --force-with-lease
        else
            # Need to recreate the filtered branch with the new commit
            git branch -D "$RELEASE_BRANCH" 2>/dev/null || true
            git checkout -b "$RELEASE_BRANCH"
            for file in "${PRIVATE_FILES[@]}"; do
                if [[ -e "$file" ]]; then
                    git rm -rf --cached "$file" 2>/dev/null || true
                fi
            done
            if ! git diff-index --quiet HEAD --; then
                git commit -m "Release $VERSION - remove private files"
            fi
            git push "$PUBLIC_REMOTE" "$RELEASE_BRANCH:main" --force-with-lease
            git checkout "$CURRENT_BRANCH"
            git branch -D "$RELEASE_BRANCH"
        fi
    fi
    
    # Create and push tag
    git tag -a "$TAG_NAME" -m "Release $VERSION"
    print_info "Created local tag: $TAG_NAME"
    git push "$PUBLIC_REMOTE" "$TAG_NAME"
    print_info "Pushed tag '$TAG_NAME' to public remote"

    # Publish to PyPI
    print_step "Publishing to PyPI..."
    publish_to_pypi "$VERSION"

    echo
    print_info "========================================"
    print_info "Release completed successfully!"
    print_info "========================================"
    print_info "GitHub:  $PUBLIC_REPO_URL"
    print_info "PyPI:    https://pypi.org/project/${PYPI_PACKAGE_NAME}/"
    print_info "Version: $VERSION"
    print_info "Tag:     $TAG_NAME"
    echo
}

# Run main function
main "$@"
