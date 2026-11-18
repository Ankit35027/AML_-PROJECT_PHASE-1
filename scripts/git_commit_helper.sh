#!/usr/bin/env bash
# ==============================================================================
# Git Commit & Author Identity Helper for 4-Member Team
# ==============================================================================
# Usage:
#   1. Configure member credentials below or via environment variables
#   2. Run: ./scripts/git_commit_helper.sh <member_number> "<commit_message>" [files...]
#      Example: ./scripts/git_commit_helper.sh 1 "feat(gat): implement dynamic graph attention layer" src/models_phase2/gat_layer.py
#   3. Or set active user for the local repo:
#      ./scripts/git_commit_helper.sh set-active <member_number>
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Default Placeholders (User will supply actual IDs/names/emails)
# ------------------------------------------------------------------------------
MEMBER1_NAME="${MEMBER1_NAME:-Member 1}"
MEMBER1_EMAIL="${MEMBER1_EMAIL:-member1@example.com}"
MEMBER1_ID="${MEMBER1_ID:-WS1_DATA}"

MEMBER2_NAME="${MEMBER2_NAME:-Member 2}"
MEMBER2_EMAIL="${MEMBER2_EMAIL:-member2@example.com}"
MEMBER2_ID="${MEMBER2_ID:-WS2_GAT_SPATIAL}"

MEMBER3_NAME="${MEMBER3_NAME:-Member 3}"
MEMBER3_EMAIL="${MEMBER3_EMAIL:-member3@example.com}"
MEMBER3_ID="${MEMBER3_ID:-WS3_TEMPORAL_HEAD}"

MEMBER4_NAME="${MEMBER4_NAME:-Member 4}"
MEMBER4_EMAIL="${MEMBER4_EMAIL:-member4@example.com}"
MEMBER4_ID="${MEMBER4_ID:-WS4_EVAL_BENCH}"

# Config file path
CONFIG_FILE="$(dirname "$0")/team_members.env"
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

get_member_info() {
    local num="$1"
    case "$num" in
        1)
            echo "$MEMBER1_NAME|$MEMBER1_EMAIL|$MEMBER1_ID"
            ;;
        2)
            echo "$MEMBER2_NAME|$MEMBER2_EMAIL|$MEMBER2_ID"
            ;;
        3)
            echo "$MEMBER3_NAME|$MEMBER3_EMAIL|$MEMBER3_ID"
            ;;
        4)
            echo "$MEMBER4_NAME|$MEMBER4_EMAIL|$MEMBER4_ID"
            ;;
        *)
            echo "Error: Member number must be 1, 2, 3, or 4" >&2
            exit 1
            ;;
    esac
}

show_help() {
    cat << EOF
Team Git Commit Helper
======================
Current Registered Roster:
  [1] WS1 (Graph Data Pipeline):     $MEMBER1_NAME <$MEMBER1_EMAIL> (ID: $MEMBER1_ID)
  [2] WS2 (GAT Spatial Attention):   $MEMBER2_NAME <$MEMBER2_EMAIL> (ID: $MEMBER2_ID)
  [3] WS3 (Temporal Backbone/Heads): $MEMBER3_NAME <$MEMBER3_EMAIL> (ID: $MEMBER3_ID)
  [4] WS4 (Evaluation & Benchmark):  $MEMBER4_NAME <$MEMBER4_EMAIL> (ID: $MEMBER4_ID)

Commands:
  ./scripts/git_commit_helper.sh commit <member_num: 1-4> "<commit_message>" [files...]
  ./scripts/git_commit_helper.sh set-active <member_num: 1-4>
  ./scripts/git_commit_helper.sh status
  ./scripts/git_commit_helper.sh configure

EOF
}

cmd="${1:-help}"

case "$cmd" in
    commit)
        shift
        member_num="${1:-}"
        msg="${2:-}"
        if [ -z "$member_num" ] || [ -z "$msg" ]; then
            echo "Usage: ./scripts/git_commit_helper.sh commit <member_num: 1-4> \"<commit_message>\" [files...]"
            exit 1
        fi
        shift 2
        files=("$@")

        info=$(get_member_info "$member_num")
        name=$(echo "$info" | cut -d'|' -f1)
        email=$(echo "$info" | cut -d'|' -f2)
        id=$(echo "$info" | cut -d'|' -f3)

        if [ ${#files[@]} -gt 0 ]; then
            git add "${files[@]}"
        fi

        formatted_msg="[${id}] ${msg}"
        echo "Creating commit as: ${name} <${email}>"
        echo "Commit message: ${formatted_msg}"

        GIT_AUTHOR_NAME="$name" \
        GIT_AUTHOR_EMAIL="$email" \
        GIT_COMMITTER_NAME="$name" \
        GIT_COMMITTER_EMAIL="$email" \
        git commit -m "$formatted_msg"
        ;;

    set-active)
        shift
        member_num="${1:-}"
        if [ -z "$member_num" ]; then
            echo "Usage: ./scripts/git_commit_helper.sh set-active <member_num: 1-4>"
            exit 1
        fi
        info=$(get_member_info "$member_num")
        name=$(echo "$info" | cut -d'|' -f1)
        email=$(echo "$info" | cut -d'|' -f2)

        git config user.name "$name"
        git config user.email "$email"
        echo "Active Git author for this repository set to: $name <$email>"
        ;;

    status)
        echo "Current active Git user.name:  $(git config user.name || echo 'Not configured')"
        echo "Current active Git user.email: $(git config user.email || echo 'Not configured')"
        echo "Current branch:                $(git rev-parse --abbrev-ref HEAD)"
        echo ""
        show_help
        ;;

    configure)
        echo "Interactive Team Setup:"
        read -r -p "Member 1 Name: " M1_NAME
        read -r -p "Member 1 Email/ID: " M1_EMAIL
        read -r -p "Member 2 Name: " M2_NAME
        read -r -p "Member 2 Email/ID: " M2_EMAIL
        read -r -p "Member 3 Name: " M3_NAME
        read -r -p "Member 3 Email/ID: " M3_EMAIL
        read -r -p "Member 4 Name: " M4_NAME
        read -r -p "Member 4 Email/ID: " M4_EMAIL

        cat > "$CONFIG_FILE" << EOF
MEMBER1_NAME="${M1_NAME:-Member 1}"
MEMBER1_EMAIL="${M1_EMAIL:-member1@example.com}"
MEMBER1_ID="${M1_NAME:-M1}"

MEMBER2_NAME="${M2_NAME:-Member 2}"
MEMBER2_EMAIL="${M2_EMAIL:-member2@example.com}"
MEMBER2_ID="${M2_NAME:-M2}"

MEMBER3_NAME="${M3_NAME:-Member 3}"
MEMBER3_EMAIL="${M3_EMAIL:-member3@example.com}"
MEMBER3_ID="${M3_NAME:-M3}"

MEMBER4_NAME="${M4_NAME:-Member 4}"
MEMBER4_EMAIL="${M4_EMAIL:-member4@example.com}"
MEMBER4_ID="${M4_NAME:-M4}"
EOF
        echo "Saved configuration to $CONFIG_FILE"
        ;;

    *)
        show_help
        ;;
esac

