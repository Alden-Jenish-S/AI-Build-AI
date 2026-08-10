#!/bin/sh

set -u

# Run the full search for every task in tasks/.

cd "$(dirname "$0")" || exit 1

BUDGET="${BUDGET:-6}"
FAILED=0

for task_dir in tasks/*; do
    if [ -d "$task_dir" ]; then
        task_name="$(basename "$task_dir")"

        echo "================================"
        echo "Running task: $task_name"
        echo "================================"

        if python3 eval/run_search.py "$task_name" --budget "$BUDGET"; then
            echo "Completed: $task_name"
        else
            echo "Failed: $task_name"
            FAILED=1
        fi
    fi
done

exit "$FAILED"
