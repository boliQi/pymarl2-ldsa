#!/bin/bash

# Script to kill "dead" python run_experiment.py processes
# Logic:
# 1. Finds all processes matching "python ... run_experiment.py"
# 2. Sorts them by start time (oldest first)
# 3. If multiple found: Kills all EXCEPT the most recent one (presumably the normal one)
# 4. If only one found: Asks for confirmation

# Ensure ps is available (standard on Linux)
if ! command -v ps &> /dev/null; then
    echo "Error: 'ps' command not found."
    exit 1
fi

# Find PIDs, Start Time, Command. Sort by Start Time.
# grep "python.*run_experiment.py" matches "python run_experiment.py", "python3 run_experiment.py", etc.
matches=$(ps -eo pid,lstart,cmd --sort=start_time | grep "python.*run_experiment.py" | grep -v grep)

if [ -z "$matches" ]; then
    echo "No 'run_experiment.py' processes found."
    exit 0
fi

echo "Found the following processes (Sorted by Start Time):"
echo "------------------------------------------------------------------------"
echo "$matches"
echo "------------------------------------------------------------------------"

# Count processes
count=$(echo "$matches" | wc -l)

if [ "$count" -gt 1 ]; then
    echo "Detected $count processes."
    echo "Automated cleanup: Keeping the MOST RECENT process, killing older ones."
    
    # Keep the PID of the newest (last line)
    keep_pid=$(echo "$matches" | tail -n 1 | awk '{print $1}')
    
    # Kill others
    echo "$matches" | head -n -1 | while read -r line; do
        kill_pid=$(echo "$line" | awk '{print $1}')
        echo "Killing OLD process: $kill_pid"
        kill -9 "$kill_pid"
    done
    
    echo "Protected NEWEST process: $keep_pid"

else
    # Only one process found
    pid=$(echo "$matches" | awk '{print $1}')
    echo "Only one process found (PID: $pid)."
    echo "WARNING: This is the ONLY running instance."
    echo "If this is your 'dead' process, say Y. If it's your 'normal' run, say N."
    
    read -p "Kill PID $pid? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo "Killing $pid..."
        kill -9 "$pid"
        echo "Done."
    else
        echo "Cancelled."
    fi
fi
