#!/bin/bash

# Check if arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <script_name> <number_of_times>"
  echo "Example: $0 main_GAIA.py 5"
  exit 1
fi

script_name=$1
n=$2

# Check if the script exists
if [ ! -f "$script_name" ]; then
    echo "Error: Script '$script_name' not found."
    exit 1
fi

for ((i=1; i<=n; i++))
do
   echo ""
   echo "Starting run #$i of $script_name..."
   echo "=============================="
   python "$script_name"
   
   # Check return code (optional)
   if [ $? -ne 0 ]; then
       echo "Run #$i failed."
       # Uncomment the next line to stop on failure
       # exit 1 
   fi
done

echo ""
echo "Completed $n runs."
