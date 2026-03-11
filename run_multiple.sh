#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_times>"
  exit 1
fi

n=$1

for ((i=1; i<=n; i++))
do
   echo ""
   echo "Starting run #$i..."
   echo "=============================="
   python main_GAIA.py
   
   # Check return code (optional)
   if [ $? -ne 0 ]; then
       echo "Run #$i failed."
       # Uncomment the next line to stop on failure
       # exit 1 
   fi
done

echo ""
echo "Completed $n runs."
