#!/bin/bash

# Define the data directory and expected output files
DATA_DIR="/data"
OUTPUT_FILES=("climate_data.db")

# Run the data pipeline
echo "Running the data pipeline..."
python3 pipeline.py

# Check if the data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Data directory '$DATA_DIR' does not exist."
  exit 1
fi

# Check if the output files are created
for file in "${OUTPUT_FILES[@]}"; do
  if [ ! -f "$DATA_DIR/$file" ]; then
    echo "Error: Expected output file '$file' does not exist in the data directory."
    exit 1
  fi
done

echo "All tests passed successfully."
