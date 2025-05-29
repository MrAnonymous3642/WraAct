#!/bin/bash

python3 ./polytope_samples/generate_polytope_samples.py
python3 ./polytope_bounds/calculate_polytope_bounds.py
python3 ./function_hulls/calculate_function_hulls.py
python3 ./hull_volumes/calculate_volumes.py

