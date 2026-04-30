#!/bin/bash

## Building RDR Model
python scrdr_interactive.py --input ./data/snake.csv --target species_name \
--tree snake_rules_demo.json --mode build | tee running_snake_demo.log

## Testing
python scrdr_interactive.py --input ./data/snake.csv --target species_name \
--tree snake_rules_demo.json --mode test