#!/bin/bash

# Start powermetrics and grep for lines containing "mW"
sudo powermetrics -i 1000 | grep "mW" >> energy.txt

