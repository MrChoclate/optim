#!/usr/bin/bash

for f in ./problems/knapsack/data/*
do
  echo $f
	cat $f | pypy3 knapsack.py
done
