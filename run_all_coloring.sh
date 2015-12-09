#!/usr/bin/bash

for f in ./problems/coloring/data/*
do
  echo $f
	cat $f | pypy3 coloring.py
done
