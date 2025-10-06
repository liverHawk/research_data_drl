#!/bin/bash
mlflow ui > /dev/null 2>&1 &
A_PID=$!

dvc queue start

if kill -0 $A_PID 2> /dev/null; then
    fuser -k 5000/tcp
fi