#!/usr/bin/bash

# TODO: take the version of spark and hadoop from CI instead of hardcoding
wget -q https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xvf spark-3.5.0-bin-hadoop3.tgz
spark-3.5.0-bin-hadoop3/sbin/start-connect-server.sh --packages org.apache.spark:spark-connect_2.12:3.5.0
