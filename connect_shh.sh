#!/bin/bash

# Parsing from: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift # past argument
      shift # past value
      ;;
    -ip|-h|--ip|--host)
      HOST="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$HOST" ]; then
  echo "Parser Error, you need to provide host ip (any of -ip, -h, --ip, --host)."
  exit 42
elif [ -z "$PORT" ]; then
  echo "Parser Error, you need to provide the port to tunnel (any of -p, --port)."
  exit 42
fi

ssh -f $HOST -L $PORT:localhost:$PORT -N
echo "Forwarding port $PORT from machine $HOST to local port $PORT"