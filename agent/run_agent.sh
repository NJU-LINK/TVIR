#!/bin/bash

# Check if required arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./run_agent.sh <llm_model> <agent> [query_id]"
    echo "Example: ./run_agent.sh claude-4-5 tvir_agent 016001"
    echo "Example: ./run_agent.sh claude-4-5 tvir_agent"
    exit 1
fi

LLM_MODEL=$1
AGENT=$2
QUERY_ID=$3


uv run python main.py \
  llm="$LLM_MODEL" \
  agent="$AGENT" \
  --query_id "$QUERY_ID" \