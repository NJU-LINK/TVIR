# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.core.pipeline import create_pipeline_components, execute_task_pipeline
from src.logging.task_logger import bootstrap_logger

logger = bootstrap_logger()


def load_query_from_json(
    query_id: str, json_path: str = "../benchmark/data/query.json"
) -> dict:
    """Load query information from JSON file by query_id.

    Args:
        query_id: The ID of the query to load (e.g., "002001")
        json_path: Path to the query JSON file

    Returns:
        Dictionary containing query information

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If query_id not found in JSON
    """
    json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(f"Query JSON file not found: {json_path}")

    with open(json_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Find the query with matching ID
    for query_item in queries:
        if query_item.get("id") == query_id:
            logger.info(
                f"Loaded query {query_id}: {query_item.get('query', '')[:100]}..."
            )
            return query_item

    raise ValueError(f"Query ID '{query_id}' not found in {json_path}")


async def amain(cfg: DictConfig, query_id: Optional[str]) -> None:
    """Asynchronous main function."""

    logger.info(OmegaConf.to_yaml(cfg))

    llm_name = cfg.llm.model_name

    if query_id and llm_name:
        # Load query from JSON file
        try:
            query_data = load_query_from_json(query_id)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading query: {e}")
            return

        task_description = query_data.get("query", "")
        task_id = f"{llm_name}/{query_id}"
    else:
        # Fallback to default task (for backward compatibility)
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_description = ""

    logger.info(f"Task description: {task_description[:200]}...")

    task_file_name = ""

    # Create pipeline components using the factory function
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg, custom_env={"TASK_ID": task_id})
    )

    # Execute task using the pipeline
    final_summary, final_boxed_answer, log_file_path = await execute_task_pipeline(
        cfg=cfg,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=str, default=None)
    args, hydra_args = parser.parse_known_args()

    with initialize_config_dir(config_dir=os.path.abspath("conf"), version_base=None):
        cfg = compose(config_name="config", overrides=hydra_args)
        asyncio.run(amain(cfg, args.query_id))


if __name__ == "__main__":
    main()
