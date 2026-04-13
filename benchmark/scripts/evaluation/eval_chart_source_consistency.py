import json
import os
from openai import OpenAI
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from json_repair import repair_json
import time

load_dotenv()

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5.2")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
STREAMING = os.getenv("STREAMING", "true").lower() == "true"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


def _parse_json_result(result):
    """Parse agent-planner result using json-repair library."""

    # Remove BOM and invisible characters
    text = result.strip("\ufeff\u200b\u200c\u200d")

    # Use json-repair to fix and parse
    try:
        repaired = repair_json(text)
        output = json.loads(repaired)

        if not isinstance(output, dict):
            raise ValueError(f"Invalid type: {type(output)}")

        return output
    except Exception as e:
        raise ValueError(f"Could not parse JSON: {e}")


def eval_chart_source_consistency(
    report_root_dir, query_id, eval_system_name, result_root_dir
):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/chart_source_consistency.json"
    VISUAL_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )
    REF_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/refs_with_web_content.json"
    )

    try:
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            figures = json.load(f)
        with open(REF_PATH, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    charts = [fig for fig in figures if fig.get("type") == "chart"]
    ref_map = {item["ref_idx"]: item["web_content"] for item in refs}

    if not charts:
        result = {
            "normalized_score": 0.0,
            "item_count": 0,
            "scored_item_count": 0,
            "evaluation_results": [],
        }
        os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
        with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ No charts found, zero score saved to {EVAL_RESULT_PATH}")
        return

    evaluation_results = []
    total_scores = []

    system_prompt = """
You are an expert Data Provenance & Chart Verification Auditor. Your task is to evaluate **Chart-Source Consistency** in a research report using only the provided inputs.

You may be provided with:
1) **Chart**: The chart itself.
2) **Caption**: The full text associated with the chart.
3) **References**: A JSON list of cited source snippets associated with the chart. Treat this list as the only permissible evidence; do not assume access to external links or any information not contained in the provided snippets.

GOAL:
Decide whether the chart is **consistent** with the provided sources.

OUT OF SCOPE:
1) Pure visual aesthetics issues (layout, fonts, colors) unless they prevent verification.
2) Real-world credibility of sources; only whether the provided snippets support the chart as presented.
3) Missing or insufficient evidence. Only focus on **contradictions** between the chart and the provided sources.

DEFINITION OF AN ISSUE:
An issue exists when at least one claim from the chart is **contradicted** by References under the same (or materially equivalent) claim context.

COUNTING RULES:
- Each distinct underlying problem counts as ONE issue.
- Multiple symptoms from the same root cause count as ONE issue.
- If there are multiple unrelated contradictions, count them as separate issues.

INSTRUCTIONS:
- **Step 1**: Read the **Chart** and **Caption** together to extract the chart’s claim(s).
- **Step 2**: Use the provided **References** as the only evidence to verify those claim(s).
- **Step 3**: Report every distinct issue you find, each with: 
  - the chart claim,
  - a short description,
  - direct quote(s) from References with ref_idx.
- **Step 4**: Count the total number of distinct issues.

OUTPUT FORMAT:
Respond with a JSON object. Do not add markdown code blocks (like ```json) around the output.
{
  "total_issues": 5,
  "issues": [
    {
      "id": 1,
      "chart_claim": "the specific claim from the chart being checked",
      "description": "brief summary of the issue",
      "evidence": [
        {
          "ref_idx": "1",
          "quote": "direct quote from the corresponding reference snippet"
        },
        ...
      ]
    }
    ...
  ]
}
"""

    for fig in tqdm(charts, desc="🤖 Evaluating chart-source consistency"):
        img_base64 = fig.get("img_base64", "")
        caption = fig.get("caption", "")
        ref_idxs = fig.get("ref_idxs", [])
        references_payload = []
        for rid in ref_idxs:
            references_payload.append(
                {
                    "ref_idx": rid,
                    "content": ref_map.get(rid, ""),
                }
            )

        if isinstance(img_base64, dict) and "error" in img_base64:
            evaluation_results.append(
                {"caption": caption, "error": img_base64["error"], "total_score": 0.0}
            )
            total_scores.append(0.0)
            continue

        if not ref_idxs:
            evaluation_results.append(
                {
                    "caption": caption,
                    "error": "absent references for source verification",
                    "total_score": 0.0,
                }
            )
            total_scores.append(0.0)
            continue

        user_prompt = f"""
Please evaluate the Chart-Source Consistency based on the provided rubric.

<caption>
{caption}
</caption>

<references>
{json.dumps(references_payload, ensure_ascii=False)}
</references>

Remember to output ONLY the JSON object. Do not include any other text or markdown formatting.
"""

        user_message_content = [
            {
                "type": "text",
                "text": user_prompt,
            },
            {"type": "image_url", "image_url": {"url": img_base64}},
        ]

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=EVAL_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=TEMPERATURE,
                    stream=STREAMING,
                )

                if STREAMING:
                    result_str = ""
                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content:
                                result_str += delta.content
                else:
                    result_str = response.choices[0].message.content

                if result_str.strip().startswith("```json"):
                    result_str = result_str.strip()[7:]
                if result_str.strip().endswith("```"):
                    result_str = result_str.strip()[:-3]
                evaluation_result = _parse_json_result(result_str)

                if "total_issues" not in evaluation_result:
                    raise ValueError(
                        f"Missing 'total_issues' field in response, result: {result_str}"
                    )

                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(4**attempt)

        if last_err is not None:
            print(f"❌ Error during evaluation: {last_err}")
            evaluation_results.append(
                {"caption": caption, "error": str(last_err), "total_score": None}
            )
            continue

        N_issues = evaluation_result.get("total_issues", None)
        if N_issues is None:
            score = None
        elif N_issues == 0:
            score = 10
        elif N_issues == 1:
            score = 9
        elif N_issues == 2:
            score = 8
        elif N_issues == 3:
            score = 7
        elif N_issues == 4:
            score = 6
        elif N_issues == 5:
            score = 5
        elif N_issues == 6:
            score = 4
        elif N_issues == 7:
            score = 3
        elif N_issues == 8:
            score = 2
        else:  # N_issues >= 9
            score = 1

        evaluation_result["total_score"] = score
        evaluation_result["caption"] = caption
        evaluation_results.append(evaluation_result)

        if score is not None:
            total_scores.append(score)

    if total_scores:
        max_score = 10
        normalized_score = sum(total_scores) / (len(total_scores) * max_score)
        normalized_score = round(normalized_score, 4)
    else:
        normalized_score = None

    result = {
        "normalized_score": normalized_score,
        "item_count": len(charts),
        "scored_item_count": len(total_scores),
        "evaluation_details": evaluation_results,
    }

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Evaluation completed, results saved to {EVAL_RESULT_PATH} ｜ Normalized Score: {normalized_score} ｜ Scored Items: {len(total_scores)}/{len(charts)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_root_dir",
        type=str,
        required=True,
        help="Root directory path for reports",
    )
    parser.add_argument(
        "--query_id", type=str, required=True, help="Evaluation report ID"
    )
    parser.add_argument(
        "--eval_system_name", type=str, required=True, help="Evaluation system name"
    )
    parser.add_argument(
        "--result_root_dir",
        type=str,
        required=True,
        help="Root directory path for evaluation results",
    )
    args = parser.parse_args()
    eval_chart_source_consistency(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
