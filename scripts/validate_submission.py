"""Compile a bundle with the installed competition compiler, without inference.

Runs under Python 3.12 in the separate harness environment. Tool signatures and
model routing are real declarations; tool bodies are inert during compilation.
Full sandbox tests and production vLLM loading remain separate required gates.
"""
import argparse
import json
from pathlib import Path

from adk_submission import (compile_submission, validate_directory, discover_adapters,
                            ModelRegistry, ToolRegistry, SkillRegistry, CallbackRegistry,
                            SubmissionLimits, GenerationConstraints, NumericRange)
from google.adk.models.lite_llm import LiteLlm


async def run_command(command: str) -> str: return "compile-only"
async def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str: return "compile-only"
async def edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str: return "compile-only"
async def write_file(filepath: str, content: str) -> str: return "compile-only"
async def get_status() -> str: return "compile-only"
async def submit_patch() -> str: return "compile-only"
async def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str: return "compile-only"
async def search_similar_code(query: str, k: int = 10) -> str: return "compile-only"
async def get_code_subgraph(nodes: list[str]) -> str: return "compile-only"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    args = parser.parse_args()
    limits = SubmissionLimits(max_total_size_bytes=3*1024**3, max_yaml_size_bytes=50*1024**2,
        max_file_count=10000, max_yaml_files=1000, max_instruction_chars=1000000,
        max_total_instruction_chars=10000000, max_agents=500, max_sub_agent_depth=50,
        allowed_file_extensions=frozenset({'.yaml','.yml','.md','.txt','.py','.json','.safetensors'}),
        adapter_extensions=frozenset({'.safetensors'}))
    validate_directory(args.directory, limits=limits)
    models, tools = ModelRegistry(), ToolRegistry()
    models.register(args.model, LiteLlm(model="openai/"+args.model, api_base="http://127.0.0.1:8000/v1", api_key="local"))
    for function in (run_command, read_file, edit_file, write_file, get_status, submit_patch,
                     get_code_neighbors, search_similar_code, get_code_subgraph):
        tools.register(function.__name__, function)
    adapters = discover_adapters(args.directory, adapter_extensions=frozenset({'.safetensors'}))
    adapters.register_all(models, lambda info: LiteLlm(model="openai/"+info.name, api_base="http://127.0.0.1:8000/v1", api_key="local"))
    compiled = compile_submission(submission_dir=args.directory, tool_registry=tools, model_registry=models,
        skill_registry=SkillRegistry(), callback_registry=CallbackRegistry(), limits=limits,
        generation_constraints=GenerationConstraints(allowed_fields=None, max_output_tokens=NumericRange(1,32768), thinking_budget=NumericRange(0,32768)))
    print(json.dumps({"status": "compiled", "agent": compiled.name, "model": args.model,
                      "inference_executed": False, "authority": "installed adk-submission; reports/harness-environment.json records source identity"}))


if __name__ == "__main__":
    main()
