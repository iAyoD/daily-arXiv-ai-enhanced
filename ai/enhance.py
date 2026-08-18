import os
import json
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict
import requests

import argparse
from tqdm import tqdm

from dotenv import load_dotenv
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

load_dotenv(REPO_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=False)

template = (BASE_DIR / "template.txt").read_text()
system = (BASE_DIR / "system.txt").read_text()

REQUIRED_AI_FIELDS = ("tldr", "motivation", "method", "result", "conclusion")
JSON_SCHEMA_INSTRUCTION = (
    "Return only one valid JSON object with exactly these string fields: "
    '"tldr", "motivation", "method", "result", and "conclusion". '
    "Every field is required and must be non-empty. "
    "Do not wrap the JSON in Markdown or add any text outside the JSON object."
)
GITHUB_API_CONNECT_TIMEOUT_SECONDS = 10
GITHUB_API_READ_TIMEOUT_SECONDS = 30
GITHUB_API_MAX_ATTEMPTS = 4
GITHUB_API_BACKOFF_BASE_SECONDS = 1
GITHUB_API_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--max_ai_attempts", type=int, default=3, help="Maximum AI generation attempts per item")
    return parser.parse_args()


def parse_ai_response(content: str, item_id: str) -> Structure:
    if not isinstance(content, str):
        raise ValueError(f"AI response for {item_id} is not text")

    try:
        ai_data = json.loads(content)
    except JSONDecodeError as e:
        raise ValueError(f"AI response for {item_id} is not valid JSON") from e

    try:
        response = Structure.model_validate(ai_data)
    except ValidationError as e:
        raise ValueError(f"AI response for {item_id} does not match the required schema: {e}") from e

    missing_or_empty = [
        field
        for field in REQUIRED_AI_FIELDS
        if not getattr(response, field).strip()
    ]
    if missing_or_empty:
        fields = ", ".join(missing_or_empty)
        raise ValueError(f"AI response for {item_id} has empty fields: {fields}")

    return response


def generate_ai_fields(chain, repair_chain, item: Dict, language: str, max_ai_attempts: int) -> Dict:
    item_id = item.get("id", "unknown")
    last_error = None
    last_response = None

    for attempt in range(1, max_ai_attempts + 1):
        if attempt == 1:
            raw_response = chain.invoke({
                "language": language,
                "content": item["summary"],
            })
        else:
            print(
                f"Retrying AI JSON generation for {item_id} "
                f"(attempt {attempt}/{max_ai_attempts}): {last_error}",
                file=sys.stderr,
            )
            raw_response = repair_chain.invoke({
                "language": language,
                "content": item["summary"],
                "previous_response": last_response or "",
                "validation_error": str(last_error),
            })

        last_response = raw_response.content
        try:
            return parse_ai_response(last_response, item_id).model_dump()
        except ValueError as e:
            last_error = e

    raise RuntimeError(f"AI generation failed schema validation for {item_id}") from last_error


def fetch_github_repo_metadata(
    owner: str,
    repo: str,
    github_token: str | None,
    *,
    max_attempts: int = GITHUB_API_MAX_ATTEMPTS,
    connect_timeout: float = GITHUB_API_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = GITHUB_API_READ_TIMEOUT_SECONDS,
    backoff_base: float = GITHUB_API_BACKOFF_BASE_SECONDS,
) -> Dict:
    """Fetch GitHub repository metadata with bounded transient-error retries."""
    if max_attempts < 1:
        raise ValueError("GitHub API max_attempts must be at least 1")
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ValueError("GitHub API timeouts must be positive")
    if backoff_base < 0:
        raise ValueError("GitHub API backoff_base must not be negative")

    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                api_url,
                headers=headers,
                timeout=(connect_timeout, read_timeout),
            )
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"GitHub API request for {owner}/{repo} failed "
                    f"after {max_attempts} attempts"
                ) from error

            retry_delay = backoff_base * (2 ** (attempt - 1))
            print(
                f"Retrying GitHub API request for {owner}/{repo} "
                f"after {type(error).__name__} "
                f"(attempt {attempt + 1}/{max_attempts}, delay {retry_delay:g}s)",
                file=sys.stderr,
            )
            time.sleep(retry_delay)
            continue

        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError(f"GitHub API response for {owner}/{repo} is not an object")

            stars = data.get("stargazers_count")
            pushed_at = data.get("pushed_at")
            if not isinstance(stars, int) or stars < 0:
                raise ValueError(
                    f"GitHub API response for {owner}/{repo} has invalid stargazers_count"
                )
            if pushed_at is not None and not isinstance(pushed_at, str):
                raise ValueError(
                    f"GitHub API response for {owner}/{repo} has invalid pushed_at"
                )

            return {
                "code_stars": stars,
                "code_last_update": pushed_at[:10] if pushed_at else "",
            }

        if response.status_code == 404:
            return {}

        http_error = requests.HTTPError(
            f"GitHub API request for {owner}/{repo} returned HTTP "
            f"{response.status_code}",
            response=response,
        )
        if response.status_code not in GITHUB_API_RETRYABLE_STATUS_CODES:
            raise http_error
        if attempt == max_attempts:
            raise RuntimeError(
                f"GitHub API request for {owner}/{repo} returned HTTP "
                f"{response.status_code} after {max_attempts} attempts"
            ) from http_error

        retry_delay = backoff_base * (2 ** (attempt - 1))
        print(
            f"Retrying GitHub API request for {owner}/{repo} after HTTP "
            f"{response.status_code} "
            f"(attempt {attempt + 1}/{max_attempts}, delay {retry_delay:g}s)",
            file=sys.stderr,
        )
        time.sleep(retry_delay)

    raise AssertionError("GitHub API retry loop exited unexpectedly")


def process_single_item(
    chain,
    repair_chain,
    item: Dict,
    language: str,
    max_ai_attempts: int,
) -> Dict:
    def github_search_content(item_data: Dict) -> str:
        content_parts = [
            item_data.get("title", ""),
            item_data.get("summary", ""),
            item_data.get("comment", ""),
        ]
        external_urls = item_data.get("external_urls", [])
        if external_urls:
            if not isinstance(external_urls, list):
                raise ValueError(f"Item {item_data.get('id', 'unknown')} external_urls must be a list")
            content_parts.extend(str(url) for url in external_urls)
        return "\n".join(str(part) for part in content_parts if part)

    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.removesuffix(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            github_token = os.environ.get("TOKEN_GITHUB")
            code_info.update(fetch_github_repo_metadata(owner, repo, github_token))
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io[^\s,)]*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    item_id = item.get("id", "unknown")
    summary = item.get("summary")
    if not isinstance(summary, str):
        raise ValueError(f"Item {item_id} is missing text summary")

    # 检测代码可用性
    code_info = check_github_code(github_search_content(item))
    if code_info:
        item.update(code_info)

    item['AI'] = generate_ai_fields(chain, repair_chain, item, language, max_ai_attempts)
    return item


def process_all_items(
    data: List[Dict],
    model_name: str,
    language: str,
    max_workers: int,
    max_ai_attempts: int,
) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(
        model=model_name,
        api_key=require_env("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        temperature=0,
    ).bind(response_format={"type": "json_object"})
    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])
    repair_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(
            template=(
                "The previous AI response for the paper abstract failed schema validation.\n\n"
                f"{JSON_SCHEMA_INSTRUCTION}\n\n"
                "Validation error:\n{validation_error}\n\n"
                "Previous response:\n{previous_response}\n\n"
                "Paper abstract:\n{content}\n\n"
                "Regenerate the complete JSON in {language}."
            )
        )
    ])

    chain = prompt_template | llm
    repair_chain = repair_prompt_template | llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(
                process_single_item,
                chain,
                repair_chain,
                item,
                language,
                max_ai_attempts,
            ): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                item_id = data[idx].get("id", "unknown")
                raise RuntimeError(f"AI enhancement failed for item {item_id} at index {idx}") from e
    
    return processed_data

def main():
    args = parse_args()
    if args.max_ai_attempts < 1:
        raise ValueError("--max_ai_attempts must be at least 1")

    model_name = require_env("MODEL_NAME")
    language = require_env("LANGUAGE")

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers,
        args.max_ai_attempts
    )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
