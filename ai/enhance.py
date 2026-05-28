import os
import json
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from email.utils import parsedate_to_datetime
from json import JSONDecodeError
from pathlib import Path
from threading import Lock
from typing import List, Dict, Optional
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
SENSITIVE_CHECK_URL = "https://spam.dw-dengwei.workers.dev"
SENSITIVE_CHECK_TIMEOUT_SECONDS = 10
SENSITIVE_CHECK_MIN_INTERVAL_SECONDS = 1.5
SENSITIVE_CHECK_MAX_ATTEMPTS = 5
SENSITIVE_CHECK_RETRY_BASE_SECONDS = 10


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


class SensitiveChecker:
    def __init__(
        self,
        url: str = SENSITIVE_CHECK_URL,
        timeout_seconds: int = SENSITIVE_CHECK_TIMEOUT_SECONDS,
        min_interval_seconds: float = SENSITIVE_CHECK_MIN_INTERVAL_SECONDS,
        max_attempts: int = SENSITIVE_CHECK_MAX_ATTEMPTS,
        retry_base_seconds: int = SENSITIVE_CHECK_RETRY_BASE_SECONDS,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("Sensitive check min interval must be non-negative")
        if max_attempts < 1:
            raise ValueError("Sensitive check max attempts must be at least 1")
        if retry_base_seconds < 0:
            raise ValueError("Sensitive check retry base must be non-negative")

        self.url = url
        self.timeout_seconds = timeout_seconds
        self.min_interval_seconds = min_interval_seconds
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self._lock = Lock()
        self._next_allowed_at = 0.0

    def is_sensitive(self, content: str, item_id: str, field_name: str) -> bool:
        if not isinstance(content, str):
            raise TypeError(f"Sensitive check content for {item_id} {field_name} must be text")

        for attempt in range(1, self.max_attempts + 1):
            resp = self._post_with_rate_limit(content)
            if resp.status_code == 429 and attempt < self.max_attempts:
                delay_seconds = self._retry_delay_seconds(resp, attempt)
                print(
                    f"Sensitive check rate limited for {item_id} {field_name}; "
                    f"retrying in {delay_seconds:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_attempts})",
                    file=sys.stderr,
                )
                self._defer_requests(delay_seconds)
                continue

            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"Sensitive check response for {item_id} {field_name} is not a JSON object"
                )
            if "sensitive" not in result:
                raise RuntimeError(
                    f"Sensitive check response for {item_id} {field_name} missing 'sensitive' field"
                )
            if not isinstance(result["sensitive"], bool):
                raise RuntimeError(
                    f"Sensitive check response for {item_id} {field_name} has non-boolean 'sensitive' field"
                )
            return result["sensitive"]

        raise RuntimeError(f"Sensitive check exhausted retries for {item_id} {field_name}")

    def _post_with_rate_limit(self, content: str) -> requests.Response:
        with self._lock:
            wait_seconds = self._next_allowed_at - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            resp = requests.post(
                self.url,
                json={"text": content},
                timeout=self.timeout_seconds,
            )
            self._next_allowed_at = time.monotonic() + self.min_interval_seconds
            return resp

    def _defer_requests(self, delay_seconds: float) -> None:
        with self._lock:
            self._next_allowed_at = max(
                self._next_allowed_at,
                time.monotonic() + delay_seconds,
            )

    def _retry_delay_seconds(self, resp: requests.Response, attempt: int) -> float:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                retry_after_date = parsedate_to_datetime(retry_after)
                if retry_after_date.tzinfo is None:
                    retry_after_date = retry_after_date.replace(tzinfo=timezone.utc)
                return max(0.0, retry_after_date.timestamp() - time.time())

        return float(self.retry_base_seconds * (2 ** (attempt - 1)))


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


def process_single_item(
    chain,
    repair_chain,
    item: Dict,
    language: str,
    max_ai_attempts: int,
    sensitive_checker: SensitiveChecker,
) -> Optional[Dict]:
    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                code_info["code_stars"] = data.get("stargazers_count", 0)
                code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            elif resp.status_code not in {403, 404}:
                resp.raise_for_status()
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
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

    # 检查 summary 字段
    if sensitive_checker.is_sensitive(summary, item_id, "summary"):
        return None

    # 检测代码可用性
    code_info = check_github_code(summary)
    if code_info:
        item.update(code_info)

    item['AI'] = generate_ai_fields(chain, repair_chain, item, language, max_ai_attempts)

    # 检查 AI 生成字段，合并为一次请求以避免触发接口限流。
    ai_content = "\n".join(str(v) for v in item.get("AI", {}).values())
    if sensitive_checker.is_sensitive(ai_content, item_id, "AI"):
        return None
    return item


def process_all_items(
    data: List[Dict],
    model_name: str,
    language: str,
    max_workers: int,
    max_ai_attempts: int,
) -> List[Optional[Dict]]:
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
    sensitive_checker = SensitiveChecker()
    
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
                sensitive_checker,
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
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
