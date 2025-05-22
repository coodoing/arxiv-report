from math import log
import os
import random
import time
import json
from venv import logger
import arxiv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from openai import OpenAI

from keywords import *
from arxiv_papers_tool import get_today_arxivpapers, get_today_arxivfile
from mlsys2025_papers_tool import get_mlsyspapers
from isca2025_papers_tool import get_iscapapers


def get_client():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


def gen_file_id():
    file_path = get_today_arxivfile()
    client = get_client()
    file_object = client.files.create(file=Path(file_path), purpose="file-extract")
    fileid =file_object.id
    return fileid


@DeprecationWarning
def analyze_papers_fileid():
    file_id = gen_file_id()
    client = get_client()
    completion = client.chat.completions.create(
        model="qwen-long",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'system', 'content': f'fileid://{file_id}'},
            {'role': 'user', 'content': '每篇论文内容理解'}
        ],
        stream=True,
        stream_options={"include_usage": True}
    )

    full_content = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content
            #print(chunk.model_dump())
    #print({full_content})
    return full_content


def analyze_papers_text(conference):

    if conference == Conference.ARXIV:
        all_papers = get_today_arxivpapers()
    elif conference == Conference.MLSYS2025:
        all_papers = get_mlsyspapers()
    elif conference == Conference.ISCA2025:
        all_papers = get_iscapapers()

    full_content = ""
    client = get_client()
    lock = threading.Lock()
    logging.info(f"开始批处理 {len(all_papers)} 篇论文")

    # def process_batch(batch_papers):
    #     nonlocal full_content
    #     completion = client.chat.completions.create(
    #         model="qwen-long-latest",
    #         messages=[
    #             {'role': 'system', 'content': 'You are a helpful assistant.'},
    #             {'role': 'system', 'content': str(batch_papers)},
    #             {'role': 'user', 'content': '对输入每篇论文内容进行理解，输出内容格式固定为：标题，摘要，论文亮点'}
    #         ],
    #         stream=True,
    #         stream_options={"include_usage": True}
    #     )
    #     temp_content = ""
    #     for chunk in completion:
    #         if chunk.choices and chunk.choices[0].delta.content:
    #             temp_content += chunk.choices[0].delta.content
    #             #print(chunk.model_dump())
    #     with lock:
    #         full_content += temp_content
    #     ##time.sleep(random.randint(1,5))
    #     logging.info(f"线程{threading.current_thread().name} 处理完成 {len(batch_papers)} 篇论文")
    #     return len(batch_papers)

    def process_batch(batch_papers):
        nonlocal full_content
        for paper in batch_papers:
            temp_content = '### [' + paper['title'] + '](' + paper['arxiv_url'] + ')' + '\n\n'
            temp_content += paper['abstract'] + '\n\n'
            temp_content += '---\n\n'
            full_content += temp_content + '\n'
        ##time.sleep(random.randint(1,5))
        logging.info(f"线程{threading.current_thread().name} 处理完成 {len(batch_papers)} 篇论文")
        return len(batch_papers)

    batch_size = 5 # 每批处理5 篇论文， 每篇论文大约200-300 tokens。保证输出内容完善
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(process_batch, [all_papers[i:i + batch_size] for i in range(0, len(all_papers), batch_size)]))

    logging.info(f"一共处理了 {sum(results)} 篇论文")

    if conference == Conference.ARXIV:
        timestamp = datetime.now().strftime('%Y%m%d')
        report_file_path = os.path.join(SAVE_DIR, f'{timestamp}_arxiv_report.md')
    elif conference == Conference.MLSYS2025:
        report_file_path = os.path.join(SAVE_DIR, f'mlsys2025_report.md')
    elif conference == Conference.ISCA2025:
        report_file_path = os.path.join(SAVE_DIR, f'isca2025_report.md')
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(full_content)


# 定时20:10执行，论文在晚8点发布
if __name__ == '__main__':
    analyzed_papers = analyze_papers_text(Conference.ARXIV)
    #analyzed_papers = analyze_papers_text(Conference.ISCA2025)