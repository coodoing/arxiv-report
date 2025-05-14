from math import log
import os
import time
import json
from venv import logger
import arxiv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from openai import OpenAI

from keywords import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUBSCRIBED_TOPICS = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.RO', 'cs.DC', 'cs.AR']
SAVE_DIR = 'arxiv_data'
IMPORTANT_ORGS = ['MIT', 'Stanford', 'Google Research']

ARXIV_API_URL = 'http://export.arxiv.org/api/query'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def get_arxiv_papers(topic, start_date):
    start_date = start_date + timedelta(days=-1)
    end_date = start_date + timedelta(days=7)
    query = f'cat:{topic} AND submittedDate:[{start_date.strftime("%Y%m%d%H%M")} TO {end_date.strftime("%Y%m%d%H%M")}]'
    logging.info(f'执行查询: {query}')
    search = arxiv.Search(
        query=query,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    client = arxiv.Client()
    papers = []
    papers_titles = []
    logging.info('开始遍历搜索结果...')
    for result in client.results(search):
        published_date = result.published.date()
        if start_date.date() <= published_date <= end_date.date():
            paper = {
                'title': result.title,
                'authors': [str(author) for author in result.authors],
                'abstract': result.summary,
                'arxiv_url': result.entry_id
            }
            if paper['title'] in papers_titles:
                continue
            papers_titles.append(paper['title'])
            papers.append(paper)
            #logging.info(f'找到符合条件的论文: {paper["title"]}')
        else:
            logging.info(f'跳过论文: {result.title}, 发表日期: {published_date}, 不在 {start_date.date()} 到 {end_date.date()} 范围内')
    logging.info(f'共找到 {len(papers)} 篇符合条件的论文。')
    return papers


def save_paper_data(papers):
    seen_titles = set()
    unique_papers = []
    for paper in papers:
        if paper['title'] not in seen_titles:
            seen_titles.add(paper['title'])
            unique_papers.append(paper)
    
    logging.info(f'过滤前paper数量: {len(unique_papers)}')
    filtered_papers = []
    for paper in unique_papers:
        if sum(keyword in paper['abstract'] for keyword in final_split_keywords_list) >= 2:
            filtered_papers.append(paper)
    logging.info(f'过滤后paper数量: {len(filtered_papers)}')    

    timestamp = datetime.now().strftime('%Y%m%d')
    file_path = os.path.join(SAVE_DIR, f'arxiv_papers_{timestamp}.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_papers, f, ensure_ascii=False, indent=4)


def get_client():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


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


def analyze_papers_text():
    all_papers = get_today_papers() 
    client = get_client()
    full_content = ""
    lock = threading.Lock()
    logging.info(f"开始批处理 {len(all_papers)} 篇论文")

    def process_batch(batch_papers):
        nonlocal full_content
        completion = client.chat.completions.create(
            model="qwen-long",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': str(batch_papers)},
                {'role': 'user', 'content': '对输入每篇论文内容进行理解，输出内容格式固定为：标题，摘要，论文亮点'}
            ],
            stream=True,
            stream_options={"include_usage": True}
        )
        temp_content = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                temp_content += chunk.choices[0].delta.content
                #print(chunk.model_dump())
        with lock:
            full_content += temp_content
        logging.info(f"当前线程处理完成 {len(batch_papers)} 篇论文")
        return len(batch_papers)

    batch_size = 20 # 假设每批处理20 篇论文， 每篇论文大约200-300 tokens
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(process_batch, [all_papers[i:i + batch_size] for i in range(0, len(all_papers), batch_size)]))

    total_processed = sum(results)
    logging.info(f"一共处理了 {total_processed} 篇论文")
    timestamp = datetime.now().strftime('%Y%m%d')
    report_file_path = os.path.join(SAVE_DIR, f'{timestamp}_arxiv_report.md')
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(full_content)


def gen_file_id():
    file_path = get_today_file()
    client = get_client()
    file_object = client.files.create(file=Path(file_path), purpose="file-extract")
    fileid =file_object.id
    return fileid


def get_today_file():
    timestamp = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(SAVE_DIR, f'arxiv_papers_{timestamp.replace('-', '')}.json')
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"文件 {file_path} 不存在")


def get_today_papers():
    latest_file = get_today_file()
    with open(latest_file, 'r', encoding='utf-8') as f:
        all_papers = json.load(f)
    logging.info(f'过滤前paper数量: {len(all_papers)}')
    return all_papers


if __name__ == '__main__':
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    all_papers = []
    for topic in SUBSCRIBED_TOPICS:
        papers = get_arxiv_papers(topic, start_date)
        all_papers.extend(papers)
    save_paper_data(all_papers)

    analyzed_papers = analyze_papers_text()