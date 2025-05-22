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


def crawler_arxivpapers(topic, start_date):
    start_date = start_date + timedelta(days=-1)
    end_date = start_date + timedelta(days=7)
    query = f'cat:{topic} AND submittedDate:[{start_date.strftime("%Y%m%d%H%M")} TO {end_date.strftime("%Y%m%d%H%M")}]'
    logging.info(f'执行查询: {query}')
    search = arxiv.Search(
        query=query,
        max_results=500,
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


def save_arxivpapers(papers):
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


def arxivpapers_handler():
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    all_papers = []
    for topic in SUBSCRIBED_TOPICS:
        papers = crawler_arxivpapers(topic, start_date)
        all_papers.extend(papers)
    save_arxivpapers(all_papers)



def get_today_arxivfile():
    timestamp = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(SAVE_DIR, f'arxiv_papers_{timestamp.replace('-', '')}.json')
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"文件 {file_path} 不存在")


def get_today_arxivpapers():
    latest_file = get_today_arxivfile()
    with open(latest_file, 'r', encoding='utf-8') as f:
        all_papers = json.load(f)
    logging.info(f'总paper数量: {len(all_papers)}')
    return all_papers


if __name__ == '__main__':
    arxivpapers_handler()