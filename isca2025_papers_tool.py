import time
from bs4 import BeautifulSoup
import json
import requests

from keywords import *


def crawler_iscapapers():

    response = requests.get("https://www.iscaconf.org/isca2025/program/")
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析网页内容
        soup = BeautifulSoup(response.content, 'html.parser')
        paper_divs = soup.find_all('div', class_='paper') 
        papers = []

        for pdiv in paper_divs:
            title = pdiv.find('div', class_='paper-title').get_text(strip=True)    
            # 提取摘要
            authors = pdiv.find('div', class_='paper-authors').get_text(strip=True)
            paper = {
                        'title': title,
                        'abstract': '',
                        'authors': authors,
                        'paper_url': ''
                    }
            papers.append(paper)    
        logging.info(f'共找到 {len(papers)} 篇符合条件的论文。')
        return papers
    else:
        logging.error(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None


def save_iscapapers(papers):
    seen_titles = set()
    unique_papers = []
    for paper in papers:
        if paper['title'] not in seen_titles:
            seen_titles.add(paper['title'])
            unique_papers.append(paper)
    
    file_path = os.path.join(SAVE_DIR, f'isca2025_papers.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(unique_papers, f, ensure_ascii=False, indent=4)


def iscapapers_handler():
    all_papers = crawler_iscapapers()
    save_iscapapers(all_papers)


def get_iscafile():
    file_path = os.path.join(SAVE_DIR, f'isca2025_papers.json')
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"文件 {file_path} 不存在")


def get_iscapapers():
    latest_file = get_iscafile()
    with open(latest_file, 'r', encoding='utf-8') as f:
        all_papers = json.load(f)
    logging.info(f'总paper数量: {len(all_papers)}')
    return all_papers


if __name__ == '__main__':
    iscapapers_handler()