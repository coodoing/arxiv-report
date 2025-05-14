from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import json

from keywords import *


def crawler_mlsyspapers():
    # 设置 Selenium WebDriver, Chrome 选项
    options = Options()
    options.add_argument("--headless")  # 启用无头模式
    # options.add_argument("--disable-gpu")  # 禁用 GPU 加速（在某些情况下可能需要）
    # options.add_argument("--no-sandbox")  # 禁用沙盒模式（在某些情况下可能需要）
    # options.add_argument("--disable-dev-shm-usage")  # 禁用 /dev/shm 的使用（在某些情况下可能需要）
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get("https://mlsys.org/virtual/2025/papers.html?filter=titles&layout=detail")
    # 等待页面加载（可以根据需要调整等待时间）
    time.sleep(5)
    # 获取页面源码
    html = driver.page_source
    # 关闭浏览器
    driver.quit()

    # 使用 BeautifulSoup 解析页面
    soup = BeautifulSoup(html, 'html.parser')
    paper_cards = soup.find_all('div', class_='pp-card pp-mode-detail')
    papers = []
    for card in paper_cards:
        # 提取标题
        title = card.find('h5', class_='card-title').get_text(strip=True)    
        # 提取摘要
        abstract = card.find('p', class_='card-text').get_text(strip=True)
        # 提取作者
        authors = card.find('h6', class_='card-subtitle text-muted').find_all('a')
        authors_names = [author.get_text(strip=True) for author in authors]
        url = card.find('a', class_='text-muted')['href']
        url = 'https://mlsys.org/virtual/2025/' + url
        paper = {
                    'title': title,
                    'abstract': abstract,
                    'authors': authors_names,
                    'paper_url': url
                }
        papers.append(paper)    
    logging.info(f'共找到 {len(papers)} 篇符合条件的论文。')
    return papers


def save_mlsyspapers(papers):
    seen_titles = set()
    unique_papers = []
    for paper in papers:
        if paper['title'] not in seen_titles:
            seen_titles.add(paper['title'])
            unique_papers.append(paper)
    
    file_path = os.path.join(SAVE_DIR, f'mlsys2025_papers.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(unique_papers, f, ensure_ascii=False, indent=4)


def mlsyspapers_handler():
    all_papers = crawler_mlsyspapers()
    save_mlsyspapers(all_papers)


def get_mlsysfile():
    file_path = os.path.join(SAVE_DIR, f'mlsys2025_papers.json')
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"文件 {file_path} 不存在")


def get_mlsyspapers():
    latest_file = get_mlsysfile()
    with open(latest_file, 'r', encoding='utf-8') as f:
        all_papers = json.load(f)
    logging.info(f'总paper数量: {len(all_papers)}')
    return all_papers


if __name__ == '__main__':
    mlsyspapers_handler()