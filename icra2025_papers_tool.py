import time
from bs4 import BeautifulSoup, NavigableString, Tag
import json
import requests
import re

from keywords import *


def crawler_icrapapers():
    
    urls = [
        "https://ras.papercept.net/conferences/conferences/ICRA25/program/ICRA25_ContentListWeb_1.html",
        "https://ras.papercept.net/conferences/conferences/ICRA25/program/ICRA25_ContentListWeb_2.html",
        "https://ras.papercept.net/conferences/conferences/ICRA25/program/ICRA25_ContentListWeb_3.html",
            ]

    papers = []
    for url in urls:
        extracted_papers = parse_full_page_content(url)
        print(f"Successfully extracted data for {len(extracted_papers)} papers.\n")
        
        for i, paper in enumerate(extracted_papers):
            # print(f"--- Paper {i+1} ---")
            # print(f"  Session Code: {paper.get('session_code', 'N/A')}")
            # print(f"  Session Title: {paper.get('session_title', 'N/A')}")
            # print(f"  Session Description: {paper.get('session_description', 'N/A')}")
            # print(f"  Paper ID: {paper.get('paper_id', 'N/A')}")
            # print(f"  Time: {paper.get('time', 'N/A')}")
            # print(f"  Title: {paper.get('title', 'N/A')}")
            # print(f"  Internal ID: {paper.get('internal_id', 'N/A')}")
            # print(f"  Authors:")
            # for author in paper.get('authors', []):
            #     print(f"    - {author['name']} ({author['affiliation']})")
            # print(f"  Keywords: {', '.join(paper.get('keywords', []))}")
            # print(f"  Abstract: {paper.get('abstract', 'N/A')[:150]}...") # Print first 150 chars
            # print("-" * 20 + "\n")
            paper = {
                        'title': paper.get('title', 'N/A'),
                        'abstract': paper.get('abstract', 'N/A'),
                        'authors': paper.get('authors', []),
                        'keywords': ', '.join(paper.get('keywords', [])),
                        'paper_url': ''
                    }
            papers.append(paper)    
        logging.info(f'共找到 {len(papers)} 篇符合条件的论文。')  # viewAbstract(
    return papers


def parse_single_paper_block(paper_rows_tags):
    print(f"Parsing paper block with {len(paper_rows_tags)} rows...")
    """
    Parses a list of <tr> Tag objects belonging to a single paper.
    """
    paper_data = {}
    authors = []
    abstract_text = "N/A"
    keywords_list = []
    
    if not paper_rows_tags:
        return None

    block_html = "".join(str(row) for row in paper_rows_tags)
    block_soup = BeautifulSoup(block_html, 'html.parser')

    try:
        # 1. Paper ID and Time (from the first row, which should be pHdr)
        phdr_tag = block_soup.find('tr', class_='pHdr') # Ensure we're looking within the block
        if not phdr_tag: # This should be the first row in paper_rows_tags
            # print("Error: Paper block does not start with a pHdr.")
            return None 
        
        paper_id_a_tag = phdr_tag.find('a')
        if paper_id_a_tag:
            full_id_text = paper_id_a_tag.get_text(strip=True)
            match_id = re.search(r"Paper\s+([\w.-]+)", full_id_text)
            if match_id:
                paper_data['paper_id'] = match_id.group(1)
            match_time = re.search(r"(\d{2}:\d{2}-\d{2}:\d{2})", full_id_text)
            if match_time:
                paper_data['time'] = match_time.group(1)

        # 2. Title and Internal ID
        title_span_tag = block_soup.find('span', class_='pTtl')
        if title_span_tag:
            title_a_tag = title_span_tag.find('a')
            if title_a_tag:
                paper_data['title'] = title_a_tag.get_text(strip=True)
                onclick_attr = title_a_tag.get('onclick')
                if onclick_attr:
                    match_onclick_id = re.search(r"viewAbstract\('(\d+)'\)", onclick_attr)
                    if match_onclick_id:
                        paper_data['internal_id'] = match_onclick_id.group(1)
        
        if not paper_data.get('title'):
             # print(f"Warning: No title found for paper block starting with ID {paper_data.get('paper_id','Unknown')}")
             return None # Critical info missing

        # 3. Authors and Affiliations
        # Authors are in <tr>s after the <hr class="thin"> row and before the abstract div row.
        hr_thin_row = block_soup.find('hr', class_='thin')
        current_row_for_authors = hr_thin_row.find_parent('tr') if hr_thin_row else None
        
        while current_row_for_authors:
            current_row_for_authors = current_row_for_authors.find_next_sibling('tr')
            if not current_row_for_authors:
                break

            tds = current_row_for_authors.find_all('td', recursive=False)
            
            # Author row: 2 TDs, first has an <a>, no colspan on either
            if len(tds) == 2 and tds[0].find('a') and \
               not tds[0].has_attr('colspan') and not tds[1].has_attr('colspan'):
                author_name_tag = tds[0].find('a')
                author_name = author_name_tag.get_text(strip=True)
                
                if ',' in author_name and author_name.count(',') == 1: # Reformat "Last, First"
                    parts = [p.strip() for p in author_name.split(',')]
                    if len(parts) == 2:
                        author_name = f"{parts[1]} {parts[0]}"

                affiliation = tds[1].get_text(strip=True)
                authors.append({'name': author_name, 'affiliation': affiliation})
            # Stop conditions for author search
            elif current_row_for_authors.find('div', id=lambda x: x and x.startswith('Ab')): # Abstract div found
                break 
            elif len(tds) == 1 and tds[0].has_attr('colspan'): # Row with colspan (likely abstract div's parent row)
                break
            elif not (len(tds) == 2 and tds[0].find('a')): # Doesn't look like an author row anymore
                break
        
        paper_data['authors'] = authors

        # 4. Abstract and Keywords
        abstract_div_id_to_find = f"Ab{paper_data['internal_id']}" if paper_data.get('internal_id') else None
        abstract_div = block_soup.find('div', id=abstract_div_id_to_find) if abstract_div_id_to_find else None
        
        if abstract_div:
            # Keywords
            keywords_strong_tag = abstract_div.find('strong', string=re.compile(r"Keywords:", re.IGNORECASE))
            if keywords_strong_tag:
                node_after_keywords_strong = keywords_strong_tag.next_sibling
                while node_after_keywords_strong:
                    if isinstance(node_after_keywords_strong, Tag) and node_after_keywords_strong.name == 'strong' and \
                       "Abstract:" in node_after_keywords_strong.get_text(strip=True):
                        break # Stop if we hit "Abstract:"
                    if isinstance(node_after_keywords_strong, Tag) and node_after_keywords_strong.name == 'a':
                        kw = node_after_keywords_strong.get_text(strip=True)
                        if kw: keywords_list.append(kw)
                    elif isinstance(node_after_keywords_strong, NavigableString):
                        text = str(node_after_keywords_strong).strip()
                        if text and text != ',':
                            parts = [k.strip() for k in text.split(',') if k.strip()]
                            keywords_list.extend(p for p in parts if p)
                    node_after_keywords_strong = node_after_keywords_strong.next_sibling
            
            # Abstract
            abstract_strong_tag = abstract_div.find('strong', string=re.compile(r"Abstract:", re.IGNORECASE))
            if abstract_strong_tag:
                node_after_abstract_strong = abstract_strong_tag.next_sibling
                temp_abstract_parts = []
                while node_after_abstract_strong:
                    if isinstance(node_after_abstract_strong, NavigableString):
                        temp_abstract_parts.append(str(node_after_abstract_strong).strip())
                    elif isinstance(node_after_abstract_strong, Tag):
                        # If abstract text is wrapped in <p> or other tags (unlikely here but for safety)
                        temp_abstract_parts.append(node_after_abstract_strong.get_text(strip=True))
                    # Check next sibling, stop if it's another structural element like a new <strong> or <br> at end of div
                    if node_after_abstract_strong.next_sibling and \
                       isinstance(node_after_abstract_strong.next_sibling, Tag) and \
                       node_after_abstract_strong.next_sibling.name in ['strong']: # Example stop condition
                        break
                    node_after_abstract_strong = node_after_abstract_strong.next_sibling
                abstract_text = " ".join(filter(None, temp_abstract_parts))


        paper_data['keywords'] = list(set(k for k in keywords_list if k)) # Unique, non-empty
        paper_data['abstract'] = abstract_text.strip()
        
    except Exception as e:
        error_id = paper_data.get('paper_id', 'Unknown')
        # print(f"Error parsing single paper block (ID: {error_id}): {e}")
        # import traceback
        # traceback.print_exc()
        return None 

    return paper_data


def parse_full_page_content(url):
    response = requests.get(url)
    if response.status_code != 200:
        return
    soup = BeautifulSoup(response.content, 'html.parser')
    html_content = soup.find_all('table', class_='trk')
    all_papers_data = []

    #print("Parsing full page content...\n" +  html_content[0].prettify() )
    # print(f"Found {len(html_content)} top-level <table> elements.")
    all_table_elements = html_content
    
    for table in all_table_elements:
        all_tr_elements = table.find_all('tr', recursive=False)
        print(f"Found {len(all_tr_elements)} top-level <tr> elements.")
        if not all_tr_elements:
            table_or_tbody = main_soup.find('tbody') or main_soup.find('table')
            if table_or_tbody:
                all_tr_elements = table_or_tbody.find_all('tr', recursive=False)

        current_paper_rows = []
        # active_session_info = {
        #     "session_code": "N/A",
        #     "session_description": "N/A",
        #     "session_title": "N/A"
        # }

        for tr_tag in all_tr_elements:
            tr_classes = tr_tag.get('class', [])
            is_shdr = 'sHdr' in tr_classes
            is_phdr = 'pHdr' in tr_classes

            if is_shdr:
                if current_paper_rows and not (current_paper_rows[0].get('class') and 'pHdr' in current_paper_rows[0].get('class',[])):
                    current_paper_rows = []

                tds = tr_tag.find_all('td', recursive=False)
                if not tds: continue

                first_td = tds[0]
                b_tag_in_first_td = first_td.find('b')

                if b_tag_in_first_td:
                    b_text = b_tag_in_first_td.get_text(strip=True)
                    
                    desc_candidate_text = ""
                    for sibling in b_tag_in_first_td.next_siblings:
                        if isinstance(sibling, NavigableString):
                            desc_candidate_text += str(sibling)
                        elif isinstance(sibling, Tag): # Less common
                            desc_candidate_text += sibling.get_text() 
                    desc_candidate_text = desc_candidate_text.replace(" ", " ").strip()
                    desc_candidate_text = ' '.join(desc_candidate_text.split()) # Normalize whitespace

                    # if re.fullmatch(r"[A-Za-z0-9]+", b_text) and len(b_text) < 10: # Likely session code
                    #     active_session_info["session_code"] = b_text
                    #     if desc_candidate_text:
                    #         active_session_info["session_description"] = desc_candidate_text
                    #     elif first_td.get_text(" ", strip=True) != b_text :
                    #         full_td_text = first_td.get_text(" ", strip=True)
                    #         active_session_info["session_description"] = full_td_text.replace(b_text, "", 1).strip()
                    # else: # Likely session title
                    #     active_session_info["session_title"] = b_text
                    #     # If description wasn't set by a code-bearing sHdr and this one has extra text
                    #     if active_session_info.get("session_description", "N/A") == "N/A" and desc_candidate_text:
                    #         active_session_info["session_description"] = desc_candidate_text
                    #     elif active_session_info.get("session_description", "N/A") == "N/A" and first_td.get_text(" ", strip=True) != b_text:
                    #         full_td_text = first_td.get_text(" ", strip=True)
                    #         active_session_info["session_description"] = full_td_text.replace(b_text, "", 1).strip()
            
            elif is_phdr:
                if current_paper_rows: # Finalize previous paper block
                    paper_info = parse_single_paper_block(current_paper_rows)
                    if paper_info and paper_info.get('title'):
                        # paper_info.update(dict(active_session_info)) # Add a copy of current session info
                        all_papers_data.append(paper_info)
                
                current_paper_rows = [tr_tag] # Start new paper block with this pHdr

            else:
                if current_paper_rows: # Only append if we've already started collecting for a paper
                    current_paper_rows.append(tr_tag)

        if current_paper_rows: # If any rows collected for the last paper
            paper_info = parse_single_paper_block(current_paper_rows)
            if paper_info and paper_info.get('title'):
                # paper_info.update(dict(active_session_info)) # Use the last known session_info
                all_papers_data.append(paper_info)
                
    return all_papers_data



def save_icrapapers(papers):
    seen_titles = set()
    unique_papers = []
    for paper in papers:
        if paper['title'] not in seen_titles:
            seen_titles.add(paper['title'])
            unique_papers.append(paper)
    
    file_path = os.path.join(SAVE_DIR, f'icra2025_papers.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(unique_papers, f, ensure_ascii=False, indent=4)


def icrapapers_handler():
    all_papers = crawler_icrapapers()
    save_icrapapers(all_papers)


def get_icrafile():
    file_path = os.path.join(SAVE_DIR, f'icra2025_papers.json')
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"文件 {file_path} 不存在")


def get_icrapapers():
    latest_file = get_icrafile()
    with open(latest_file, 'r', encoding='utf-8') as f:
        all_papers = json.load(f)
    logging.info(f'总paper数量: {len(all_papers)}')
    return all_papers


if __name__ == '__main__':
    icrapapers_handler()