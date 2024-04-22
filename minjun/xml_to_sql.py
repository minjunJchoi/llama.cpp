from dataclasses import dataclass
from unidecode import unidecode

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import requests
import urllib3
urllib3.disable_warnings()

from difflib import SequenceMatcher

import pickle


# Data class to add notes to paper
@dataclass
class Paper:
    bib_id: str
    title: str
    authors: list
    source: str
    volume: str
    page: str
    year: str
    label: str
    doi: str
    notes: list

    def __init__(self, bib_id: str, title: str, authors: list, source: str, volume: str, page: str, year: str, doi: str, notes: list):
        self.bib_id = bib_id
        self.title = title
        self.authors = authors
        self.source = source
        self.volume = volume
        self.page = page
        self.year = year
        self.label = "\"" + ", ".join(authors) + f", {source}" + f", {year}" + "\""
        self.doi = doi
        self.notes = notes

    def __eq__(self, other):
        same_class = isinstance(other, Paper)
        
        same_paper = False

        self_list = [self.page, self.authors[0], self.volume]
        other_list = [other.page, other.authors[0], other.volume]
        if (matching_score(self_list, other_list) > 0.9*len(self_list)):
            same_paper = True

        self_list = [self.authors[0], self.title]
        other_list = [other.authors[0], other.title]
        if (matching_score(self_list, other_list) > 0.9*len(self_list)):
            same_paper = True

        return same_class and same_paper


# parsing biblStruct in xml (GROBID) 
def parse_biblStruct(bibl_struct, namespace):
    
    # check bib_id
    bib_id = bibl_struct.attrib.get('{http://www.w3.org/XML/1998/namespace}id')
    bib_id = bib_id.strip() if bib_id is not None else None

    # title
    title = bibl_struct.find('.//tei:title', namespace).text
    title = title.strip() if title is not None else None

    # authors
    if bibl_struct.find('.//tei:author/tei:persName/tei:surname', namespace) is not None:
        author_search = bibl_struct.findall('.//tei:author/tei:persName/tei:surname', namespace)[:3] 
        authors = [unidecode(author.text).strip() for author in author_search]
    else:
        authors = None

    # source
    source = bibl_struct.find('.//tei:title[@level="j"]', namespace)
    source = source.text.strip() if source is not None else None

    # volume
    volume = bibl_struct.find('.//tei:biblScope[@unit="volume"]', namespace)
    volume = volume.text.strip() if volume is not None else None

    # page number
    page_search = bibl_struct.find('.//tei:biblScope[@unit="page"]', namespace)
    if page_search is not None:
        if page_search.get('from') is not None:
            page = page_search.get('from').strip()
            if page.isdigit() == False:
                page = page + page_search.get('to')
        else:
            page = page_search.text.strip()
    else:
        page = None

    # publication year
    year_search = bibl_struct.find('.//tei:date[@type="published"]', namespace)
    year = year_search.attrib['when'].split('-')[0] if year_search is not None else None

    return bib_id, title, authors, source, volume, page, year


# update paper information: two steps (1) find or correct title from scholar.google.com (2) get others using 
# google scholar 
def update_via_gscholar(paper, verbose=False):
    query = f"""{paper.title+"," if paper.title is not None else " "} \
                {"+".join(paper.authors)+"," if paper.authors is not None else " "} \
                {paper.source+"," if paper.source is not None else " "} \
                {paper.volume+"," if paper.volume is not None else " "} \
                {paper.page+"," if paper.page is not None else " "} \
                {paper.year if paper.year is not None else " "}"""
    query = "+".join(query.split())

    # url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={query}&btnG="
    url = f"https://serpapi.com/search.json?engine=google_scholar&q={query}"
    if verbose: print(f'My Query GScholar: {url}')  

    response = requests.get(url, verify=False)
    data = response.json()
    paper.title = data['organic_results'][0]['title']
   
    # soup = BeautifulSoup(response.text, "html.parser")
    
    # title = soup.find("h3", class_="gs_rt")
    # paper.title = title.text.strip() if title is not None else None
    

    return paper

# crossref
def search_via_crossref(paper, verbose=False):

    # query = f"""{paper.title+"," if paper.title is not None else " "} \
    #             {"+".join(paper.authors)+"," if paper.authors is not None else " "} \
    #             {paper.year+"," if paper.year is not None else " "}"""
    # query = "+".join(query.split())

    query = f"""{paper.title+"," if paper.title is not None else " "} \
                {"+".join(paper.authors)+"," if paper.authors is not None else " "} \
                {paper.source+"," if paper.source is not None else " "} \
                {paper.volume+"," if paper.volume is not None else " "} \
                {paper.page+"," if paper.page is not None else " "} \
                {paper.year if paper.year is not None else " "}"""
    query = "+".join(query.split())

    number_of_items = 2

    url = f"http://api.crossref.org/works?query.bibliographic={query}&rows={number_of_items}&mailto=cmj0417@gmail.com"
    if verbose: print(f'My Query CrossRef: {url}')  

    response = requests.get(url, verify=False)
    data = response.json()

    crossref_list = []
    for i in range(number_of_items):
        subdata = data['message']['items'][i]
        item = Paper(bib_id=None, title=None, authors=[], source=None, volume=None, page=None, year=None, doi=None, notes=[])
        
        # authors; first thing to 
        try: 
            author_search = subdata['author'][:3]
            item.authors = [unidecode(author['family']).strip() for author in author_search]
        except:
            pass

        # doi number
        try:
            item.doi = subdata['DOI']
        except:
            pass
        
        # title
        try: 
            item.title = subdata['title'][0]
            item.title = item.title.strip()
        except:
            pass

        # source
        try:
            item.source = subdata['container-title'][0]
            item.source = item.source.strip()
        except:
            pass

        # volume
        try:
            item.volume = subdata['volume']
            item.volume = item.volume.strip()
        except:
            pass

        # page number
        try:
            item.page = subdata['article-number']
            item.page = item.page.strip()
        except:
            try:
                item.page = subdata['page']
                item.page = item.page.split('-')[0]
            except:
                pass

        # publication year
        try:
            item.year = subdata['published']['date-parts'][0][0]
            item.year = str(item.year).strip()
        except:
            pass

        # generate label
        item.label = "\"" + ", ".join(item.authors) + f", {item.source}" + f", {item.year}" + "\""

        # print searched item
        crossref_list.append(item)
        print(item)

    return crossref_list


# matching score  
def matching_score(one_list, other_list):
    score = 0
    for i in range(len(one_list)):
        if (one_list[i] is not None) and (other_list[i] is not None):
            s = SequenceMatcher(None, one_list[i].lower(), other_list[i].lower())
            # print(one_list[i], other_list[i], s.ratio())
            score += s.ratio()
    
    return score
    

# Data class to add notes to paper
@dataclass
class Paper:
    bib_id: str
    title: str
    authors: list
    source: str
    volume: str
    page: str
    year: str
    label: str
    doi: str
    notes: list

    def __init__(self, bib_id: str, title: str, authors: list, source: str, volume: str, page: str, year: str, doi: str, notes: list):
        self.bib_id = bib_id
        self.title = title
        self.authors = authors
        self.source = source
        self.volume = volume
        self.page = page
        self.year = year
        self.label = "\"" + ", ".join(authors) + f", {source}" + f", {year}" + "\""
        self.doi = doi
        self.notes = notes

    def __eq__(self, other):
        same_class = isinstance(other, Paper)
        
        same_paper = False

        if len(self.authors) > 0 and len(other.authors) > 0:
            self_list = [self.page, self.authors[0], self.volume]
            other_list = [other.page, other.authors[0], other.volume]
            if (matching_score(self_list, other_list) > 0.9*len(self_list)):
                same_paper = True

            self_list = [self.authors[0], self.title]
            other_list = [other.authors[0], other.title]
            if (matching_score(self_list, other_list) > 0.9*len(self_list)):
                same_paper = True

        return same_class and same_paper



#### ======================================= References in XML to SQL ======================================= ####

# paper list 
paper_list_file = 'papers/paper_list.pkl'
try: 
    with open(paper_list_file, 'rb') as file:
        paper_list = pickle.load(file)

    print(f">> paper list loaded from {paper_list_file}")
except FileNotFoundError:
    paper_list = []

    print(f"{paper_list_file} not found")

# read xml file and parse
xml_file = 'papers/xml/test_notitle.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

# Define namespace
namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

# Find all biblStruct elements and update their bibliographic information
bibl_structs = root.findall('.//tei:biblStruct', namespace)

for bibl_struct in bibl_structs:
    # # only consider biblStruct elements with bib id published in journal 
    # if (bibl_struct.attrib.get('{http://www.w3.org/XML/1998/namespace}id', None) is None) or \
    #    (bibl_struct.find('.//tei:title[@level="j"]', namespace) is None): 
    #     continue
    
    # get bibliographic information from xml(pdf) file directly
    bib_id, title, authors, source, volume, page, year = parse_biblStruct(bibl_struct, namespace)

    # generate paper instance 
    paper = Paper(bib_id=bib_id, title=title, authors=authors, source=source, \
                            volume=volume, page=page, year=year, doi=None, notes=[])

    print()
    print("bib_id:", paper.bib_id)
    print("Title:", paper.title)
    print("Authors:", paper.authors)
    print("Source:", paper.source)
    print("Volume:", paper.volume)
    print("Page:", paper.page)
    print("Year:", paper.year)
    print("Label:", paper.label)
    print("DOI:", paper.doi)
    print()

    # check if there exists same paper in paper_list
    ismember = False
    for other in paper_list:
        if paper == other:
            print(">> same paper is found in the list")
            other.bib_id = paper.bib_id
            paper = other
            ismember = True
            break

    # search through crossref API to find find DOI and more
    if ismember == False: 
        # paper = update_via_gscholar(paper, verbose=True)

        ismatch = False        
        search_list = search_via_crossref(paper, verbose=True)
        for item in search_list:
            if paper == item:
                item.bib_id = paper.bib_id
                paper = item
                print(">> paper is updated using MATCHED search result via crossref")
                ismatch = True
                break
        
        if ismatch == False:
            search_list[0].bib_id = paper.bib_id
            paper = search_list[0] # force matching with the first item
            print(">> paper is updated using FORCED match")

    print()
    print("bib_id:", paper.bib_id)
    print("Title:", paper.title)
    print("Authors:", paper.authors)
    print("Source:", paper.source)
    print("Volume:", paper.volume)
    print("Page:", paper.page)
    print("Year:", paper.year)
    print("Label:", paper.label)
    print("DOI:", paper.doi)
    print()

    # append to list
    if ismember == False:
        paper_list.append(paper)
                
    # save the paper list
    with open(paper_list_file, 'wb') as file:
        pickle.dump(paper_list, file)

        print(f"paper list saved in {paper_list_file}")

    print("=================================")
    
    # if paper.bib_id == 'b0':
    #     break

print()    
for paper in paper_list:
    print(paper)
    print()

# print(paper_list[0] == paper_list[0])



# # find notes for this paper (bib_id) in this xml file
# # bib_id == None -> use body text (original paper)
# # bib_id is not None -> use sentences citing bib_id

# if paper.bib_id == None:
#     # paper.notes.append(body_text)
#     # get abstract text 
#     abstract = root.find('.//tei:abstract', namespace)
#     if abstract is not None:
#         abstract_text = abstract.find(".//tei:p", namespace).text.strip()
#         print(abstract_text)
#         print()

#     body = root.find('.//tei:body', namespace)
#     if body is not None:
#         body_text = body.findall(".//tei:p", namespace)
#         for text in body_text:
#             print(text.text)
#         # print(body_text)
#         print()

#     # # Extract body text
#     # body = root.find(".//{http://www.tei-c.org/ns/1.0}body")
#     # body_text = body.find(".//{http://www.tei-c.org/ns/1.0}p").text.strip() if body is not None else "Body text not found"

#     # # Print the extracted texts
#     # print("Abstract:", abstract_text)
#     # print("Body Text:", body_text)










# import sqlite3

# # SQLite 데이터베이스 연결
# conn = sqlite3.connect('publications.db')
# cursor = conn.cursor()

# # 테이블 생성
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS publications (
#     xml_id TEXT PRIMARY KEY,
#     title TEXT,
#     authors TEXT,
#     source TEXT,
#     volume TEXT,
#     page TEXT,
#     published_date TEXT
# )
# ''')

# # 데이터베이스에 데이터 삽입
# for pub in publications:
#     cursor.execute('''
#     INSERT INTO publications (xml_id, title, authors, source, volume, page, published_date)
#     VALUES (?, ?, ?, ?, ?, ?, ?)
#     ''', (pub.xml_id, pub.title, ", ".join(pub.authors), pub.source, pub.volume, pub.page, pub.published_date))

# # 변경사항 저장
# conn.commit()

# # 연결 종료
# conn.close()

# print("Data saved to SQL database successfully.")













# from langchain_community.document_loaders import UnstructuredXMLLoader

# loader = UnstructuredXMLLoader(
#     "papers/xml/test.xml",
# )
# docs = loader.load()
# print(len(docs))
# # print(docs[0].page_content)



# # Transform
# from langchain_community.document_transformers import BeautifulSoupTransformer

# bs_transformer = BeautifulSoupTransformer()
# docs = bs_transformer.transform_documents(
#     docs, tags_to_extract=["p", "body"]
# )
# print(len(docs))
# print(docs[0])




# from bs4 import BeautifulSoup

# with open('papers/xml/test.xml', 'r') as f:
#     xml = f.read()
#     soup = BeautifulSoup(xml, features="xml")

#     for result in soup.find_all("author"):
#         print(result.text)



# # Only keep contents of class_to_find
# bs4_strainer = bs4.SoupStrainer(class_=(class_to_find))
# loader = WebBaseLoader(
#     web_paths=(web_path,),
#     bs_kwargs={"parse_only": bs4_strainer},
# )
# loader.requests_kwargs = {'verify':False}
# docs = loader.load()


# texts = str(soup.findAll(text=True)).replace('\\n','')

# response = requests.get(web_path, verify=False)
# # webpage parsing
# soup = BeautifulSoup(response.text, "html.parser")
# elements = soup.select_one("#prologue > dl > dd:nth-child(1) > ul > li.p_title > a")
# elements = soup.select_one("#page-content > div:nth-child(3) > div.article-content > div.article-text.wd-jnl-art-abstract.cf > p")


# elements = soup.find_all(attrs={'itemprop':'articleBody'})
# elements = soup.select('.inline-eqn')

# print("=================================")
# for element in elements:
#     print(element.text)
