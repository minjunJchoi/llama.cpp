## pdf -> xml -> split -> embedding

from grobid_client.grobid_client import GrobidClient

# if __name__ == "__main__":
client = GrobidClient(config_path="./papers/config.json")
client.process("processFulltextDocument", "./papers/pdf", output="./papers/xml/", consolidate_citations=True, tei_coordinates=True, force=True)
# client.process("processFulltextDocument", "./papers/pdf", output="./papers/xml/")
# 


# # rsp = client.process_pdf(service_name, pdf_file, 
# #                          generateIDs=True, 
# #                          consolidate_header=True, 
# #                          consolidate_citations=True, 
# #                          include_raw_citations=True, 
# #                          include_raw_affiliations=True, 
# #                          tei_coordinates=True, 
# #                          segment_sentences=True)




# from langchain_community.document_loaders.generic import GenericLoader
# from langchain_community.document_loaders.parsers import GrobidParser

# loader = GenericLoader.from_filesystem(
#     "./papers/",
#     glob="*",
#     suffixes=[".pdf"],
#     parser=GrobidParser(segment_sentences=False),
# )
# docs = loader.load()

# print(docs[3].page_content)
# print(docs[3].metadata)



# from langchain_community.document_loaders.generic import GenericLoader
# from langchain_community.document_loaders.parsers import GrobidParser

# #Produce chunks from article paragraphs: if segment_sentences=False
# #Produce chunks from article sentences: if segment_sentences=True
# loader = GenericLoader.from_filesystem(
#     "/Users/mjchoi/Work/codes/llama.cpp/minjun/papers/pdf/",
#     glob="*",
#     suffixes=[".pdf"],
#     show_progress=True,
#     parser=GrobidParser(segment_sentences=True)
# )

# docs = loader.load()

# # print(docs[3].page_content)
# # print(docs[3].page_content)


