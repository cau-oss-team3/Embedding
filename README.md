# Embedding
Repository for testing embedding

환경은 narae .venv의 폴더를 그대로 복사하여 구성하였습니다.

#####
1. Make semantic chunk from pdf file
  ```
  pdf2chunk.py learn-web-development-python-hands.pdf > RAG.txt
  ```
2. Edit ouptut text (remove useless sentence)
3. Generate embeddings
  ``` 
  embed_tool.py RAG.txt RAG_DB.pkl
  ```
