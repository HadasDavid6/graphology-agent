from core.tools.rag.store import load_corpus, LexicalIndex

chunks = load_corpus('core/agents/graphology/corpus')
index = LexicalIndex(chunks)

queries = [
    'מה אומרות אותיות גדולות',
    'איך משקים פרחים',
]

for q in queries:
    hits = index.search(q, k=1)
    if hits:
        print(f'Query: {q}')
        print(f'  Best score: {hits[0][1]:.2f}\n')
    else:
        print(f'Query: {q}')
        print(f'  No results\n')
