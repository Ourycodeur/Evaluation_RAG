from utils.vector_store import VectorStoreManager
vectore_store_manager = VectorStoreManager()
#len(vectore_store_manager.document_chunks)

results = vectore_store_manager.search(
    "Quel joueur a marqué le plus de points ?",
    k=3
)

print(len(results))
print(results[:1])