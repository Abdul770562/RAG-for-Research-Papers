from config.settings import settings


def main():
    print("Configuration Loaded Successfully")

    print(f"Embedding Model : {settings.embedding_model}")
    print(f"Chunk Size      : {settings.chunk_size}")
    print(f"Chunk Overlap   : {settings.chunk_overlap}")
    print(f"Top K           : {settings.top_k}")
    print(f"LLM Model       : {settings.llm_model}")


if __name__ == "__main__":
    main()