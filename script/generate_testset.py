from app.evaluation.dataset_loader import generate_testset
from app.llm import get_llm
from app.rag.embeddings import BGEEmbeddings


def main():
    generate_testset(
        test_size=20,
        output_path="data_base/eval/generated_testset.csv",
        llm = get_llm(),
        embeddings = BGEEmbeddings(),
    )


if __name__ == "__main__":
    main()