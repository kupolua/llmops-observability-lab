import os
from voyageai import Client  # type: ignore[attr-defined]
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    client = Client(api_key=os.environ["VOYAGE_API_KEY"])

    texts = [
        "кот пьёт молоко",
        "кошка лакает молочко",
        "автомобиль на парковке",
    ]

    response = client.embed(
        texts,
        model="voyage-3.5",
        input_type="document",
    )

    print(f"Получили {len(response.embeddings)} векторов")
    print(f"Размерность каждого: {len(response.embeddings[0])}")
    print(f"Первые 5 чисел первого вектора: {response.embeddings[0][:5]}")
    print(f"Токенов потрачено: {response.total_tokens}")


if __name__ == "__main__":
    main()
