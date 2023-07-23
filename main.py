import typer
from sagasift.sagaquizer.sagaquizer import Sagaquizer


def run_quiz(
    epub_file: str = typer.Option(..., help="Path to the epub file."),
    question_count_per_chunk: int = typer.Option(5, help="Number of questions per chunk."),
    max_tokens: int = typer.Option(1024, help="Maximum number of tokens."),
    temperature: float = typer.Option(1.0, help="Temperature for the AI model."),
    model: str = typer.Option("gpt-4", help="Name of the AI model to use."),
    max_chunk_count: int = typer.Option(..., help="Maximum chunk count.")
):
    quizer = Sagaquizer(
        epub_file=epub_file,
        question_count_per_chunk=question_count_per_chunk,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
    )
    quizer.load_book()
    quizer.generate_questions(max_chunk_count=max_chunk_count)
    quizer.run_quiz()


def main():
    typer.run(run_quiz)


if __name__ == "__main__":
    main()
