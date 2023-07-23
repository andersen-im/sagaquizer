import os
from dataclasses import dataclass

import openai
from jinja2 import Template
from bs4 import BeautifulSoup
from ebooklib import epub
import json
from openai import ChatCompletion
import dotenv


dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


@dataclass
class Chapter:
    chapter_id: str
    title: str
    paragraphs: list[str]


@dataclass
class Question:
    prompt: str
    answer_id: str
    choices: dict[str, str]
    chapter_id: str


def load_chapter(chapter: epub.EpubHtml) -> list[str]:
    soup = BeautifulSoup(chapter.content, "html.parser")
    paragraphs = list()
    for paragraph in soup.find_all("p"):
        paragraph_text = paragraph.text.strip()
        paragraph_text = " ".join(paragraph_text.split())
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return paragraphs


def chunk_chapter(chapter: Chapter, min_chunk_length: int, max_chunk_length: int) -> list[str]:
    chunks: list[str] = list()
    current_chunk: list[str] = list()
    current_length: int = 0

    for paragraph in chapter.paragraphs:
        # If adding the current paragraph doesn't exceed the max_chunk_length, add it to the current chunk
        if current_length + len(paragraph) <= max_chunk_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph)
        else:
            # If the current chunk is smaller than min_chunk_length, add the paragraph to it anyway
            if current_length < min_chunk_length:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
            else:
                # Otherwise, finalize the current chunk and start a new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)

    # Append the remaining chunk if it's not empty and larger than min_chunk_length
    if current_chunk and current_length >= min_chunk_length:
        chunks.append("\n\n".join(current_chunk))

    return chunks


PROMPT_TEMPLATE = """
Write {{question_count}} multiple-choice questions for the following text from "{{book_title}}".

--- START OF TEXT ---

{{chunk}}

--- END OF TEXT ---

Each question should have 4 choices (A, B, C, D) and 1 correct answer.

Only output as a properly formatted JSON list of objects with the following structure:

```json
[
    {
        "prompt": "What is the answer to this question?",
        "choices": {
            "A": "Choice A",
            "B": "Choice B",
            "C": "Choice C",
            "D": "Choice D",
        },
        "answer_id": "B"
    },
    {
        "prompt": "What is the answer to this question?",
        "choices": {
            "A": "Choice A",
            "B": "Choice B",
            "C": "Choice C",
            "D": "Choice D",
        },
        "answer_id": "C"
    },
    ...
]
```

Remember, only output a string in the form of a JSON list of objects. Do not output anything else. 

No explanations, no comments, no other text, just the JSON string.
"""


def extract_json(candidate_json: str) -> str | None:
    # Scan through the candidate_json from end to start
    for i in range(len(candidate_json), -1, -1):
        try:
            # Try to parse the substring as JSON
            json.loads(candidate_json[:i])
            return candidate_json[:i]
        except json.JSONDecodeError:
            pass
    return None


class Sagaquizer:
    def __init__(
        self,
        epub_file: str,
        question_count_per_chunk: int = 5,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        model: str = "gpt-4",
    ):
        self.epub_file = epub_file
        self.question_count_per_chunk = question_count_per_chunk
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.book_title = "Unknown Title"
        self.chapters: list[Chapter] = list()
        self.questions: list[Question] = list()

    def generate_question(self, chunk: str, chapter_id: str) -> list[Question]:
        template = Template(PROMPT_TEMPLATE.strip())
        content = template.render(
            question_count=self.question_count_per_chunk,
            book_title=self.book_title,
            chunk=chunk,
        )
        chat_completion = ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful master teacher and quiz maker that knows a lot about books, "
                               "and you are very knowledgeable about data structures and JSON."
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        candidate_json = chat_completion.choices[0].message.content.strip()
        final_json = extract_json(candidate_json)
        if not final_json:
            return list()
        question_dicts = json.loads(final_json)
        questions = [
            Question(
                prompt=question_dict["prompt"],
                answer_id=question_dict["answer_id"],
                choices=question_dict["choices"],
                chapter_id=chapter_id,
            )
            for question_dict in question_dicts
        ]
        return questions

    def load_book(self):
        book = epub.read_epub(self.epub_file)
        self.book_title = book.title
        for chapter_ref in book.toc:
            chapter = book.get_item_with_href(chapter_ref.href)
            paragraphs = load_chapter(chapter)
            self.chapters.append(
                Chapter(
                    chapter_id=chapter_ref.href,
                    title=chapter_ref.title,
                    paragraphs=paragraphs,
                )
            )

    def generate_questions(self, max_chunk_count: int | None = None):
        chunk_count = 1
        for chapter in self.chapters:
            chunks = chunk_chapter(
                chapter=chapter,
                min_chunk_length=500,
                max_chunk_length=2000,
            )
            for chunk in chunks:
                questions = self.generate_question(
                    chunk=chunk,
                    chapter_id=chapter.chapter_id,
                )
                self.questions.extend(questions)
                chunk_count += 1
                if max_chunk_count and chunk_count > max_chunk_count:
                    return

    def run_quiz(self):
        for i, question in enumerate(self.questions):
            print(f"Question {i + 1}: {question.prompt}")
            print("Choices:")
            for choice, answer in question.choices.items():
                print(f"{choice}: {answer}")

            user_answer = input("Your answer (A, B, C, D): ").strip().upper()

            correct_answer = question.choices[question.answer_id]

            if user_answer == question.answer_id:
                print("Correct!")
            else:
                print(f"Wrong. The correct answer was {question.answer_id}: {correct_answer}")
            print("\n-----------------\n")
