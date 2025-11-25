# Stepan Martiugin https://t.me/compmtx
# requirements: pdfplumber, openai, tqdm
"""Simple DeepSeek OCR converter for pdf files"""

import asyncio
import base64
import io
import time

import pdfplumber
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)
from tqdm.asyncio import tqdm


async def process_pdf_file(
    path_to_pdf_file: str,
    client: AsyncOpenAI,
    prompt: str,
    semaphore_slots: int = 50, # To control the number of concurrent workers
    dpi: int = 300
) -> None:
    print(f'received {path_to_pdf_file}')
    images = _cut_file_to_images(file=path_to_pdf_file, dpi=dpi)

    recognized_pages = await _ocr_images(
        images=images, client=client, prompt=prompt, semaphore_slots=semaphore_slots,
    )

    _save_ocr_results(recognized_pages=recognized_pages, file=path_to_pdf_file)


def _cut_file_to_images(file: str, dpi: int) -> list[str]:
    images = []
    with pdfplumber.open(file) as pdf_doc:
        for page in pdf_doc.pages:
            image = page.to_image(resolution=dpi, antialias=True)
            with io.BytesIO() as buffer:
                image.save(buffer, format='PNG')
                images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        return images


async def _ocr_images(
    images: list[str], client: AsyncOpenAI, prompt: str, semaphore_slots: int
) -> list[str]:

    # To control the number of concurrent workers
    semaphore = asyncio.Semaphore(semaphore_slots)
    text_prompt = ChatCompletionContentPartTextParam(type='text', text=prompt)
    tasks = [
        _prompt_to_model(
            semaphore=semaphore,
            client=client,
            content=[
                text_prompt,
                ChatCompletionContentPartImageParam(
                    type='image_url', image_url={'url': f'data:image/png;base64,{image}'},
                )
            ],
        ) for image in images
    ]
    return await tqdm.gather(*tasks, desc='Processing images')


async def _prompt_to_model(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    content: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam],
) -> str | None:
    try:
        async with semaphore:
            response = await client.with_options(max_retries=3).chat.completions.create(
                model='deepseek-ai/DeepSeek-OCR',
                messages=[ChatCompletionUserMessageParam(role='user', content=content)],
                seed=0,
                temperature=0,
                extra_body={'ngram_size': 30, 'window_size': 90},
            )
            return response.choices[0].message.content
    except Exception as error:
        print(f'error {error}')
        return None


def _save_ocr_results(recognized_pages: list[str | None], file: str) -> None:
    to_save = f'{file}.txt'
    with open(to_save, 'w', encoding='utf-8') as f:
        for page_idx, page in enumerate(recognized_pages, start=1):
            if page is None:
                print(f'page {page_idx} was not recognized')
            else:
                f.write(page)
                f.write('\n\n')
        print(f'saved to {to_save}')


if __name__ == '__main__':

    async def process_pdfs():

        aclient = AsyncOpenAI(
            base_url='url',
            api_key='api_key',
        )
        path = r'path_to_folder'
        files = ('file_1.pdf', 'file_2.pdf')

        for file in files:
            start_time = time.perf_counter()
            await process_pdf_file(
                path_to_pdf_file=f'{path}\\{file}', client=aclient, prompt='<image>\nFree OCR.'
            )
            elapsed = (time.perf_counter() - start_time) / 60
            print(f'processed {file}, for {elapsed:.2f} min')

    asyncio.run(process_pdfs())
