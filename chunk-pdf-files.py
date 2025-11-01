import fitz
import os
import re
import json
from tqdm import tqdm
import random

CHAPTER_PATTERN = re.compile(r'(Chapter\s*\d+|Section\s*\d+)', re.IGNORECASE)

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = fix_hyphenation(text)
        pages.append({"page": i + 1, "text": text})
    return pages

def fix_hyphenation(text):
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\n+', '\n', text)
    return text

def split_into_paragraphs(text):
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}|\r{2,}", text) if len(p.strip()) > 20]
    return raw_paragraphs

def split_paragraph_by_wordcount(text, max_words=400):
    words = text.split()
    splits = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        splits.append(chunk)
    return splits

def merge_and_split_paragraphs(paragraphs, min_len=100, max_len=400):
    merged = []
    buffer = ""

    for para in paragraphs:
        if not buffer:
            buffer = para
        else:
            buffer += " " + para

        word_count = len(buffer.split())
        if word_count >= min_len:
            if word_count > max_len:
                splits = split_paragraph_by_wordcount(buffer, max_len)
                merged.extend(splits)
                buffer = ""
            else:
                merged.append(buffer.strip())
                buffer = ""

    if buffer:
        merged.append(buffer.strip())

    return merged

def detect_chapter(text):
    match = CHAPTER_PATTERN.search(text)
    return match.group(0) if match else "Unknown"

def extract_paragraphs(pdf_path):
    pages = extract_text_by_page(pdf_path)
    results = []
    current_chapter = "Unknown"

    for page in pages:
        chapter_match = detect_chapter(page["text"])
        if chapter_match:
            current_chapter = chapter_match

        raw_paragraphs = split_into_paragraphs(page["text"])
        processed_paragraphs = merge_and_split_paragraphs(raw_paragraphs)

        for para in processed_paragraphs:
            results.append({
                "type": "paragraph",
                "source": os.path.basename(pdf_path),
                "page": page["page"],
                "chapter": current_chapter,
                "text": para
            })

    return results

def extract_images(pdf_path, image_output_dir):
    doc = fitz.open(pdf_path)
    os.makedirs(image_output_dir, exist_ok=True)
    image_data = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        text = page.get_text("text")
        text = fix_hyphenation(text)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_index+1}_{img_index}.{image_ext}"
            image_path = os.path.join(image_output_dir, image_name)

            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            if len(text) > 800:
                start_idx = random.randint(0, len(text) - 800)
                context_text = text[start_idx:start_idx + 800]
            else:
                context_text = text

            image_data.append({
                "type": "image_pair",
                "pdf_file": os.path.basename(pdf_path),
                "page": page_index + 1,
                "image_path": image_name,
                "context_text": context_text
            })

    return image_data

def save_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    base_pth = './data/rag_data'
    parser.add_argument("--pdf_dir", type=str, default=base_pth+"/pdf_files")
    parser.add_argument("--image_dir", type=str, default=base_pth+"/chunked_pdf/images")
    parser.add_argument("--para_output", type=str, default=base_pth+"/chunked_pdf/paragraphs.jsonl")
    parser.add_argument("--image_output", type=str, default=base_pth+"/chunked_pdf/image_pairs.jsonl")
    args = parser.parse_args()

    all_paragraphs = []
    all_images = []

    pdf_files = [f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")]
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(args.pdf_dir, pdf_file)

        paras = extract_paragraphs(pdf_path)
        imgs = extract_images(pdf_path, args.image_dir)

        all_paragraphs.extend(paras)
        all_images.extend(imgs)

    save_jsonl(all_paragraphs, args.para_output)
    save_jsonl(all_images, args.image_output)

    print(f"\n✅ Success！no. paragraph: {len(all_paragraphs)}，images: {len(all_images)}")
    print(f"📄 text output：{args.para_output}")
    print(f"🖼 img output：{args.image_output} + img file save:：{args.image_dir}")
