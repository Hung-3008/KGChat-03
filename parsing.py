import re
import json


def parse_with_chunks(filepath):
    document = {"title": "", "chunk": new_chunk("chunk_DOC"), "chapters": []}
    current_chapter = None
    current_section = None
    current_subsection = None
    current_parent = document

    pending_img_srcs = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Match headings with 1 or more '#' (not limited to 6)
        heading = re.match(r'^(#+)\s+(.+)', stripped)
        if heading:
            flush_pending_images(pending_img_srcs, current_parent)
            level = len(heading.group(1))
            title = heading.group(2)

            # Cap level at 6 for any heading with 7+ hashes
            if level > 6:
                level = 6

            if level == 1:
                document["title"] = title
                current_parent = document
            elif level == 2:
                # Check if this is a non-structural heading (Table, Recommendation, etc.)
                # These should NOT create new chapters or reset the section hierarchy
                if re.match(r'^(Table\s+\d|Recommendation|Recommendations$|Tables of)', title):
                    # Treat as content under current parent — don't create a new chapter
                    current_parent["chunk"]["content"] += stripped + "\n"
                    i += 1
                    continue
                else:
                    current_chapter = make_node(title=title, chunk_name=f"chunk_CH_{title}")
                    document["chapters"].append(current_chapter)
                    current_section = current_subsection = None
                    current_parent = current_chapter
            elif level == 3:
                m = re.match(r'(\d+)\.\s+(.+)', title)
                if m:
                    current_section = make_node(title=m.group(2), section_no=m.group(1), chunk_name=f"chunk_SEC_{m.group(1)}")
                    if current_chapter:
                        current_chapter["sections"].append(current_section)
                    current_subsection = None
                    current_parent = current_section
            elif level == 4:
                # Handle combined titles (from malformed 7-hash headings):
                # e.g. "5.9. Endomyocardial and pericardial biopsy 5.9.1. Endomyocardial biopsy"
                combined = re.match(r'([\d.]+)\.\s+(.+?)\s+([\d.]+)\.\s+(.+)', title)
                if combined and current_section:
                    # Create the parent subsection (e.g. 5.9)
                    current_subsection = make_node(title=combined.group(2), section_no=combined.group(1), chunk_name=f"chunk_SS_{combined.group(1)}")
                    current_section["subsections"].append(current_subsection)
                    # Create the child sub-subsection (e.g. 5.9.1)
                    sub_sub = make_node(title=combined.group(4), section_no=combined.group(3), chunk_name=f"chunk_SSS_{combined.group(3)}")
                    current_subsection["subsections"].append(sub_sub)
                    current_parent = sub_sub
                else:
                    m = re.match(r'([\d.]+)\.\s+(.+)', title)
                    if m and current_section:
                        current_subsection = make_node(title=m.group(2), section_no=m.group(1), chunk_name=f"chunk_SS_{m.group(1)}")
                        current_section["subsections"].append(current_subsection)
                        current_parent = current_subsection
            elif level == 5:
                m = re.match(r'([\d.]+)\.\s+(.+)', title)
                if m and current_section and current_subsection:
                    sub_sub = make_node(title=m.group(2), section_no=m.group(1), chunk_name=f"chunk_SSS_{m.group(1)}")
                    current_subsection["subsections"].append(sub_sub)
                    current_parent = sub_sub
            elif level == 6:
                m = re.match(r'([\d.]+)\.\s+(.+)', title)
                if m and current_subsection:
                    sub_sub_sub = make_node(title=m.group(2), section_no=m.group(1), chunk_name=f"chunk_SSSS_{m.group(1)}")
                    current_subsection["subsections"].append(sub_sub_sub)
                    current_parent = sub_sub_sub
            i += 1
            continue

        table_title_match = re.match(r'<div[^>]*>(Table\s+\d+.+?)</div>', stripped)
        if table_title_match:
            table_name = table_title_match.group(1).strip()
            j = i + 1
            table_html = ""
            while j < len(lines):
                if '<table' in lines[j]:
                    table_html = lines[j].strip()
                    break
                j += 1

            table_images = re.findall(r'<img\s+src="([^"]+)"', table_html)

            current_parent["chunk"]["tables"].append({
                "type": "table",
                "name": table_name,
                "content": table_html,
                "images": table_images
            })
            i = (j + 1) if table_html else (i + 1)
            continue

        img_match = re.match(r'<div[^>]*><img\s+src="([^"]+)"[^>]*/>\s*</div>', stripped)
        if img_match:
            pending_img_srcs.append(img_match.group(1))
            i += 1
            continue

        fig_match = re.match(r'<div[^>]*>(Figure\s+\d+.+?)</div>', stripped)
        if fig_match and pending_img_srcs:
            current_parent["chunk"]["images"].append({
                "type": "image",
                "images": list(pending_img_srcs),
                "caption": fig_match.group(1).strip()
            })
            pending_img_srcs.clear()
            i += 1
            continue

        if stripped and not stripped.startswith('<table'):
            current_parent["chunk"]["content"] += line

        i += 1

    flush_pending_images(pending_img_srcs, current_parent)
    return document


def new_chunk(name=""):
    return {"name": name, "content": "", "tables": [], "images": []}


def make_node(title, chapter_no=None, section_no=None, chunk_name=""):
    node = {
        "title": title,
        "chunk": new_chunk(chunk_name),
        "sections": [],
        "subsections": []
    }
    if chapter_no is not None:
        node["chapter_no"] = chapter_no
    if section_no is not None:
        node["section_no"] = section_no
    return node


def flush_pending_images(pending, parent):
    for src in pending:
        parent["chunk"]["images"].append({
            "type": "image", "images": [src], "caption": ""
        })
    pending.clear()


if __name__ == "__main__":
    filepath = "output_markdown/ehaf192_merged.md"
    document = parse_with_chunks(filepath)
    output_path = "parsed_structure_final.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

