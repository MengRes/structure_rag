#!/usr/bin/env python3
"""
MinerU citation linking script.
Converts citation markers in Markdown to hyperlinks pointing to the reference list.
"""

import re
import sys
from pathlib import Path


def extract_references(md_content):
    """Extract the References section; return a mapping from number to content."""
    references = {}

    # Find References section
    ref_pattern = r'^# References\s*\n(.*?)(?=\n# |\Z)'
    ref_match = re.search(ref_pattern, md_content, re.MULTILINE | re.DOTALL)

    if not ref_match:
        return references

    ref_section = ref_match.group(1)

    # Extract each reference item (numbered)
    ref_item_pattern = r'^(\d+)\.\s+(.+?)(?=\n\d+\.|\Z)'
    for match in re.finditer(ref_item_pattern, ref_section, re.MULTILINE | re.DOTALL):
        ref_num = int(match.group(1))
        ref_text = match.group(2).strip()
        # Normalize whitespace
        ref_text = re.sub(r'\s+', ' ', ref_text)
        references[ref_num] = ref_text

    return references


def extract_citations(md_content):
    """Extract all citation markers; return list of (position, numbers)."""
    citations = []

    # Match [n] or [n, n, ...]
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'

    for match in re.finditer(citation_pattern, md_content):
        start_pos = match.start()
        end_pos = match.end()
        citation_text = match.group(0)
        citation_nums = [int(x.strip()) for x in match.group(1).split(',')]

        citations.append({
            'start': start_pos,
            'end': end_pos,
            'text': citation_text,
            'numbers': citation_nums
        })

    return citations


def create_citation_links(md_content, references):
    """Turn citation markers into hyperlinks."""
    citations = extract_citations(md_content)

    # Replace from end to start to avoid offset issues
    for citation in reversed(citations):
        citation_nums = citation['numbers']

        link_parts = []
        for num in citation_nums:
            if num in references:
                link_parts.append(f'[{num}](#ref-{num})')
            else:
                link_parts.append(f'[{num}]')

        if len(link_parts) == 1:
            replacement = link_parts[0]
        else:
            inner_text = ', '.join([str(num) for num in citation_nums])
            if citation_nums[0] in references:
                replacement = f'[{inner_text}](#ref-{citation_nums[0]})'
            else:
                replacement = f'[{inner_text}]'

        md_content = (
            md_content[:citation['start']] +
            replacement +
            md_content[citation['end']:]
        )

    return md_content


def add_reference_anchors(md_content):
    """Add anchors to each reference entry."""
    ref_pattern = r'^(# References\s*\n)(.*?)(?=\n# |\Z)'

    def add_anchor(match):
        header = match.group(1)
        ref_section = match.group(2)

        ref_item_pattern = r'^(\d+)\.\s+'

        def add_id(m):
            ref_num = m.group(1)
            return f'<a id="ref-{ref_num}"></a>\n{ref_num}. '

        ref_section = re.sub(ref_item_pattern, add_id, ref_section, flags=re.MULTILINE)

        return header + ref_section

    md_content = re.sub(ref_pattern, add_anchor, md_content, flags=re.MULTILINE | re.DOTALL)

    return md_content


def process_markdown(input_file, output_file=None):
    """Process Markdown file: add citation links and reference anchors."""
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: file not found: {input_file}")
        return False

    with open(input_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    references = extract_references(md_content)
    print(f"Found {len(references)} reference(s)")

    citations = extract_citations(md_content)
    print(f"Found {len(citations)} citation marker(s)")

    md_content = create_citation_links(md_content, references)
    md_content = add_reference_anchors(md_content)

    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_linked.md"

    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Saved to: {output_path}")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python link_citations.py <input.md> [output.md]")
        print("Example: python link_citations.py 2510.06592v1.md 2510.06592v1_linked.md")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    process_markdown(input_file, output_file)
