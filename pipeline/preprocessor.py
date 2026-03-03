"""
Text Preprocessor Module for Legal PDF Documents

Cleans and normalizes extracted legal PDF text for summarization pipeline.
Optimized for Indian court judgments with section classification and citation extraction.

Main Features:
- Page classification (metadata, facts, issues, arguments, reasoning, etc.)
- Smart header/footer removal
- Citation extraction and normalization
- Section-based text organization
- Summarization-ready output generation

Usage:
    # As a library
    from preprocessor import preprocess, get_summarization_text, get_sections
    
    result = preprocess(extracted_document)
    text = get_summarization_text(extracted_document)
    sections = get_sections(extracted_document)
    
    # Command line (interactive)
    python preprocessor.py
    
    # Command line (with arguments)
    python preprocessor.py path/to/case.json -o output/case.json
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter, OrderedDict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default directory where processed files are stored
# Resolves to <project_root>/data/processed/ regardless of cwd
PROCESSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"

# Module exports
__all__ = [
    'TextPreprocessor',
    'PageType',
    'preprocess',
    'preprocess_file',
    'preprocess_directory',
    'get_summarization_text',
    'get_sections',
    'get_clean_text',
    'get_metadata',
    'get_citations',
]


class PageType(Enum):
    """Classification of page types in legal judgments with relevance weights."""
    METADATA = "metadata"       # Case info, bench, citations - NOT for summarization
    HEADNOTE = "headnote"       # Summary/holding - HIGH weight
    FACTS = "facts"             # Case facts - MEDIUM weight
    ISSUES = "issues"           # Legal issues - HIGH weight
    ARGUMENTS = "arguments"     # Counsel arguments - MEDIUM weight
    REASONING = "reasoning"     # Main judgment reasoning - HIGHEST weight
    DISSENT = "dissent"         # Dissenting opinion - LOW weight
    ORDER = "order"             # Final order/decision - MEDIUM weight
    UNKNOWN = "unknown"


class PatternMatcher:
    """Handles all regex pattern matching for text cleaning."""
    
    def __init__(self):
        self.footer_patterns = [
            r'Indian Kanoon\s*-\s*https?://indiankanoon\.org/doc/\d+/?\s*\d*',
            r'\n\s*\d{1,3}\s*$',
        ]
        
        self.citation_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(\[?\d{4}\]?\s*[A-Z\.]+\s*\d+\)',
            r'\d{4}\s+AIR\s+\d+',
            r'\d{4}\s+SCR\s+\d+',
            r'\[\d{4}\]\s+[A-Z\.]+\s+\d+',
            r'[Ss]ection\s+\d+[A-Za-z]*(?:\s*\(\d+\)(?:\s*\([a-z]\))?)?',
            r'[Aa]rticle\s+\d+(?:\s*\(\d+\)(?:\s*\([a-z]\))?)?',
            r'(?:the\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act,?\s+\d{4}',
        ]
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.footer_regex = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) 
            for p in self.footer_patterns
        ]
        self.citation_regex = [re.compile(p) for p in self.citation_patterns]


class SectionClassifier:
    """Classifies pages and sections based on legal document structure."""
    
    def __init__(self):
        self.metadata_markers = [
            'PETITIONER', 'RESPONDENT', 'DATE OF JUDGMENT', 'BENCH',
            'CITATION', 'CITATOR INFO', 'Equivalent citations',
            'Author:', 'RF ', 'E ', 'R ', 'MV ', 'F '
        ]
        
        self.section_markers = {
            PageType.HEADNOTE: [
                'HEADNOTE', 'HEAD NOTE', 'Held', 'HELD', 'holding'
            ],
            PageType.FACTS: [
                'FACTS', 'FACTS OF THE CASE', 'The facts', 'factual background',
                'Brief facts', 'Facts in brief', 'factual matrix'
            ],
            PageType.ISSUES: [
                'ISSUES', 'ISSUES FOR CONSIDERATION', 'The question',
                'questions for consideration', 'Points for determination',
                'The issues', 'The point for consideration'
            ],
            PageType.ARGUMENTS: [
                'ARGUMENTS', 'SUBMISSIONS', 'Learned counsel',
                'It was argued', 'It is contended', 'It was submitted',
                'counsel for the appellant', 'counsel for the respondent',
                'Shri', 'Sri', 'Mr.'
            ],
            PageType.REASONING: [
                'JUDGMENT', 'ORIGINAL JURISDICTION', 'The judgment',
                'We have heard', 'I have considered', 'ANALYSIS',
                'We have carefully considered', 'Having heard',
                'I have given my anxious consideration', 'In my view'
            ],
            PageType.DISSENT: [
                'dissenting', 'DISSENT', 'DISSENTING OPINION',
                'I am unable to agree', 'With respect, I disagree',
                'delivered a separate', 'MINORITY VIEW'
            ],
            PageType.ORDER: [
                'ORDER', 'FINAL ORDER', 'Appeal allowed', 'Appeal dismissed',
                'Petition allowed', 'Petition dismissed', 'disposed of',
                'The appeal is', 'writ petition is'
            ]
        }
    
    def classify_page(self, text: str, page_num: int, total_pages: int) -> PageType:
        """Classify a page based on its content and position."""
        text_lower = text.lower()
        
        # Early pages are often metadata
        if page_num < 3:
            metadata_count = sum(
                1 for marker in self.metadata_markers 
                if marker.lower() in text_lower
            )
            if metadata_count >= 2:
                return PageType.METADATA
        
        # Check for section markers
        for page_type, markers in self.section_markers.items():
            if any(marker.lower() in text_lower for marker in markers):
                return page_type
        
        return PageType.UNKNOWN


class TextCleaner:
    """Handles all text cleaning operations."""
    
    def __init__(self, pattern_matcher: PatternMatcher):
        self.pattern_matcher = pattern_matcher
    
    def detect_repeated_header(self, pages: List[Dict]) -> Optional[str]:
        """Detect case title that repeats across pages."""
        if len(pages) < 2:
            return None
        
        first_lines = [
            page.get('text', '').strip().split('\n')[0].strip()
            for page in pages
            if page.get('text', '').strip()
        ]
        
        if not first_lines:
            return None
        
        counter = Counter(first_lines)
        most_common, count = counter.most_common(1)[0]
        
        # Must appear in at least half the pages and be substantial
        if count >= len(pages) // 2 and len(most_common) > 20:
            return most_common
        
        return None
    
    def remove_repeated_headers(self, text: str, header: str) -> str:
        """Remove repeated header from text."""
        if not header:
            return text
        return '\n'.join(
            line for line in text.split('\n') 
            if line.strip() != header.strip()
        )
    
    def remove_footers(self, text: str) -> Tuple[str, int]:
        """Remove footer patterns from text."""
        footers_removed = 0
        for pattern in self.pattern_matcher.footer_regex:
            matches = pattern.findall(text)
            footers_removed += len(matches)
            text = pattern.sub('', text)
        return text, footers_removed
    
    def remove_page_numbers(self, text: str) -> str:
        """Remove standalone page numbers."""
        lines = text.split('\n')
        cleaned = [
            line for line in lines
            if not (line.strip().isdigit() and len(line.strip()) <= 3)
        ]
        return '\n'.join(cleaned)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        # Remove carriage returns
        text = text.replace('\r', '')
        # Normalize spaces within lines
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize multiple newlines to double newlines (paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def clean_basic(self, text: str, header: Optional[str] = None) -> str:
        """Apply basic cleaning operations."""
        text = self.remove_repeated_headers(text, header) # type: ignore
        text, _ = self.remove_footers(text)
        text = self.remove_page_numbers(text)
        text = self.normalize_whitespace(text)
        return text


class CitationExtractor:
    """Extracts and manages legal citations."""
    
    def __init__(self, pattern_matcher: PatternMatcher):
        self.pattern_matcher = pattern_matcher
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract all citations from text."""
        citations = []
        for pattern in self.pattern_matcher.citation_regex:
            matches = pattern.findall(text)
            if isinstance(matches[0] if matches else None, tuple):
                citations.extend(match[0] if match else match for match in matches)
            else:
                citations.extend(matches)
        return citations
    
    def normalize_citation(self, citation: str) -> str:
        """Normalize a citation string."""
        citation = ' '.join(citation.split())
        citation = re.sub(r'\s*\(\s*', ' (', citation)
        citation = re.sub(r'\s*\)\s*', ') ', citation)
        return citation.strip()
    
    def deduplicate_citations(self, citations: List[str]) -> List[str]:
        """Remove duplicate citations while preserving order."""
        seen = set()
        unique = []
        for citation in citations:
            normalized = self.normalize_citation(citation)
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                unique.append(normalized)
        return unique


class TextPreprocessor:
    """
    Main preprocessor for legal document text.
    Coordinates all cleaning, classification, and extraction operations.
    """
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.classifier = SectionClassifier()
        self.cleaner = TextCleaner(self.pattern_matcher)
        self.citation_extractor = CitationExtractor(self.pattern_matcher)
    
    def classify_pages(self, pages: List[Dict]) -> List[Dict]:
        """Classify all pages with their types."""
        total_pages = len(pages)
        classified = []
        
        for i, page in enumerate(pages):
            text = page.get('text', '')
            page_type = self.classifier.classify_page(text, i, total_pages)
            
            classified.append({
                **page,
                'page_type': page_type.value,
                'page_index': i
            })
        
        return classified
    
    def merge_sections(self, pages: List[Dict]) -> Dict[str, Any]:
        """Merge pages into logical sections."""
        sections = {}
        current_section = None
        current_text = []
        
        for page in pages:
            page_type = PageType(page['page_type'])
            
            # Start new section or continue current
            if page_type != PageType.UNKNOWN and page_type != PageType.METADATA:
                if current_section and current_text:
                    # Save previous section
                    section_name = current_section.value
                    if section_name not in sections:
                        sections[section_name] = {
                            'text': '',
                            'page_range': []
                        }
                    sections[section_name]['text'] += '\n\n' + '\n'.join(current_text)
                    sections[section_name]['page_range'].append(page['page_index'])
                
                current_section = page_type
                current_text = [page.get('text', '')]
            elif current_section:
                current_text.append(page.get('text', ''))
        
        # Save last section
        if current_section and current_text:
            section_name = current_section.value
            if section_name not in sections:
                sections[section_name] = {
                    'text': '',
                    'page_range': []
                }
            sections[section_name]['text'] += '\n\n' + '\n'.join(current_text)
            sections[section_name]['page_range'].append(page['page_index'])
        
        # Clean section texts
        for section_name in sections:
            sections[section_name]['text'] = self.cleaner.normalize_whitespace(
                sections[section_name]['text']
            )
        
        return sections
    
    def generate_summarization_text(self, sections: Dict[str, Any]) -> str:
        """Generate text optimized for summarization with section markers."""
        # Define priority order for summarization
        priority_order = [
            'headnote',
            'facts', 
            'issues',
            'arguments',
            'reasoning',
            'order'
        ]
        
        parts = []
        for section_name in priority_order:
            if section_name in sections:
                text = sections[section_name].get('text', '').strip()
                if text:
                    marker = f"[{section_name.upper()}]"
                    parts.append(f"{marker}\n{text}")
        
        return '\n\n'.join(parts)
    
    def clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing pipeline for a document.
        
        Args:
            document: Dict with 'pages' key containing list of page dicts
        
        Returns:
            Dict with processed results including sections, citations, and summary text
        """
        pages = document.get('pages', [])
        if not pages:
            logger.warning("No pages found in document")
            return {'error': 'No pages found'}
        
        # Detect and clean repeated headers
        repeated_header = self.cleaner.detect_repeated_header(pages)
        
        # Clean each page
        cleaned_pages = []
        for page in pages:
            text = page.get('text', '')
            cleaned_text = self.cleaner.clean_basic(text, repeated_header)
            cleaned_pages.append({
                **page,
                'text': cleaned_text
            })
        
        # Classify pages
        classified_pages = self.classify_pages(cleaned_pages)
        
        # Extract all citations
        all_text = ' '.join(p.get('text', '') for p in classified_pages)
        citations = self.citation_extractor.extract_citations(all_text)
        unique_citations = self.citation_extractor.deduplicate_citations(citations)
        
        # Merge into sections
        sections = self.merge_sections(classified_pages)
        
        # Generate summarization-ready text
        summarization_text = self.generate_summarization_text(sections)
        
        # Separate metadata
        metadata_pages = [
            p for p in classified_pages 
            if p['page_type'] == PageType.METADATA.value
        ]
        metadata_text = '\n\n'.join(p.get('text', '') for p in metadata_pages)
        
        return {
            'filename': document.get('filename', 'unknown'),
            'total_pages': len(pages),
            'sections': sections,
            'summarization_input': {
                'text': summarization_text,
                'word_count': len(summarization_text.split())
            },
            'metadata': {
                'text': metadata_text,
                'repeated_header': repeated_header
            },
            'citations': {
                'total_found': len(citations),
                'unique_citations': unique_citations
            },
            'page_classification': {
                page_type.value: sum(
                    1 for p in classified_pages 
                    if p['page_type'] == page_type.value
                )
                for page_type in PageType
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_preprocessor_instance: Optional[TextPreprocessor] = None

def _get_preprocessor() -> TextPreprocessor:
    """Get or create singleton preprocessor instance."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = TextPreprocessor()
    return _preprocessor_instance


def preprocess(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a document dict (from PDF extraction).
    
    Args:
        document: Dict with 'pages' key containing extracted text
    
    Returns:
        Processed document with sections, citations, and clean text
    """
    preprocessor = _get_preprocessor()
    return preprocessor.clean_document(document)


def preprocess_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Process a single JSON file and save the result to disk.

    Args:
        input_path: Path to input JSON file
        output_path: Output path for results.  When omitted, the processed
                     file is saved to ``PROCESSED_DIR`` (data/processed/)
                     using the same filename as the input.

    Returns:
        Processed document dict
    """
    input_path = Path(input_path)

    with open(input_path, 'r', encoding='utf-8') as f:
        document = json.load(f)

    result = preprocess(document)

    # Default to PROCESSED_DIR when no explicit output path is given
    if output_path is None:
        output_path = PROCESSED_DIR / input_path.name

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved processed file to: {output_path}")

    return result


def preprocess_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.json"
) -> List[Dict[str, Any]]:
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        pattern: File pattern to match (default: "*.json")
    
    Returns:
        List of processed document dicts
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob(pattern))
    results = []
    
    logger.info(f"Processing {len(json_files)} files from {input_dir}")
    
    for json_file in json_files:
        try:
            output_file = output_dir / json_file.name
            result = preprocess_file(json_file, output_file)
            results.append(result)
            logger.info(f"✓ {json_file.name}")
        except Exception as e:
            logger.error(f"✗ {json_file.name}: {e}")
            results.append({'filename': json_file.name, 'error': str(e)})
    
    return results


def get_summarization_text(document: Dict[str, Any]) -> str:
    """
    Get clean text ready for summarization pipeline.
    
    Args:
        document: Extracted document dict
    
    Returns:
        Summarization-ready text with section markers
    """
    result = preprocess(document)
    return result.get('summarization_input', {}).get('text', '')


def get_sections(document: Dict[str, Any]) -> Dict[str, str]:
    """
    Get individual section texts for legal intent retrieval.
    
    Args:
        document: Extracted document dict
    
    Returns:
        Dict mapping section names to their text content
    """
    result = preprocess(document)
    sections_data = result.get('sections', {})
    return {
        name: data.get('text', '') if isinstance(data, dict) else data
        for name, data in sections_data.items()
    }


def get_clean_text(document: Dict[str, Any], include_markers: bool = False) -> str:
    """
    Get clean text for embedding generation.
    
    Args:
        document: Extracted document dict
        include_markers: Include section markers like [FACTS], [REASONING]
    
    Returns:
        Clean text string
    """
    result = preprocess(document)
    text = result.get('summarization_input', {}).get('text', '')
    
    if not include_markers:
        # Remove section markers for cleaner embeddings
        text = re.sub(
            r'\[(?:HEADNOTE|FACTS|ISSUES|ARGUMENTS|REASONING|ORDER|DISSENT)\]\n?',
            '',
            text
        )
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
    
    return text


def get_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metadata extracted from document.
    
    Args:
        document: Extracted document dict
    
    Returns:
        Dict with metadata text and repeated header
    """
    result = preprocess(document)
    return result.get('metadata', {})


def get_citations(document: Dict[str, Any]) -> List[str]:
    """
    Get normalized, deduplicated citations.
    
    Args:
        document: Extracted document dict
    
    Returns:
        List of unique citation strings
    """
    result = preprocess(document)
    return result.get('citations', {}).get('unique_citations', [])


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """
    Interactive CLI entry point for preprocessing.
    Prompts for input path if not provided via arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess legal PDF text for summarization pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for input)
  python preprocessor.py
  
  # Single file
  python preprocessor.py case.json -o processed_case.json
  
  # Directory batch processing
  python preprocessor.py data/extracted -o data/processed
  
  # With custom pattern
  python preprocessor.py data/cases -o data/clean --pattern "supreme_*.json"
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        type=Path,
        help='Input JSON file or directory (optional - will prompt if not provided)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file or directory (optional - defaults to input_processed/)'
    )
    parser.add_argument(
        '--pattern',
        default='*.json',
        help='File pattern for directory processing (default: *.json)'
    )
    
    args = parser.parse_args()
    
    # Interactive input if not provided
    if args.input is None:
        print("\n=== Legal Text Preprocessor ===\n")
        input_str = input("Enter input path (JSON file or directory): ").strip()
        
        if not input_str:
            logger.error("No input path provided")
            return
        
        args.input = Path(input_str)
    
    # Validate input path
    if not args.input.exists():
        logger.error(f"Input path not found: {args.input}")
        return
    
    # Default output path → data/processed/
    if args.output is None:
        if args.input.is_file():
            args.output = PROCESSED_DIR / args.input.name
        else:
            args.output = PROCESSED_DIR
    
    # Process
    try:
        if args.input.is_file():
            logger.info(f"Processing file: {args.input}")
            result = preprocess_file(args.input, args.output)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ Processing complete!")
            logger.info(f"  Input:  {args.input}")
            logger.info(f"  Output: {args.output}")
            logger.info(f"  Pages:  {result.get('total_pages', 0)}")
            logger.info(f"  Words:  {result.get('summarization_input', {}).get('word_count', 0)}")
            logger.info(f"  Citations: {result.get('citations', {}).get('total_found', 0)}")
            logger.info(f"{'='*60}\n")
            
        else:
            logger.info(f"Processing directory: {args.input}")
            results = preprocess_directory(args.input, args.output, args.pattern)
            
            successful = sum(1 for r in results if 'error' not in r)
            failed = len(results) - successful
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ Batch processing complete!")
            logger.info(f"  Input:      {args.input}")
            logger.info(f"  Output:     {args.output}")
            logger.info(f"  Successful: {successful} files")
            logger.info(f"  Failed:     {failed} files")
            logger.info(f"{'='*60}\n")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return


if __name__ == '__main__':
    main()