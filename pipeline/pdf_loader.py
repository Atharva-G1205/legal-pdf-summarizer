"""
PDF Loader Module -
Handles text extraction for text based and scanned
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import pdf2image
    import numpy as np
except ImportError:
    pytesseract = None
    pdf2image = None
    Image = None
    ImageFilter = None
    ImageOps = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Handles PDF ingestion with automatic detection of PDF type and OCR fallback.
    
    Features:
    - Text-based PDF extraction using pdfplumber
    - Enhanced OCR fallback using Tesseract for scanned PDFs
    - Automatic PDF type detection
    - Per-page text extraction
    - Advanced image preprocessing for better OCR accuracy
    """
    
    def __init__(self, ocr_enabled: bool = True):
        """
        Initialize PDF loader.
        
        Args:
            ocr_enabled: Enable OCR fallback for scanned PDFs
        """
        self.ocr_enabled = ocr_enabled
        
        # Check dependencies
        if pdfplumber is None:
            raise ImportError("pdfplumber is not installed. Install with: pip install pdfplumber")
        
        if ocr_enabled and (pytesseract is None or pdf2image is None):
            logger.warning("OCR dependencies not available. Install with: pip install pytesseract pdf2image pillow")
            logger.warning("OCR fallback will be disabled.")
            self.ocr_enabled = False
    
    def extract_text_from_page(self, page) -> str:
        """
        Extract text from a pdfplumber page object.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Extracted text from the page
        """
        try:
            text = page.extract_text()
            return text if text else ""
        except Exception as e:
            logger.error(f"Error extracting text from page: {str(e)}")
            return ""
    
    def is_text_based_pdf(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """
        Detect if PDF is text-based or scanned by checking first few pages.
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample for detection
            
        Returns:
            True if PDF appears to be text-based, False otherwise
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_check = min(sample_pages, len(pdf.pages))
                text_chars = 0
                
                for i in range(pages_to_check):
                    text = self.extract_text_from_page(pdf.pages[i])
                    text_chars += len(text.strip())
                
                # If we get reasonable amount of text, it's text-based
                # Threshold: at least 100 characters per page on average
                avg_chars_per_page = text_chars / pages_to_check
                return avg_chars_per_page > 100
                
        except Exception as e:
            logger.error(f"Error detecting PDF type: {str(e)}")
            return False
    
    def extract_with_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF using pdfplumber (text-based PDFs).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with page number and extracted text
        """
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = self.extract_text_from_page(page)
                    
                    pages_data.append({
                        "page": page_num,
                        "text": text,
                        "extraction_method": "pdfplumber"
                    })
                    
            logger.info(f"Successfully extracted {len(pages_data)} pages using pdfplumber")
            
        except Exception as e:
            logger.error(f"Error extracting PDF with pdfplumber: {str(e)}")
            
        return pages_data
    
    def adaptive_threshold(self, image):
        """
        Apply adaptive thresholding to improve text contrast.
        
        Args:
            image: PIL Image object
            
        Returns:
            Thresholded PIL Image
        """
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Calculate local threshold using mean of local neighborhood
        from scipy import ndimage
        
        # Apply Gaussian filter for local average
        local_mean = ndimage.gaussian_filter(img_array.astype(float), sigma=15)
        
        # Apply adaptive threshold
        # Pixels darker than local mean are considered text (black)
        threshold_img = img_array > (local_mean - 10)  # Small offset for better results
        
        # Convert back to PIL Image
        result = Image.fromarray((threshold_img * 255).astype(np.uint8))
        
        return result
    
    def preprocess_image_for_ocr(self, image, method='adaptive'):
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            method: Preprocessing method ('adaptive', 'simple', 'aggressive')
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed (some PDFs have CMYK or other color modes)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Resize if image is too small (improves OCR accuracy)
        min_width = 2000
        if image.width < min_width:
            scale_factor = min_width / image.width
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
            logger.debug(f"Upscaled image to {new_size}")
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        if method == 'adaptive':
            # Enhance contrast first
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Apply adaptive thresholding
            try:
                image = self.adaptive_threshold(image)
            except Exception as e:
                logger.warning(f"Adaptive threshold failed, using simple method: {e}")
                # Fallback to simple thresholding
                image = ImageOps.autocontrast(image)
                threshold = 180
                image = image.point(lambda p: 255 if p > threshold else 0)
        
        elif method == 'simple':
            # Simple contrast enhancement and thresholding
            image = ImageOps.autocontrast(image, cutoff=1)
            
            # Apply sharpening
            image = image.filter(ImageFilter.SHARPEN)
            
            # Simple thresholding
            threshold = 180
            image = image.point(lambda p: 255 if p > threshold else 0)
        
        elif method == 'aggressive':
            # More aggressive preprocessing for very poor quality scans
            
            # Strong contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Denoise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Sharp enhancement
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Adaptive threshold
            try:
                image = self.adaptive_threshold(image)
            except:
                # Fallback
                image = ImageOps.autocontrast(image)
                image = image.point(lambda p: 255 if p > 170 else 0)
        
        return image
    
    def extract_with_ocr(self, pdf_path: Path, dpi: int = 300, lang: str = 'eng', 
                        preprocessing: str = 'adaptive') -> List[Dict[str, Any]]:
        """
        Extract text from PDF using OCR (scanned PDFs).
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF to image conversion (300-400 recommended)
            lang: Tesseract language code (e.g., 'eng', 'eng+hin' for English+Hindi)
            preprocessing: Preprocessing method ('adaptive', 'simple', 'aggressive')
            
        Returns:
            List of dictionaries with page number and extracted text
        """
        if not self.ocr_enabled:
            logger.error("OCR is not enabled or dependencies are missing")
            return []
        
        pages_data = []
        
        try:
            # Convert PDF to images with optimal DPI
            logger.info(f"Converting PDF to images for OCR (DPI: {dpi})...")
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='png',
                grayscale=False,  # Keep in color for better preprocessing
                use_pdftocairo=True,  # Better quality conversion
                thread_count=4  # Speed up conversion
            )
            
            # Optimized Tesseract configuration
            # PSM 6: Assume uniform block of text (better for documents)
            # OEM 1: Neural nets LSTM engine (best accuracy)
            custom_config = r'--oem 1 --psm 6'
            
            # Process each page
            for page_num, image in enumerate(images, start=1):
                logger.info(f"Processing page {page_num}/{len(images)} with OCR...")
                
                # Try preprocessing methods in order of preference
                best_text = ""
                best_confidence = 0
                
                for method in [preprocessing]:  # You can add fallback methods if needed
                    try:
                        # Preprocess image
                        processed_image = self.preprocess_image_for_ocr(image, method=method)
                        
                        # Perform OCR with configuration
                        text = pytesseract.image_to_string(
                            processed_image,
                            lang=lang,
                            config=custom_config
                        )
                        
                        # Get confidence if possible
                        try:
                            data = pytesseract.image_to_data(processed_image, lang=lang, 
                                                            config=custom_config, output_type=pytesseract.Output.DICT)
                            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        except:
                            avg_confidence = 0
                        
                        if len(text.strip()) > len(best_text.strip()) or avg_confidence > best_confidence:
                            best_text = text
                            best_confidence = avg_confidence
                            
                    except Exception as e:
                        logger.warning(f"Preprocessing method '{method}' failed: {e}")
                        continue
                
                # Clean up extracted text
                text = best_text.strip()
                
                pages_data.append({
                    "page": page_num,
                    "text": text,
                    "extraction_method": "ocr",
                    "ocr_confidence": best_confidence if best_confidence > 0 else None
                })
                
                logger.info(f"Page {page_num}: Extracted {len(text)} characters" + 
                          (f" (confidence: {best_confidence:.1f}%)" if best_confidence > 0 else ""))
            
            logger.info(f"Successfully extracted {len(pages_data)} pages using OCR")
            
        except Exception as e:
            logger.error(f"Error extracting PDF with OCR: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        return pages_data
    
    def load_pdf(self, pdf_path: Path, force_ocr: bool = False, 
                ocr_dpi: int = 300, ocr_lang: str = 'eng',
                preprocessing: str = 'adaptive') -> Dict[str, Any]:
        """
        Load and extract text from a PDF file with automatic method detection.
        
        Args:
            pdf_path: Path to PDF file
            force_ocr: Force OCR extraction even if PDF is text-based
            ocr_dpi: DPI for OCR conversion (300-400 recommended)
            ocr_lang: Tesseract language code
            preprocessing: Preprocessing method for OCR
            
        Returns:
            Dictionary containing metadata and extracted pages
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        # Detect PDF type
        if not force_ocr:
            is_text_based = self.is_text_based_pdf(pdf_path)
            logger.info(f"PDF Type: {'Text-based' if is_text_based else 'Scanned (requires OCR)'}")
        else:
            is_text_based = False
            logger.info("Forcing OCR extraction")
        
        # Extract pages
        if is_text_based:
            pages = self.extract_with_pdfplumber(pdf_path)
        else:
            if self.ocr_enabled:
                pages = self.extract_with_ocr(pdf_path, dpi=ocr_dpi, lang=ocr_lang, 
                                             preprocessing=preprocessing)
            else:
                logger.error("PDF appears to be scanned but OCR is not available")
                pages = self.extract_with_pdfplumber(pdf_path)  # Try anyway
        
        # Calculate statistics
        total_chars = sum(len(page['text']) for page in pages)
        avg_chars_per_page = total_chars / len(pages) if pages else 0
        
        # Calculate average confidence for OCR
        avg_confidence = None
        if pages and pages[0].get("extraction_method") == "ocr":
            confidences = [p.get("ocr_confidence", 0) for p in pages if p.get("ocr_confidence")]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        result = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path.absolute()),
            "num_pages": len(pages),
            "total_characters": total_chars,
            "avg_chars_per_page": avg_chars_per_page,
            "extraction_method": pages[0]["extraction_method"] if pages else "failed",
            "ocr_avg_confidence": avg_confidence,
            "pages": pages
        }
        
        logger.info(f"Extraction complete:")
        logger.info(f"  Pages: {len(pages)}")
        logger.info(f"  Total characters: {total_chars:,}")
        logger.info(f"  Avg chars/page: {avg_chars_per_page:.0f}")
        if avg_confidence:
            logger.info(f"  Avg OCR confidence: {avg_confidence:.1f}%")
        
        return result
    
    def load_directory(self, directory: Path, output_dir: Optional[Path] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Directory containing PDF files
            output_dir: Optional directory to save extracted JSON files
            **kwargs: Additional arguments to pass to load_pdf
            
        Returns:
            List of extraction results
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all PDF files
        pdf_files = list(directory.glob("**/*.pdf")) + list(directory.glob("**/*.PDF"))
        logger.info(f"\nFound {len(pdf_files)} PDF files in {directory}")
        
        if not pdf_files:
            logger.warning("No PDF files found!")
            return []
        
        # Process each PDF
        results = []
        for i, pdf_path in enumerate(pdf_files, start=1):
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")
            
            try:
                result = self.load_pdf(pdf_path, **kwargs)
                results.append(result)
                
                # Save to JSON if output directory specified
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / f"{pdf_path.stem}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved to: {output_file}")
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
                results.append({
                    "filename": pdf_path.name,
                    "filepath": str(pdf_path.absolute()),
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    def save_result(self, result: Dict[str, Any], output_path: Path):
        """
        Save extraction result to JSON file.
        
        Args:
            result: Extraction result dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved result to: {output_path}")


def main():
    """Main function with improved OCR settings"""
    loader = PDFLoader(ocr_enabled=True)
    
    project_root = Path(__file__).parent.parent
    pdf_dir = project_root / "organized_legal_dataset" / "test"
    output_dir = project_root / "data" / "extracted_text"
    output_dir.mkdir(exist_ok=True, parents=True)

    pdf_path = Path(input("Enter pdf path: "))

    print("Processing pdf with improved OCR...")

    # Use improved OCR settings
    result = loader.load_pdf(
        pdf_path,
        ocr_dpi=300,  # Can increase to 400 for better quality
        ocr_lang='eng',  # Change if needed (e.g., 'eng+hin')
        preprocessing='adaptive'  # Options: 'adaptive', 'simple', 'aggressive'
    )

    output_path = output_dir / f"{pdf_path.stem}.json"
    loader.save_result(result, output_path)

    print(f"\n✓ Processing complete!")
    print(f"  Extracted {result['num_pages']} pages")
    print(f"  Total characters: {result['total_characters']:,}")
    if result.get('ocr_avg_confidence'):
        print(f"  Average OCR confidence: {result['ocr_avg_confidence']:.1f}%")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main()