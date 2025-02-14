import re
import time
import pdfplumber

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def segment_text(text, max_words=150):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    segments = []
    current_segment = []
    current_word_count = 0
    
    for line in lines:
        words_in_line = line.split()
        num_words = len(words_in_line)
        
        if current_word_count + num_words > max_words and current_segment:
            segments.append(" ".join(current_segment))
            current_segment = []
            current_word_count = 0
        
        current_segment.append(line)
        current_word_count += num_words

    if current_segment:
        segments.append(" ".join(current_segment))
    
    return segments

# # Example usage:
# if __name__ == "__main__":
#     pdf_path = "documents/Designing_Machine_Learning_Systems.pdf"
    
#     # Assuming extract_text() is defined in your pdf_processor module:
#     from pdf_processor import extract_text
    
#     full_text = extract_text(pdf_path)
#     segments = segment_text_by_word_count(full_text, max_words=150)
    
#     print(f"Extracted {len(segments)} segments from the textbook.")
#     for idx, seg in enumerate(segments, start=1):
#         print(f"\n--- Segment {idx} (First 200 chars) ---")
#         print(seg[:200])