import string
import re

def is_valid_sentence(sentence):
    # Tách câu thành các từ
    words = sentence.split()
    
    # Kiểm tra độ dài của câu
    if len(words) >= 10:
        return False
    
    # Kiểm tra dấu câu
    for char in sentence:
        if char in string.punctuation and char not in [',', ';']:
            return False

    return True

def rewrite(sentence):
    # Các danh từ và placeholder
    replacements = {
        r'\bit\b': '<movie>',          # Thay thế 'it' bằng <movie>
        r'\bthat movie\b': '<movie>',  # Thay thế 'that movie' bằng <movie>
        r'\bthis movie\b': '<movie>'
    }
    
    # Thay thế các danh từ theo quy định trong replacements
    for pattern, replacement in replacements.items():
        sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
    
    return sentence

def rewrite2(sentence):
    verb_patterns = [
        r'recommend', r'suggest', r'offer', r'try', r'try out' , r"how about", r"and", r"with", r"but", r"otherwise", r"whereas", r"meanwhile", r"while",
        r'however', r'more', r'moreover', r"instead", r"without"
    ]
    
    # Kết hợp các mẫu động từ thành một biểu thức chính quy
    verb_pattern = r'|'.join(verb_patterns)
    
    # Tìm <|endoftext|> ở cuối câu và động từ ngay trước nó
    match = re.search(rf'({verb_pattern})\s*<\|endoftext\|>$', sentence)
    
    if match:
        # Nếu tìm thấy, thêm <movie> sau động từ và xóa <|endoftext|>
        verb = match.group(1)
        sentence = re.sub(rf'{verb}\s*<\|endoftext\|>$', f'{verb} <movie>', sentence)
    else:
        # Nếu không tìm thấy, chỉ xóa <|endoftext|> nếu nó ở cuối câu
        sentence = re.sub(r'<\|endoftext\|>$', '', sentence)
    
    # Loại bỏ khoảng trắng thừa
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    return sentence