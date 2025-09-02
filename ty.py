
with open(r'C:\Users\Pro\Desktop\PROJECT\Live\myproject\myapp_data.json', 'rb') as source_file:
    raw_data = source_file.read()

try:
    text = raw_data.decode('utf-16')
except UnicodeDecodeError:
    text = raw_data.decode('utf-8-sig')

with open('cleaned_data.json', 'w', encoding='utf-8') as target_file:
    target_file.write(text)

print("✅ File re-encoded to UTF-8 successfully.")