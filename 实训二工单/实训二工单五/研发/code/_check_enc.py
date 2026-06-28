"""Check encoding corruption"""
import sys, locale, os

print(f'Preferred encoding: {locale.getpreferredencoding()}')
print(f'FS encoding: {sys.getfilesystemencoding()}')

# The garbled path that comes from __file__ via terminal
garbled = r'C:\Users\freedom\Desktop\招股书问答智能体'
# The correct path using unicode escapes
correct = 'C:\\Users\\freedom\\Desktop\\' + '\u62db\u80a1\u4e66\u95ee\u7b54\u667a\u80fd\u4f53'

print(f'\ngarbled == correct: {garbled == correct}')
print(f'garbled: {\" \".join(hex(ord(c)) for c in garbled)}')
print(f'correct: {\" \".join(hex(ord(c)) for c in correct)}')
print(f'\nBoth exist: {os.path.exists(garbled)} / {os.path.exists(correct)}')
